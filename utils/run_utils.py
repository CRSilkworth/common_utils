from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import (
    create_function,
    run_with_expected_type,
    run_with_generator,
)
from utils.type_utils import (
    deserialize_typehint,
    get_known_types,
    describe_allowed,
    describe_json_schema,
    chunked_type_map,
)
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.gcp_utils import (
    upload_via_signed_post,
    read_from_gcs_signed_urls,
    read_from_gcs_signed_url,
    request_policy,
)
from utils.string_utils import data_to_readable_string
import logging
import copy
import json
import traceback
import inspect


def run_docs(
    doc_data: Dict[Text, Dict[Text, Any]],
    run_order: List[Text],
    token: Text,
    dash_app_url: Text,
    user_id: Text,
    calc_graph_id: Text,
    attributes: Optional[List[Text]] = None,
    with_db: bool = True,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):
    allowed_modules = get_known_types(with_db=with_db)

    not_attributes = {"full_name", "name", "subclass_str"}
    db_required = {"db", "model"}
    outputs = {}
    cleanups = []
    run_config = run_config if run_config else {}
    logging.info("Deserializing values")
    for doc_id in doc_data:
        outputs[doc_id] = {}
        for att, att_dict in doc_data[doc_id].items():
            if att in not_attributes:
                continue
            if not with_db and att in db_required:
                continue

            deserialized_value, output, _cleanups = get_value_from_att_dict(
                att_dict, with_db
            )

            if output:
                outputs[doc_id][att] = output
                continue
            else:
                outputs[doc_id][att] = {
                    "failed": False,
                    "combined_output": "",
                    "stdout_output": "",
                    "stderr_output": "",
                }
            att_dict["value"] = deserialized_value
            cleanups.extend(_cleanups)

    for doc_to_run in run_order:
        doc_full_name = doc_data[doc_to_run]["full_name"]
        logging.info(f"Preparing to run {doc_full_name}")

        skip_run = False
        attributes = (
            list(doc_data[doc_to_run].keys()) if attributes is None else attributes
        )
        # Run all the runners associate with this doc
        for att in attributes:
            if att in not_attributes:
                continue
            if not with_db and att in ("model",):
                continue

            att_dict = doc_data[doc_to_run][att]
            if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                continue

            logging.info(f"Running {att}")

            # Set all the arguments to the function to run
            runner_kwargs = {}
            for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                input_doc_dict = doc_data[input_doc_id]

                attributes_output = outputs[input_doc_id]
                for _, output in attributes_output.items():
                    if output["failed"]:
                        outputs[doc_to_run][att] = failed_output(
                            "Upstream failure from "
                            f"{input_doc_dict['full_name']}:"
                            f" {output['stderr_output']}"
                        )
                        skip_run = True
                        break
                if skip_run:
                    break

                runner_kwargs[var_name] = get_doc_object(
                    var_name, input_doc_dict, with_db=with_db
                )

            if skip_run:
                continue

            header_code = ""
            if att == "model":
                header_code = att_dict["class_def"]

            # Convert the functionv string to a callable function
            func, output = create_function(
                function_name=att_dict["function_name"],
                function_header=att_dict["function_header"],
                function_string=att_dict["function_string"],
                allowed_modules=allowed_modules,
                header_code=header_code,
                global_vars=kwargs.get("globals", {}),
            )

            outputs[doc_to_run][att] = output
            if output["failed"] or not func:
                continue

            logging.info(f"Running {doc_full_name}: {att}")
            if not att_dict.get("chunked", False):
                run_output = run_with_expected_type(
                    func, runner_kwargs, att_dict["value_type"], with_db=with_db
                )
                outputs[doc_to_run][att] = run_output

                if run_output["failed"]:
                    continue

                serialized_output = prepare_output(
                    att,
                    att_dict,
                    run_output,
                    user_id,
                    calc_graph_id,
                    doc_to_run,
                    token,
                    dash_app_url,
                    with_db,
                )
                att_dict["value"] = run_output["value"]
                del serialized_output["value"]

                outputs[doc_id][att] = serialized_output
            else:
                run_generator = run_with_generator(
                    func, runner_kwargs, att_dict["value_type"], with_db=with_db
                )
                buffer = []
                output_chunks = []
                buffer_size = 0
                chunk_file_num = 0
                definitions = None
                failed = False
                max_chunk_file_size = run_config.get("max_chunk_file_size", 1e8)

                for chunk_num, run_output_chunk in enumerate(run_generator):
                    if run_output_chunk["failed"]:
                        buffer = []
                        failed = True
                        break

                    value = run_output_chunk["value_chunk"]
                    _value, serialize_output = attempt_serialize(
                        value, chunked_type_map[att_dict["value_type"]], with_db=with_db
                    )
                    if serialize_output:
                        output_chunks.append(serialize_output)
                        continue

                    chunk_size = len(_value if _value is not None else "")

                    # If adding this chunk would exceed the limit, upload the current
                    # buffer
                    if buffer and buffer_size + chunk_size > max_chunk_file_size:
                        output_chunks.append(
                            upload_group(
                                attribute_name=att,
                                chunk_file_num=chunk_file_num,
                                buffer=buffer,
                                buffer_size=buffer_size,
                                att_dict=att_dict,
                                user_id=user_id,
                                calc_graph_id=calc_graph_id,
                                doc_id=doc_to_run,
                                token=token,
                                dash_app_url=dash_app_url,
                                with_db=with_db,
                                definitions=definitions,
                            )
                        )
                        definitions = output_chunks["definitions"]
                        chunk_file_num += 1
                        buffer = []
                        buffer_size = 0

                    # Add current chunk to buffer
                    buffer.append(run_output_chunk)
                    buffer_size += chunk_size

                # Upload remaining chunks if any
                if buffer:
                    output_chunks.append(
                        upload_group(
                            attribute_name=att,
                            chunk_file_num=chunk_file_num,
                            buffer=buffer,
                            buffer_size=buffer_size,
                            att_dict=att_dict,
                            user_id=user_id,
                            calc_graph_id=calc_graph_id,
                            doc_id=doc_to_run,
                            token=token,
                            dash_app_url=dash_app_url,
                            with_db=with_db,
                            definitions=definitions,
                        )
                    )

                output = combine_outputs(output_chunks, att_dict, failed, with_db)

                att_dict["value"] = output["value"]
                del output["value"]
                outputs[doc_id][att] = output

    logging.info("Cleaning up connections")
    # cleanup any connections
    for cleanup in cleanups:
        cleanup()

    return outputs


class DotDict(dict):
    def __init__(self, *args, var_name: Text, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_name = var_name

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.var_name}' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.var_name}' has no attribute '{key}'")


def get_doc_object(
    var_name: Text, doc_dict: Dict[Text, Dict[Text, Any]], with_db: bool = True
) -> DotDict:
    obj = {}
    for att in doc_dict:
        if isinstance(doc_dict[att], str):
            obj[att] = doc_dict[att]
        elif isinstance(doc_dict[att], dict):
            value = doc_dict[att].get("value", None)
            if inspect.isgenerator(value):
                obj[att] = value
            else:
                obj[att] = copy.deepcopy(value)
        else:
            raise ValueError(f"Unhandled attribute dict type: {type(doc_dict[att])}")

    return DotDict(obj, var_name=var_name)


def get_value_from_att_dict(att_dict: Dict[Text, Any], with_db: bool):
    local_type = deserialize_typehint(att_dict["_local_type"], with_db=with_db)
    att_dict["value_type"] = deserialize_typehint(
        att_dict["_value_type"], with_db=with_db
    )
    value, output, _cleanups = attempt_deserialize(
        att_dict["_local_rep"], local_type, with_db=with_db
    )
    size = len(att_dict["_local_rep"] if att_dict["_local_rep"] is not None else "")
    if output:
        output["size"] = size
        return value, output, _cleanups

    if att_dict.get("gcs_stored", False):
        if not att_dict.get("chunked", False):
            value = (
                read_from_gcs_signed_url(att_dict["signed_urls"][0], with_db=with_db)
                if att_dict["signed_urls"]
                else None
            )
            value, output, _cleanups = attempt_deserialize(
                value, att_dict["value_type"], with_db=with_db
            )
        else:
            value = (
                read_from_gcs_signed_urls(att_dict["signed_urls"], with_db=with_db)
                if att_dict["signed_urls"]
                else None
            )

            def deserialized_gen(generator):
                for file_content in generator:
                    file_content, output, _ = attempt_deserialize(
                        file_content, att_dict["value_type"], with_db=with_db
                    )
                    if output:
                        raise ValueError(
                            "failed to deserialize file from generator: "
                            f"{output['stderr_output']}"
                        )
                    for item in file_content:
                        yield item

            value = deserialized_gen(value)
    elif att_dict.get("model", False):
        value, output, _cleanups = attempt_deserialize(
            value, att_dict["value_type"], with_db=with_db
        )

    return value, output, _cleanups


def prepare_output(
    attribute_name,
    att_dict,
    output,
    user_id,
    calc_graph_id,
    doc_id,
    token,
    dash_app_url,
    with_db,
):
    schema = describe_allowed(output["value"], with_db=with_db)
    try:
        preview = data_to_readable_string(output["value"])
    except TypeError:
        return failed_output(
            f"Failed to create schema for output.\n{traceback.format_exc()}"
        )

    if len(preview) > 1000:
        preview = (
            preview[:100]
            + "\n...\n"
            + preview[-100:]
            + "\n\nValue too large to show. This is it's basic"
            f" format:\n\n"
            f"{json.dumps(schema, indent=2)}"
        )

    value = output["value"]
    _value, serialize_output = attempt_serialize(
        value, att_dict["value_type"], with_db=with_db
    )
    if serialize_output:
        return serialize_output

    _local_rep = _value
    size = len(_value if _value is not None else "")
    if att_dict.get("gcs_stored", False):
        policy = request_policy(
            dash_app_url,
            {
                "token": token,
                "user_id": user_id,
                "calc_graph_id": calc_graph_id,
                "doc_id": doc_id,
                "attribute_name": attribute_name,
                "version": att_dict["new_version"],
                "chunk_file_num": 0,
            },
            token=token,
            with_db=with_db,
        )
        status = upload_via_signed_post(policy, _value, with_db=with_db)
        if status not in (200, 204):
            return failed_output(
                f"Failed to upload file to gcs. Got status code {status}"
            )

        local_rep = (att_dict["bucket"], att_dict["new_version"])
        local_type = deserialize_typehint(att_dict["_local_type"], with_db=with_db)
        _local_rep, serialized_output = attempt_serialize(
            local_rep, local_type, with_db=with_db
        )
        if serialized_output:
            return serialize_output

    output["_local_rep"] = _local_rep
    output["_local_type"] = att_dict["_local_type"]
    output["_schema"] = json.dumps(schema)
    output["preview"] = preview
    output["size_delta"] = size - att_dict["value_size"]

    return output


def combine_outputs(chunk_output_files, att_dict, failed, with_db):
    output = {
        "failed": False,
        "value": None,
        "combined_output": "",
        "stdout_output": "",
        "stderr_output": "",
    }
    chunk_schemas = None
    definitions = None
    size = 0
    num_chunks = 0
    signed_urls = []
    for chunk_file_output in chunk_output_files:
        if chunk_schemas is None:
            chunk_schemas = []
        chunk_schemas.extend(chunk_file_output.get("chunk_schemas", []))

        output["combined_output"] += "\n".join(chunk_file_output["combined_outputs"])
        output["stdout_output"] += "\n".join(chunk_file_output["stdout_outputs"])
        output["stderr_output"] += "\n".join(chunk_file_output["stderr_outputs"])

        signed_urls.append(chunk_file_output.get("signed_url", None))
        definitions = chunk_file_output.get("definitions", None)

        size += chunk_file_output.get("buffer_size", 0)
        num_chunks = chunk_file_output.get("num_chunks", 0)
    if failed:
        output["value"] = None
        return output

    value = read_from_gcs_signed_urls(signed_urls, with_db=with_db)

    def deserialized_gen(generator):
        for file_content in generator:
            file_content, output, _ = attempt_deserialize(
                file_content, att_dict["value_type"], with_db=with_db
            )
            if output:
                raise ValueError(
                    "failed to deserialize file from generator: "
                    f"{output['stderr_output']}"
                )
            for item in file_content:
                yield item

    output["value"] = deserialized_gen(value)
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "x-type": "generator",
        "type": "array",
        "minItems": num_chunks,
        "maxItems": num_chunks,
        "items": chunk_schemas[0] if chunk_schemas else {},
        "definitions": definitions if definitions else [],
    }
    output["_schema"] = json.dumps(schema, indent=2)
    output["preview"] = output["_schema"]

    local_rep = (att_dict["bucket"], att_dict["new_version"])
    local_type = deserialize_typehint(att_dict["_local_type"], with_db=with_db)
    _local_rep, _ = attempt_serialize(local_rep, local_type, with_db=with_db)

    output["_local_rep"] = _local_rep

    output["size_delta"] = size - att_dict["value_size"]

    return output


def upload_group(
    attribute_name,
    chunk_file_num,
    buffer,
    buffer_size,
    att_dict,
    user_id,
    calc_graph_id,
    doc_id,
    token,
    dash_app_url,
    with_db,
    definitions,
):
    # Combine all value_chunks into one
    combined_values = [chunk["value_chunk"] for chunk in buffer]
    combined_outputs = [chunk["combined_output"] for chunk in buffer]
    stdout_outputs = [chunk["stdout_output"] for chunk in buffer]
    stderr_outputs = [chunk["stderr_output"] for chunk in buffer]
    chunk_schemas = []
    for value_chunk in combined_values:
        chunk_schema, definitions = describe_json_schema(
            value_chunk, with_db=with_db, definitions=definitions
        )
        chunk_schemas.append(chunk_schema)

    _values, serialize_output = attempt_serialize(
        combined_values, chunked_type_map[att_dict["value_type"]], with_db=with_db
    )
    if serialize_output:
        return serialize_output

    policy = request_policy(
        dash_app_url,
        {
            "token": token,
            "user_id": user_id,
            "calc_graph_id": calc_graph_id,
            "doc_id": doc_id,
            "attribute_name": attribute_name,
            "version": att_dict["new_version"],
            "chunk_file_num": chunk_file_num,
        },
        token=token,
        with_db=with_db,
    )

    status = upload_via_signed_post(policy, _values, with_db=with_db)
    if status not in (200, 204):
        return failed_output(f"Failed to upload file to gcs. Got status code {status}")

    return {
        "chunk_file_num": chunk_file_num,
        "chunk_schemas": chunk_schemas,
        "definitions": definitions,
        "url": policy["url"],
        "signed_url": policy["signed_url"],
        "combined_outputs": combined_outputs,
        "stdout_outputs": stdout_outputs,
        "stderr_outputs": stderr_outputs,
        "num_chunks": len(combined_values),
        "size": buffer_size,
    }

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
    chunked_type_map,
    describe_allowed,
)
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.preview_utils import value_to_preview
from utils.chunk_utils import combine_chunk_outputs, upload_chunk_file
from utils.value_utils import upload_serialized_value
from utils.generator_utils import generator_from_urls
from utils.gcp_utils import read_from_gcs_signed_url
import logging
import copy
import json
import inspect
import requests
import os


def run_docs(
    doc_data: Dict[Text, Dict[Text, Any]],
    docs_to_run: List[Text],
    auth_data: Dict[Text, Text],
    attributes_to_run: Optional[List[Text]] = None,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):
    allowed_modules = get_known_types()

    not_attributes = {"full_name", "name", "subclass_str"}
    outputs = {}
    cleanups = []
    failures = {}
    run_config = run_config if run_config else {}
    logging.info("Deserializing values")
    for doc_id in doc_data:
        outputs[doc_id] = {}
        failures[doc_id] = {}
        for att, att_dict in doc_data[doc_id].items():
            if att in not_attributes:
                continue

            deserialized_value, output, _cleanups = get_value_from_att_dict(att_dict)

            if output:
                outputs[doc_id][att] = output
                failures[doc_id][att] = True
                continue
            else:
                outputs[doc_id][att] = {
                    "failed": False,
                    "combined_output": "",
                    "stdout_output": "",
                    "stderr_output": "",
                }
                failures[doc_id][att] = False
            att_dict["value"] = deserialized_value
            cleanups.extend(_cleanups)

    for doc_to_run in docs_to_run:
        doc_full_name = doc_data[doc_to_run]["full_name"]
        logging.info(f"Preparing to run {doc_full_name}")

        skip_run = False
        attributes_to_run = (
            list(doc_data[doc_to_run].keys())
            if attributes_to_run is None
            else attributes_to_run
        )
        # Run all the attributes associate with this doc
        for att in attributes_to_run:
            if att in not_attributes:
                continue

            att_dict = doc_data[doc_to_run][att]
            if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                continue

            logging.info(f"Running {doc_full_name}-{att}")
            print(f"Running {doc_full_name} {att}")
            # Set all the arguments to the function to run
            runner_kwargs = {}
            for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                input_doc_dict = doc_data[input_doc_id]

                attribute_failures = failures[input_doc_id]
                for _, attribute_failed in attribute_failures.items():
                    if attribute_failed:
                        output = failed_output(
                            "Upstream failure from " f"{input_doc_dict['full_name']}"
                        )
                        failures[doc_to_run][att] = True
                        skip_run = True
                        break
                if skip_run:
                    break

                runner_kwargs[var_name] = get_doc_object(var_name, input_doc_dict)

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

            if skip_run or output["failed"] or not func:
                print(f"Skipping {doc_full_name}-{att}")
                send_output(
                    {doc_to_run: {att: output}},
                    [doc_to_run],
                    auth_data,
                    kwargs.get("caller"),
                )
                continue

            logging.info(f"Running {doc_full_name}: {att}")
            if not att_dict.get("chunked", False):
                output = run_with_expected_type(
                    func, runner_kwargs, att_dict["value_type"]
                )

                output = prepare_output(att, att_dict, output, doc_to_run, auth_data)
            else:
                run_generator = run_with_generator(
                    func, runner_kwargs, att_dict["value_type"]
                )
                buffer = []
                output_chunks = []
                buffer_size = 0
                chunk_file_num = 0
                definitions = None
                failed = False
                max_chunk_file_size = run_config.get("max_chunk_file_size", 1e8)
                for run_output_chunk in run_generator:
                    if run_output_chunk["failed"]:
                        buffer.append(run_output_chunk)
                        failed = True
                        break

                    value = run_output_chunk["value_chunk"]
                    _value, serialize_output = attempt_serialize(
                        value, chunked_type_map[att_dict["value_type"]]
                    )
                    if serialize_output:
                        buffer.append(serialize_output)
                        failed = True
                        break

                    chunk_size = len(_value if _value is not None else "")

                    # If adding this chunk would exceed the limit, upload the current
                    # buffer
                    if buffer and buffer_size + chunk_size > max_chunk_file_size:
                        output_chunks.append(
                            upload_chunk_file(
                                doc_id=doc_to_run,
                                attribute_name=att,
                                att_dict=att_dict,
                                chunk_file_num=chunk_file_num,
                                buffer=buffer,
                                buffer_size=buffer_size,
                                auth_data=auth_data,
                                definitions=definitions,
                            )
                        )
                        definitions = output_chunks[-1]["definitions"]
                        chunk_file_num += 1
                        buffer = []
                        buffer_size = 0

                    # Add current chunk to buffer
                    buffer.append(run_output_chunk)
                    buffer_size += chunk_size

                # Upload remaining chunks if any
                if buffer:
                    output_chunks.append(
                        upload_chunk_file(
                            doc_id=doc_to_run,
                            attribute_name=att,
                            att_dict=att_dict,
                            chunk_file_num=chunk_file_num,
                            buffer=buffer,
                            buffer_size=buffer_size,
                            auth_data=auth_data,
                            definitions=definitions,
                        )
                    )

                output = combine_chunk_outputs(output_chunks, att_dict, failed)
            att_dict["value"] = output["value"]
            att_dict["signed_urls"] = output["signed_urls"]
            del output["value"]

            send_output(
                {doc_to_run: {att: output}},
                [doc_to_run],
                auth_data,
                kwargs.get("caller"),
            )

    logging.info("Cleaning up connections")
    # cleanup any connections
    for cleanup in cleanups:
        cleanup()


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


def get_doc_object(var_name: Text, doc_dict: Dict[Text, Dict[Text, Any]]) -> DotDict:
    obj = {}
    for att in doc_dict:
        if isinstance(doc_dict[att], str):
            obj[att] = doc_dict[att]
        elif isinstance(doc_dict[att], dict):
            value = doc_dict[att].get("value", None)
            if inspect.isgenerator(value):
                # Recreate the generator in case it's been used.
                obj[att] = generator_from_urls(
                    doc_dict[att]["signed_urls"], doc_dict[att]["value_type"]
                )
            else:
                obj[att] = copy.deepcopy(value)
        else:
            raise ValueError(f"Unhandled attribute dict type: {type(doc_dict[att])}")

    return DotDict(obj, var_name=var_name)


def get_value_from_att_dict(att_dict: Dict[Text, Any]):
    local_type = deserialize_typehint(att_dict["_local_type"])
    att_dict["value_type"] = deserialize_typehint(att_dict["_value_type"])
    value, output, _cleanups = attempt_deserialize(att_dict["_local_rep"], local_type)
    size = len(att_dict["_local_rep"] if att_dict["_local_rep"] is not None else "")
    if output:
        output["size"] = size
        return value, output, _cleanups

    if att_dict.get("gcs_stored", False):
        if not att_dict.get("chunked", False):
            value = (
                read_from_gcs_signed_url(att_dict["signed_urls"][0])
                if att_dict["signed_urls"]
                else None
            )
            value, output, _cleanups = attempt_deserialize(
                value, att_dict["value_type"]
            )
        else:
            value = generator_from_urls(att_dict["signed_urls"], att_dict["value_type"])
    elif att_dict.get("model", False):
        value, output, _cleanups = attempt_deserialize(value, att_dict["value_type"])

    return value, output, _cleanups


def prepare_output(attribute_name, att_dict, output, doc_id, auth_data):
    preview = value_to_preview(output["value"])

    value = output["value"]
    _value, serialize_output = attempt_serialize(value, att_dict["value_type"])
    if serialize_output:
        return serialize_output

    _local_rep = _value
    size = len(_value if _value is not None else "")

    if att_dict.get("gcs_stored", False):
        policy = upload_serialized_value(
            _value, doc_id, attribute_name, att_dict["new_version"], auth_data
        )
        local_rep = (att_dict["bucket"], att_dict["new_version"])
        local_type = deserialize_typehint(att_dict["_local_type"])
        _local_rep, _ = attempt_serialize(local_rep, local_type)

        output["signed_urls"] = [policy["signed_url"]]

    output["_local_rep"] = _local_rep
    output["_local_type"] = att_dict["_local_type"]
    output["_schema"] = json.dumps(describe_allowed(output["value"]))
    output["preview"] = preview
    output["size_delta"] = size - att_dict["value_size"]

    return output


def send_output(outputs, docs_to_run, auth_data, caller):
    # Send the attribute result back to the backend
    data = {
        "docs_to_run": docs_to_run,
        "outputs": outputs,
        "caller": caller,
        "auth_data": auth_data,
        "run_completed": False,
    }
    print("-" * 10)
    print(data)
    print("-" * 10)

    requests.post(
        os.path.join(auth_data["dash_app_url"], "job-result"),
        json=data,
        headers={"Authorization": f"Bearer {auth_data['token']}"},
    )

from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import create_function, run_with_expected_type
from utils.type_utils import deserialize_typehint, get_known_types, describe_allowed
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.gcp_utils import read_from_gcs_signed_url, upload_via_signed_post
from utils.string_utils import data_to_readable_string
import logging
import copy
import json

# Define allowed modules for dynamic function execution


async def run_docs(
    doc_data: Dict[Text, Dict[Text, Any]],
    run_order: List[Text],
    attributes: Optional[List[Text]] = None,
    with_db: bool = True,
    **kwargs,
):
    allowed_modules = get_known_types(with_db=with_db)

    not_attributes = {"full_name", "name", "subclass_str"}
    db_required = {"db", "model"}
    outputs = {}
    cleanups = []

    logging.info("Deserializing values")
    for doc_id in doc_data:
        outputs[doc_id] = {}
        for att, att_dict in doc_data[doc_id].items():
            if att in not_attributes:
                continue
            if not with_db and att in db_required:
                continue

            deserialized_value, output, _cleanups = await get_value_from_att_dict(
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
            )

            outputs[doc_to_run][att] = output
            if output["failed"] or not func:
                continue

            logging.info(f"Running {doc_full_name}: {att}")

            run_output = run_with_expected_type(
                func, runner_kwargs, att_dict["value_type"], with_db=with_db
            )
            outputs[doc_to_run][att] = run_output

            if not run_output["failed"]:
                att_dict["value"] = run_output["value"]

    logging.info("Serializing values")
    for doc_id in run_order:
        doc_full_name = doc_data[doc_id]["full_name"]

        for att in attributes:
            if att in not_attributes:
                continue
            att_dict = doc_data[doc_id][att]
            output = outputs[doc_id][att]
            if output["failed"] or "value" not in output:
                continue

            output = await prepare_output(att, att_dict, output, with_db)
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
            obj[att] = copy.deepcopy(doc_dict[att].get("value", None))
        else:
            raise ValueError(f"Unhandled attribute dict type: {type(doc_dict[att])}")

    return DotDict(obj, var_name=var_name)


async def get_value_from_att_dict(att_dict: Dict[Text, Any], with_db: bool):
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
        value = await read_from_gcs_signed_url(att_dict["signed_url"], with_db=with_db)
        size = len(value if value is not None else "")
    if att_dict.get("gcs_stored", False) or att_dict.get("model", False):
        value, output, _cleanups = attempt_deserialize(
            value, att_dict["value_type"], with_db=with_db
        )

    att_dict["size"] = size
    return value, output, _cleanups


async def prepare_output(att, att_dict, output, with_db):
    schema = describe_allowed(output["value"], with_db=with_db)
    try:
        preview = data_to_readable_string(output["value"])
    except TypeError:
        preview = ""

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
        status = await upload_via_signed_post(
            att_dict["signed_post_policy"], _value, with_db=with_db
        )
        if status not in (200, 204):
            return failed_output(
                f"Failed to upload file to gcs. Got status code {status}"
            )

        local_rep = f"gs://{att_dict['bucket']}/{att_dict['new_blob_name']}"
        local_type = deserialize_typehint(att_dict["_local_type"], with_db=with_db)
        _local_rep, serialized_output = attempt_serialize(
            local_rep, local_type, with_db=with_db
        )
        if serialized_output:
            return serialize_output

    del output["value"]

    output["_local_rep"] = _local_rep
    output["_local_type"] = att_dict["_local_type"]
    output["_schema"] = json.dumps(schema)
    output["preview"] = preview
    output["size_delta"] = size - att_dict["size"]
    return output

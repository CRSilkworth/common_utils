from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import create_function, run_with_expected_type
from utils.type_utils import deserialize_typehint, get_known_types
from utils.serialize_utils import attempt_deserialize, attempt_serialize
import logging
import copy

# Define allowed modules for dynamic function execution


def run_docs(
    doc_data: Dict[Text, Dict[Text, Any]],
    run_order: List[Text],
    runner_keys: Optional[List[Text]] = None,
    with_db: bool = True,
    **kwargs,
):
    allowed_modules = get_known_types(with_db=with_db)

    outputs = {}
    cleanups = []

    logging.warning("Deserializing values")
    for doc_id in doc_data:
        outputs[doc_id] = {"attributes": {}, "runners": {}}
        for att, att_dict in doc_data[doc_id]["attributes"].items():
            if not with_db and att in ("model", "conn"):
                continue

            att_dict["type"] = deserialize_typehint(att_dict["_type"], with_db=with_db)
            deserialized_value, output, _cleanups = attempt_deserialize(
                att_dict["_value"], att_dict["type"], with_db=with_db
            )

            if output:
                outputs[doc_id]["attributes"][att] = output
                continue
            else:
                outputs[doc_id]["attributes"][att] = {
                    "failed": False,
                    "combined_output": "",
                    "stdout_output": "",
                    "stderr_output": "",
                }

            att_dict["value"] = deserialized_value
            cleanups.extend(_cleanups)

    for doc_to_run in run_order:
        doc_full_name = doc_data[doc_to_run]["attributes"]["full_name"]["value"]
        logging.warning(f"Preparing to run {doc_full_name}")

        skip_run = False

        doc_runner_keys = get_doc_runner_keys(
            runner_keys, doc_data[doc_to_run]["runners"]
        )
        # Run all the runners associate with this doc
        for runner_key in doc_runner_keys:
            if not with_db and runner_key in ("model_builder",):
                continue
            logging.warning(f"Running {runner_key}")
            runner_dict = doc_data[doc_to_run]["runners"][runner_key]
            att_dict = doc_data[doc_to_run]["attributes"][runner_dict["attribute_key"]]

            # Set all the arguments to the function to run
            runner_kwargs = {}
            for var_name, input_doc_id in runner_dict["var_name_to_id"].items():
                input_att_dict = doc_data[input_doc_id]["attributes"]

                attributes_output = outputs[input_doc_id]["attributes"]
                for att, output in attributes_output.items():
                    if output["failed"]:
                        outputs[doc_to_run]["runners"][runner_key] = failed_output(
                            "Upstream failure from "
                            f"{input_att_dict['full_name']['value']}:"
                            f" {output['stderr_output']}"
                        )
                        skip_run = True
                        break
                if skip_run:
                    break

                runner_kwargs[var_name] = get_doc_object(
                    input_att_dict, with_db=with_db
                )

            if skip_run:
                continue

            header_code = ""
            if runner_key == "model_builder":
                header_code = runner_dict["class_def"]

            # Convert the functionv string to a callable function
            func, output = create_function(
                function_name=runner_dict["function_name"],
                function_header=runner_dict["function_header"],
                function_string=runner_dict["function_string"],
                allowed_modules=allowed_modules,
                header_code=header_code,
            )

            outputs[doc_to_run]["runners"][runner_key] = output
            if output["failed"] or not func:
                continue

            logging.warning(f"Running {doc_full_name}: {runner_key}")

            run_output = run_with_expected_type(
                func, runner_kwargs, att_dict["type"], with_db=with_db
            )
            outputs[doc_to_run]["runners"][runner_key] = run_output

            if not run_output["failed"]:
                att_dict["value"] = run_output["value"]

    logging.warning("Serializing values")
    for doc_id in outputs:
        doc_runner_keys = get_doc_runner_keys(runner_keys, outputs[doc_id]["runners"])
        doc_full_name = doc_data[doc_id]["attributes"]["full_name"]["value"]

        for runner_key in doc_runner_keys:
            runner_dict = doc_data[doc_id]["runners"][runner_key]
            att_dict = doc_data[doc_to_run]["attributes"][runner_dict["attribute_key"]]

            if outputs[doc_id]["runners"][runner_key]["failed"]:
                continue

            # You don't want to actually deserialize model on the other end this it
            # involves  running arbitrary user code
            if runner_key == "model_builder":
                value = {
                    "model": outputs[doc_id]["runners"][runner_key]["value"],
                    "class_def": runner_dict["class_def"],
                }
            else:
                value = outputs[doc_id]["runners"][runner_key]["value"]

            value, output = attempt_serialize(value, att_dict["type"], with_db=with_db)
            if output:
                outputs[doc_id]["runners"][runner_key] = output
            else:
                outputs[doc_id]["runners"][runner_key]["value"] = value

    # Drop all the attribute stuff since that information should appear in the
    # corresponding runner dict
    outputs = {n: d["runners"] for n, d in outputs.items()}

    logging.warning("Cleaning up connections")
    # cleanup any connections
    for cleanup in cleanups:
        cleanup()

    return outputs


def get_doc_runner_keys(
    runner_keys: Optional[List[Text]], runners_dict: Dict[Text, Any]
):
    doc_runner_keys = (
        runner_keys if runner_keys is not None else list(runners_dict.keys())
    )
    if "value_setter" in doc_runner_keys:
        doc_runner_keys = ["value_setter"] + [
            k for k in doc_runner_keys if k != "value_setter"
        ]
    if "plotter" in doc_runner_keys:
        doc_runner_keys = [k for k in doc_runner_keys if k != "plotter"] + ["plotter"]
    return doc_runner_keys


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def get_doc_object(
    att_dict: Dict[Text, Dict[Text, Any]], with_db: bool = True
) -> DotDict:
    obj = {}
    for att in att_dict:
        if not with_db and att in ("model", "conn"):
            continue
        obj[att] = copy.deepcopy(att_dict[att]["value"])

    return DotDict(obj)

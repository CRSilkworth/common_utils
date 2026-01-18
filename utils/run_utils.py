from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import run_with_expected_type, run_with_generator
from utils.downloader import cached_stream_subgraph_by_key, prefetch
from utils.doc_obj import DocObj
from utils.datetime_utils import to_isos, to_datetimes
import logging
import requests
import os
from google.oauth2 import service_account
import json


def run(
    doc_data: Dict[Text, Dict[Text, Any]],
    docs_to_run: List[Text],
    auth_data: Dict[Text, Text],
    clone_nums: List[int],
    time_ranges: List[List[Text]],
    attributes_to_run: Optional[List[Text]] = None,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):

    time_ranges = [(to_datetimes(tr[0]), to_datetimes(tr[1])) for tr in time_ranges]

    if auth_data["sa_key"]:
        key_dict = json.loads(auth_data["sa_key"])
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        fs_kwargs = dict(credentials=credentials, project=key_dict["project_id"])
    else:
        os.environ["FIRESTORE_EMULATOR_HOST"] = (
            "firestore-emulator.default.svc.cluster.local:8080"
        )
        fs_kwargs = {}

    from google.cloud import firestore

    fs_db = firestore.Client(**fs_kwargs)

    run_config = run_config if run_config else {}

    doc_objs = {}
    doc_id_to_full_name = {}
    doc_full_name_to_id = {}
    for doc_id in doc_data:
        doc = DocObj(
            doc_id=doc_id,
            fs_db=fs_db,
            full_name=doc_data[doc_id]["full_name"],
            doc_dict=doc_data[doc_id],
            auth_data=auth_data,
            global_vars=kwargs.get("globals", {}),
        )

        doc_id_to_full_name[doc_id] = doc.full_name
        doc_full_name_to_id[doc.full_name] = doc_id
        doc_objs[doc.full_name] = doc
        if doc.doc_id not in docs_to_run:
            continue
        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not attribute.runnable or attribute.no_function_body:
                continue

    ref_dict = get_ref_dict(
        docs_to_run, doc_id_to_full_name, doc_objs, attributes_to_run
    )

    prefetch(
        fs_db=fs_db,
        auth_data=auth_data,
        docs_to_run=docs_to_run,
        ref_dict=ref_dict,
        doc_id_to_full_name=doc_id_to_full_name,
        clone_nums=clone_nums,
    )
    full_space = get_full_space(
        clone_nums=clone_nums, time_ranges=time_ranges, doc_objs=doc_objs
    )
    data_iterator = cached_stream_subgraph_by_key(
        fs_db=fs_db,
        auth_data=auth_data,
        full_space=full_space,
        docs_to_run=docs_to_run,
        ref_dict=ref_dict,
        clone_nums=clone_nums,
        doc_id_to_full_name=doc_id_to_full_name,
        doc_full_name_to_id=doc_full_name_to_id,
    )

    for run_key, data_dict in data_iterator:
        clone_num, time_range, full_name, att = run_key
        doc = doc_objs[full_name]
        doc.doc_objs = doc_objs
        doc.doc_id_to_full_name = doc_id_to_full_name
        attribute = doc.attributes[att]
        attribute._set_context(
            clone_num=clone_num,
            time_range=time_range,
        )
        if (
            time_range[0] < attribute.valid_time_range[0]
            or time_range[1] > attribute.valid_time_range[1]
        ):
            continue
        if clone_num not in attribute.valid_clone_nums:
            continue

        logging.warning(f"Running {clone_num}, {time_range}, {doc.full_name}, {att}")

        upstream_failure = False
        # Set all the arguments to the function to run
        runner_kwargs = {}
        runner_kwargs["clone_num"] = clone_num
        if att == "data":
            runner_kwargs["time_range"] = time_range

        for var_name, input_doc_id in attribute.var_name_to_id.items():
            input_full_name = doc_id_to_full_name[input_doc_id]
            input_doc = doc_objs[input_full_name]
            runner_kwargs[var_name] = input_doc
            if input_doc.failures():
                output = failed_output(
                    "Upstream failure from "
                    f"{input_doc.full_name}: {sorted(input_doc.failures())}"
                )
                attribute._add_output(output)
                upstream_failure = True
                logging.warning(
                    ("upstream failure", input_doc.full_name, input_doc.failures())
                )
                break
            for input_att, input_attribute in input_doc.attributes.items():
                if not input_attribute.runnable:
                    continue
                input_attribute._set_context(
                    clone_num=clone_num,
                    time_range=time_range,
                )
                if input_attribute.chunked:
                    input_attribute._set_val(
                        input_attribute.get_iterator(
                            clone_nums=[clone_num],
                            time_range=time_range,
                        )
                    )
                else:
                    input_attribute._set_val(
                        data_dict.get((input_full_name, input_att)), serialized=True
                    )

        block_key = [
            clone_num,
            to_isos(time_range),
            0,
        ]
        if block_key in attribute.locks:
            logging.warning(f"lock found {block_key}")
            attribute._upload_chunk(run_key=run_key, value_chunk=None, lock_value=True)
            continue

        func = attribute.get_func()

        if not func:
            continue

        if upstream_failure:
            attribute._delete_value_file_blocks(version=attribute.new_version)
            continue

        if attribute.chunked:
            run_generator = run_with_generator(
                func, runner_kwargs, attribute.value_type
            )
            for chunk_num, run_output_chunk in enumerate(run_generator):
                attribute._add_output(run_output_chunk)
                if run_output_chunk["failed"]:
                    attribute._delete_value_file_blocks(version=attribute.new_version)
                    break

                attribute._upload_chunk(
                    run_key=run_key,
                    chunk_num=chunk_num,
                    value_chunk=run_output_chunk["value"],
                )

        else:
            output = run_with_expected_type(func, runner_kwargs, attribute.value_type)
            attribute._add_output(output)
            if not output["failed"]:
                attribute._upload_chunk(run_key=run_key, value_chunk=output["value"])
            else:
                attribute._delete_value_file_blocks(version=attribute.new_version)
    logging.warning("Sending output")
    outputs = {}
    for doc in doc_objs.values():

        if doc.doc_id not in docs_to_run:
            continue

        upstream_failure = False
        attributes_to_run = (
            list(doc_data[doc.doc_id].keys())
            if attributes_to_run is None
            else attributes_to_run
        )
        outputs[doc.doc_id] = {}
        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not attribute.runnable:
                continue
            outputs[doc.doc_id][att] = attribute._get_output()

            attribute._finalize()
    send_output(outputs, auth_data, caller=kwargs.get("caller"))

    logging.warning("Cleaning up connections")
    # cleanup any connections
    for doc in doc_objs.values():
        for attribute in doc.attributes.values():
            if attribute.cleanup:
                attribute.cleanup()


def get_full_space(clone_nums, time_ranges, doc_objs):

    full_space = []
    for clone_num in clone_nums:
        for time_range in time_ranges:
            full_space.append((clone_num, time_range))
    full_space.sort()
    for doc_obj in doc_objs.values():
        for _, attribute in doc_obj.attributes.items():
            attribute._set_full_space(full_space, clone_nums, time_ranges)
    return full_space


def get_ref_dict(docs_to_run, doc_id_to_full_name, doc_objs, attributes_to_run):
    ref_dict = {}

    for doc_to_run in docs_to_run:
        full_name = doc_id_to_full_name[doc_to_run]
        doc = doc_objs[full_name]
        ref_dict[full_name] = {}

        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue

            attribute = doc.attributes[att]
            if not attribute.runnable or attribute.no_function_body:
                continue
            ref_dict[full_name][att] = {
                "full_name": full_name,
                "attribute_name": att,
                "doc_id": doc_to_run,
                "new_version": attribute.new_version,
                "old_version": attribute.old_version,
                "inputs": [],
            }

            for _, input_doc_id in attribute.var_name_to_id.items():
                input_full_name = doc_id_to_full_name[input_doc_id]
                input_doc = doc_objs[input_full_name]
                for input_att, input_attribute in input_doc.attributes.items():
                    if not input_attribute.runnable:
                        continue
                    # if not input_attribute.version:
                    #     continue
                    ref_dict[doc.full_name][att]["inputs"].append(
                        {
                            "doc_id": input_doc_id,
                            "full_name": input_doc.full_name,
                            "attribute_name": input_att,
                            "new_version": input_attribute.new_version,
                            "old_version": input_attribute.old_version,
                        }
                    )

    return ref_dict


def send_output(
    outputs,
    auth_data,
    caller: Optional[Text] = None,
):

    # Send the attribute result back to the backend
    data = {
        "docs_to_run": list(outputs.keys()),
        "outputs": outputs,
        "caller": caller,
        "auth_data": auth_data,
        "run_completed": False,
        "run_output": {"failed": False, "message": ""},
    }
    requests.post(
        os.path.join(auth_data["dash_app_url"], "job-result"),
        json=data,
        headers={"Authorization": f"Bearer {auth_data['token']}"},
    )

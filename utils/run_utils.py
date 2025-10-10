from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import run_with_expected_type, run_with_generator
from utils.downloader import (
    prefetch_subgraph,
    cached_stream_subgraph_by_key,
    save_bytes_to_disk,
    MAX_CACHE_BYTES,
)
from utils.doc_obj import DocObj
from utils.datetime_utils import to_micro
from utils.serialize_utils import attempt_serialize
import datetime
import logging
import requests
import os


def run_sims(
    doc_data: Dict[Text, Dict[Text, Any]],
    docs_to_run: List[Text],
    auth_data: Dict[Text, Text],
    attributes_to_run: Optional[List[Text]] = None,
    time_ranges_keys: Optional[List[Text]] = None,
    sim_iter_nums: Optional[List[int]] = None,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):

    run_config = run_config if run_config else {}

    time_ranges_keys_to_run = set()
    sim_iter_nums_to_run = set()
    doc_objs = {}
    doc_id_to_full_name = {}
    for doc_id in doc_data:
        doc = DocObj(
            doc_id=doc_id,
            doc_dict=doc_data[doc_id],
            auth_data=auth_data,
            global_vars=kwargs.get("globals", {}),
        )
        if auth_data["calc_graph_id"] == doc_id:
            full_name = "__CALCGRAPH__"
        else:
            full_name = doc.full_name.val

        doc_id_to_full_name[doc_id] = full_name
        doc_objs[full_name] = doc
        if doc.doc_id not in docs_to_run:
            continue
        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not attribute.runnable or attribute.no_function_body:
                continue
            time_ranges_keys_to_run.update(attribute.time_ranges_keys)
            sim_iter_nums_to_run.update(attribute.sim_iter_nums)

    calc_graph_doc = doc_objs["__CALCGRAPH__"]

    ref_dict = get_ref_dict(
        docs_to_run, doc_id_to_full_name, doc_objs, attributes_to_run
    )
    if time_ranges_keys is not None:
        time_ranges_keys_to_run = time_ranges_keys_to_run & set(time_ranges_keys)
    if sim_iter_nums is not None:
        sim_iter_nums_to_run = sim_iter_nums_to_run & set(sim_iter_nums)

    start_key = prefetch_subgraph(
        auth_data, ref_dict, sim_iter_nums_to_run, time_ranges_keys_to_run
    )
    print("start_key", start_key)

    run_key_iterator = get_run_key_iterator(
        calc_graph_doc=calc_graph_doc,
        ref_dict=ref_dict,
        time_ranges_keys=time_ranges_keys_to_run,
        sim_iter_nums=sim_iter_nums_to_run,
        is_calc_graph_run=calc_graph_doc.doc_id in docs_to_run,
    )

    data_iterator = cached_stream_subgraph_by_key(
        auth_data=auth_data,
        run_key_iterator=run_key_iterator,
        ref_dict=ref_dict,
        time_ranges_keys=time_ranges_keys_to_run,
        sim_iter_nums=sim_iter_nums_to_run,
        start_key=start_key,
    )

    for run_key, data_dict in data_iterator:
        sim_iter_num, time_range, time_ranges_key, full_name, att = run_key
        _run_key = (
            sim_iter_num,
            time_range[0].isoformat(),
            time_range[1].isoformat(),
            time_ranges_key,
            full_name,
            att,
        )
        doc = doc_objs[full_name]
        attribute = doc.attributes[att]
        attribute._set_context(
            sim_iter_num=sim_iter_num,
            time_ranges_key=time_ranges_key,
            time_range=time_range,
        )
        if sim_iter_num not in attribute.sim_iter_nums:
            continue
        if time_ranges_key not in attribute.time_ranges_keys:
            continue

        logging.info(
            f"Running {sim_iter_num}, {time_range}, {time_ranges_key},"
            f" {doc.full_name.val}, {att}"
        )
        print(
            f"Running {sim_iter_num}, {time_range}, {time_ranges_key},"
            f" {doc.full_name.val}, {att}"
        )

        upstream_failure = False
        # Set all the arguments to the function to run
        runner_kwargs = {}
        runner_kwargs["sim_iter_num"] = sim_iter_num
        runner_kwargs["time_ranges_key"] = time_ranges_key
        runner_kwargs["time_range"] = time_range
        for var_name, input_doc_id in attribute.var_name_to_id.items():
            input_full_name = doc_id_to_full_name[input_doc_id]
            input_doc = doc_objs[input_full_name]
            runner_kwargs[var_name] = input_doc
            if input_doc.failures():
                output = failed_output(
                    "Upstream failure from "
                    f"{input_doc.full_name.val}: {sorted(input_doc.failures())}"
                )
                attribute._add_output(output)
                upstream_failure = True
                print("upstream failure", input_doc.full_name.val, input_doc.failures())
                break
            for _, input_attribute in input_doc.attributes.items():
                if not input_attribute.runnable:
                    continue
                input_attribute._set_context(
                    sim_iter_num=sim_iter_num,
                    time_ranges_key=time_ranges_key,
                    time_range=time_range,
                )
                if input_attribute.chunked:
                    input_attribute._set_val(
                        input_attribute.get_iterator(
                            sim_iter_nums=[sim_iter_num],
                            time_ranges_keys=[time_ranges_key],
                            time_range=time_range,
                        )
                    )
                else:
                    input_attribute._set_val(
                        data_dict.get(input_attribute.value_file_ref), serialized=True
                    )

        block_key = [
            sim_iter_num,
            time_ranges_key,
            time_range[0].isoformat(),
            time_range[1].isoformat(),
            0,
        ]
        if block_key in attribute.overrides:
            attribute._upload_chunk(value_chunk=None, overriden=True)
            attribute._flush()
            continue

        # Convert the functionv string to a callable function

        if upstream_failure or not attribute.func:
            print(f"Skipping {doc.full_name.val}-{att}")
            attribute._send_output(caller=kwargs.get("caller"))
            continue

        if attribute.chunked:
            run_generator = run_with_generator(
                attribute.func, runner_kwargs, attribute.value_type
            )
            for chunk_num, run_output_chunk in enumerate(run_generator):
                attribute._add_output(run_output_chunk)
                if run_output_chunk["failed"]:
                    break

                attribute._upload_chunk(
                    chunk_num=chunk_num, value_chunk=run_output_chunk["value"]
                )
                attribute._flush()
                _value_chunk, _ = attempt_serialize(
                    run_output_chunk["value"], attribute.value_type
                )
                block_bytes = _value_chunk.encode("utf-8")
                print("saving", _run_key)
                save_bytes_to_disk(_run_key, block_bytes, MAX_CACHE_BYTES)

        else:
            output = run_with_expected_type(
                attribute.func, runner_kwargs, attribute.value_type
            )
            attribute._add_output(output)
            if not output["failed"]:
                print(
                    f"uploading {sim_iter_num}, {time_range}, {time_ranges_key},"
                    f" {doc.full_name.val}, {att}"
                )
                attribute._upload_chunk(chunk_num=0, value_chunk=output["value"])
                attribute._flush()

    logging.info("Sending output")
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
            if not attribute.runnable or attribute.no_function_body:
                continue
            outputs[doc.doc_id][att] = attribute._get_output()
    send_output(outputs, auth_data, caller=kwargs.get("caller"))

    logging.info("Cleaning up connections")
    # cleanup any connections
    for doc in doc_objs.values():
        for attribute in doc.attributes.values():
            if attribute.cleanup:
                attribute.cleanup()


def get_run_key_iterator(
    calc_graph_doc: DocObj,
    sim_iter_nums: List[str] = None,
    time_ranges_keys: List[str] = None,
    ref_dict: Optional[Dict[Text, Dict[Text, Dict[Text, Any]]]] = None,
    is_calc_graph_run: bool = False,
):

    if is_calc_graph_run:
        sims = {"0": {}}
        all_time_ranges = {}
    else:
        try:
            _, sims = next(
                calc_graph_doc.attributes["sims"].get_iterator(
                    sim_iter_nums=[0], time_ranges_keys=["__BEGIN_TIME__"]
                )
            )
        except StopIteration:
            sims = {"0": {}}
        try:
            __, all_time_ranges = next(
                calc_graph_doc.attributes["all_time_ranges"].get_iterator(
                    sim_iter_nums=[0], time_ranges_keys=["__BEGIN_TIME__"]
                )
            )
        except StopIteration:
            all_time_ranges = {
                "__BEGIN_TIME__": [(datetime.datetime.min, datetime.datetime.min)]
            }

    if not sims:
        sims = [{}]
    all_time_ranges["__BEGIN_TIME__"] = [(datetime.datetime.min, datetime.datetime.min)]

    results = []
    for sim_iter_num, _ in enumerate(sims):
        if sim_iter_nums and sim_iter_num not in sim_iter_nums:
            continue
        for time_ranges_key, time_ranges in all_time_ranges.items():
            if time_ranges_keys and time_ranges_key not in time_ranges_keys:
                continue
            for time_range in time_ranges:
                time_range = to_micro(time_range[0]), to_micro(time_range[1])
                for full_name in ref_dict:
                    for att in ref_dict[full_name]:
                        results.append(
                            (sim_iter_num, time_range, time_ranges_key, full_name, att)
                        )

    # Sort by (sim_iter_num, end, start, time_ranges_key)
    results.sort(
        key=lambda x: (
            x[0],  # sim_iter_num
            x[1][0],  # time_range_start
            x[1][1],  # time_range_end
            x[2],  # time_ranges_key
            x[3],
            x[4],
        )
    )

    for item in results:
        yield item


def get_ref_dict(docs_to_run, doc_id_to_full_name, doc_objs, attributes_to_run):
    # value_file_ref_groups = []
    # index_to_doc_id_att = []
    ref_dict = {}
    for doc_to_run in docs_to_run:
        full_name = doc_id_to_full_name[doc_to_run]
        doc = doc_objs[full_name]
        ref_dict[full_name] = {}
        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not attribute.runnable or attribute.no_function_body:
                continue
            ref_dict[full_name][att] = {
                "full_name": full_name,
                "attribute_name": att,
                "doc_id": doc_to_run,
                "value_file_ref": attribute.value_file_ref,
                "inputs": [],
            }

            # value_file_ref_groups.append([])
            # index_to_doc_id_att.append((doc_to_run, att))
            for _, input_doc_id in attribute.var_name_to_id.items():
                input_full_name = doc_id_to_full_name[input_doc_id]
                input_doc = doc_objs[input_full_name]
                for input_att, input_attribute in input_doc.attributes.items():
                    if not input_attribute.runnable:
                        continue
                    if not input_attribute.value_file_ref:
                        continue
                    ref_dict[doc.full_name.val][att]["inputs"].append(
                        {
                            "doc_id": input_doc_id,
                            "full_name": input_doc.full_name.val,
                            "attribute_name": input_att,
                            "value_file_ref": input_attribute.value_file_ref,
                        }
                    )
                    # value_file_ref_groups[-1].append(input_attribute.value_file_ref)

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

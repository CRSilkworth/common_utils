from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import run_with_expected_type, run_with_generator
from utils.downloader import prefetch_subgraph, cached_stream_subgraph_by_key
from utils.doc_obj import DocObj
from utils.datetime_utils import to_micro
from utils.type_utils import TimeRange
from utils.uploader import BatchUploader
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

    uploader = BatchUploader(auth_data=auth_data)

    time_ranges_keys_to_run = set()
    sim_iter_nums_to_run = set()
    doc_objs = {}
    doc_id_to_full_name = {}
    for doc_id in doc_data:
        doc = DocObj(
            doc_id=doc_id,
            full_name=doc_data[doc_id]["full_name"],
            doc_dict=doc_data[doc_id],
            auth_data=auth_data,
            global_vars=kwargs.get("globals", {}),
        )

        doc_id_to_full_name[doc_id] = doc.full_name
        doc_objs[doc.full_name] = doc
        if doc.doc_id not in docs_to_run:
            continue
        for att, attribute in doc.attributes.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not attribute.runnable or attribute.no_function_body:
                continue
            time_ranges_keys_to_run.update(attribute.time_ranges_keys)
            sim_iter_nums_to_run.update(attribute.sim_iter_nums)

    calc_graph_doc = doc_objs[doc_id_to_full_name[auth_data["calc_graph_id"]]]

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
    full_space = get_full_space(calc_graph_doc, doc_objs, docs_to_run)

    data_iterator = cached_stream_subgraph_by_key(
        auth_data=auth_data,
        full_space=full_space,
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
            f" {doc.full_name}, {att}"
        )
        print(
            f"Running {sim_iter_num}, {time_range}, {time_ranges_key},"
            f" {doc.full_name}, {att}"
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
                    f"{input_doc.full_name}: {sorted(input_doc.failures())}"
                )
                attribute._add_output(output)
                upstream_failure = True
                print("upstream failure", input_doc.full_name, input_doc.failures())
                break
            for input_att, input_attribute in input_doc.attributes.items():
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
                        data_dict.get((input_full_name, input_att)), serialized=True
                    )

        block_key = [
            sim_iter_num,
            time_ranges_key,
            time_range[0].isoformat(),
            time_range[1].isoformat(),
            0,
        ]
        if block_key in attribute.overrides:
            attribute._upload_chunk(
                uploader=uploader, _run_key=_run_key, value_chunk=None, overriden=True
            )
            continue

        # Convert the functionv string to a callable function

        if upstream_failure or not attribute.func:
            print(f"Skipping {doc.full_name}-{att}")
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
                    uploader=uploader,
                    _run_key=_run_key,
                    chunk_num=chunk_num,
                    value_chunk=run_output_chunk["value"],
                )

        else:
            output = run_with_expected_type(
                attribute.func, runner_kwargs, attribute.value_type
            )
            attribute._add_output(output)
            if not output["failed"]:
                attribute._upload_chunk(
                    uploader=uploader, _run_key=_run_key, value_chunk=output["value"]
                )
    uploader.flush_batch()
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


def get_all_time_ranges(
    calc_graph_doc, is_calc_graph_run
) -> Dict[Text, List[TimeRange]]:
    all_time_ranges = {}

    if not is_calc_graph_run:

        try:
            __, all_time_ranges = next(
                calc_graph_doc.attributes["all_time_ranges"].get_iterator(
                    sim_iter_nums=[0],
                    time_ranges_keys=["__BEGIN_TIME__"],
                    use_cache=False,
                )
            )
        except StopIteration:
            all_time_ranges = {
                "__BEGIN_TIME__": [(datetime.datetime.min, datetime.datetime.min)]
            }

    all_time_ranges["__BEGIN_TIME__"] = [(datetime.datetime.min, datetime.datetime.min)]
    return all_time_ranges


def get_sims(calc_graph_doc, is_calc_graph_run) -> List[Dict[Text, Any]]:
    sims = {"0": {}}
    if not is_calc_graph_run:
        try:
            _, sims = next(
                calc_graph_doc.attributes["sims"].get_iterator(
                    sim_iter_nums=[0],
                    time_ranges_keys=["__BEGIN_TIME__"],
                    use_cache=False,
                )
            )
        except StopIteration:
            sims = [{}]

    if not sims:
        sims = [{}]
    return sims


def get_full_space(calc_graph_doc, doc_objs, docs_to_run):

    is_calc_graph_run = calc_graph_doc.doc_id in docs_to_run
    sims = get_sims(calc_graph_doc, is_calc_graph_run)
    all_time_ranges = get_all_time_ranges(calc_graph_doc, is_calc_graph_run)

    full_space = []
    for sim_iter_num, _ in enumerate(sims):
        for time_ranges_key, time_ranges in all_time_ranges.items():
            for time_range in time_ranges:
                time_range = to_micro(time_range[0]), to_micro(time_range[1])
                full_space.append((sim_iter_num, time_range, time_ranges_key))
    full_space.sort()
    for doc_obj in doc_objs.values():
        for _, attribute in doc_obj.attributes.items():
            attribute._set_full_space(full_space)
    return full_space


def get_ref_dict(docs_to_run, doc_id_to_full_name, doc_objs, attributes_to_run):
    ref_dict = {}
    att_run_order = {
        k: i
        for i, k in enumerate(["sims", "all_time_ranges", "basic", "model", "plot"])
    }
    for doc_to_run in docs_to_run:
        full_name = doc_id_to_full_name[doc_to_run]
        doc = doc_objs[full_name]
        ref_dict[full_name] = {}

        doc_att_order = sorted(
            doc.attributes.keys(),
            key=lambda k: att_run_order.get(k, len(att_run_order)),
        )
        for att in doc_att_order:
            if not (
                attributes_to_run is None
                or att in attributes_to_run
                or att not in doc.attributes
            ):
                continue

            attribute = doc.attributes[att]
            if not attribute.runnable or attribute.no_function_body:
                continue
            ref_dict[full_name][att] = {
                "full_name": full_name,
                "attribute_name": att,
                "doc_id": doc_to_run,
                "value_file_ref": attribute.value_file_ref,
                "inputs": [],
            }

            for _, input_doc_id in attribute.var_name_to_id.items():
                input_full_name = doc_id_to_full_name[input_doc_id]
                input_doc = doc_objs[input_full_name]
                for input_att, input_attribute in input_doc.attributes.items():
                    if not input_attribute.runnable:
                        continue
                    if not input_attribute.value_file_ref:
                        continue
                    ref_dict[doc.full_name][att]["inputs"].append(
                        {
                            "doc_id": input_doc_id,
                            "full_name": input_doc.full_name,
                            "attribute_name": input_att,
                            "value_file_ref": input_attribute.value_file_ref,
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

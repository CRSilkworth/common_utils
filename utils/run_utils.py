from typing import Text, Dict, Any, Optional, List
from utils.misc_utils import failed_output
from utils.function_utils import (
    create_function,
    run_with_expected_type,
    run_with_generator,
)
from utils.type_utils import get_known_types
from utils.generator_utils import merge_generators
from utils.doc_obj import DocObj
import datetime
import logging


def run_docs(
    doc_data: Dict[Text, Dict[Text, Any]],
    docs_to_run: List[Text],
    auth_data: Dict[Text, Text],
    attributes_to_run: Optional[List[Text]] = None,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):

    run_config = run_config if run_config else {}
    doc_objs = {
        d: DocObj(
            doc_id=d,
            doc_dict=doc_data[d],
            auth_data=auth_data,
        )
        for d in doc_data
    }

    calc_graph_doc = doc_objs[auth_data["calc_graph_id"]]
    for doc_to_run in docs_to_run:
        doc = doc_objs[doc_to_run]

        upstream_failure = False
        attributes_to_run = (
            list(doc_data[doc_to_run].keys())
            if attributes_to_run is None
            else attributes_to_run
        )
        # Run all the attributes associate with this doc
        for att, att_dict in doc.att_dicts.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                continue

            logging.info(f"Running {doc.full_name}-{att}")
            print(f"Running {doc.full_name} {att}")
            # Set all the arguments to the function to run
            for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                input_doc = doc_objs[input_doc_id]

                if input_doc.failures():
                    output = failed_output(
                        "Upstream failure from "
                        f"{input_doc.full_name}: {sorted(input_doc.failures())}"
                    )
                    doc.add_output(att, output)
                    upstream_failure = True
                    break

            if upstream_failure:
                break

            # Convert the functionv string to a callable function
            func, output = create_function(
                function_name=att_dict["function_name"],
                function_header=att_dict["function_header"],
                function_string=att_dict["function_string"],
                allowed_modules=get_known_types(),
                global_vars=kwargs.get("globals", {}),
            )
            doc.add_output(att, output)

            if upstream_failure or output["failed"] or not func:
                print(f"Skipping {doc.full_name}-{att}")
                doc.send_output(att, caller=kwargs.get("caller"))
                continue

            logging.info(f"Running {doc.full_name}: {att}")

            all_gens = {
                (calc_graph_doc.doc_id, "sims_all_time_ranges"): sims_time_ranges_iter(
                    calc_graph_doc=calc_graph_doc,
                    sim_iter_nums=att_dict["sim_iter_nums"],
                    time_ranges_keys=att_dict["time_ranges_keys"],
                    is_calc_graph_run=calc_graph_doc.doc_id in docs_to_run,
                )
            }
            runner_kwargs = {}
            for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                input_doc = doc_objs[input_doc_id]
                runner_kwargs[var_name] = input_doc
                for input_att, input_att_dict in input_doc.att_dicts.items():
                    if not input_att_dict.get("generator", False):
                        continue
                    all_gens[(input_doc_id, input_att)] = input_doc.get_iterator(
                        input_att,
                        sim_iter_nums=doc.att_dicts[att]["sim_iter_nums"],
                        time_ranges_keys=doc.att_dicts[att]["time_ranges_keys"],
                    )

            merged = merge_generators(all_gens.values())
            for (sim_iter_num, time_ranges_key, time_range), values in merged:
                for (input_doc_id, input_att), value in zip(all_gens.keys(), values):
                    doc_objs[input_doc_id][input_att] = value
                runner_kwargs["sim_iter_num"] = sim_iter_num
                runner_kwargs["time_ranges_key"] = time_ranges_key
                runner_kwargs["time_range"] = time_range

                failed = False
                if att_dict["chunked"]:
                    run_generator = run_with_generator(
                        func, runner_kwargs, att_dict["value_type"]
                    )
                    for chun_num, run_output_chunk in enumerate(run_generator):
                        doc.add_output(att, run_output_chunk)
                        if run_output_chunk["failed"]:
                            failed = True
                            break

                        doc.upload_chunk(
                            att=att,
                            sim_iter_num=sim_iter_num,
                            time_ranges_key=time_ranges_key,
                            time_range=time_range,
                            chunk_num=chun_num,
                            value_chunk=run_output_chunk["value"],
                        )

                else:
                    output = run_with_expected_type(
                        func, runner_kwargs, att_dict["value_type"]
                    )
                    doc.add_output(att, output)
                    if not output["failed"]:
                        doc.upload_chunk(
                            att=att,
                            sim_iter_num=sim_iter_num,
                            time_ranges_key=time_ranges_key,
                            time_range=time_range,
                            chunk_num=0,
                            value_chunk=output["value"],
                        )
                    else:
                        failed = True
                # If it succeeds then switch to using the new value file/clear old
                # values.
                if not failed:
                    doc.finalize_value_update(att)
                context = {
                    "sim_iter_num": sim_iter_num,
                    "time_ranges_key": time_ranges_key,
                    "time_range": time_range,
                }

                doc.send_output(att, caller=kwargs.get("caller"), context=context)
    logging.info("Cleaning up connections")
    # cleanup any connections
    for doc in doc_objs.values():
        for cleanup in doc.cleanups.values():
            try:
                cleanup()
            except Exception:
                continue


def run_sims(
    doc_data: Dict[Text, Dict[Text, Any]],
    docs_to_run: List[Text],
    auth_data: Dict[Text, Text],
    attributes_to_run: Optional[List[Text]] = None,
    run_config: Optional[Dict[Text, Any]] = None,
    **kwargs,
):

    run_config = run_config if run_config else {}
    doc_objs = {
        d: DocObj(
            doc_id=d,
            doc_dict=doc_data[d],
            auth_data=auth_data,
        )
        for d in doc_data
    }

    for doc in doc_objs.values():
        if doc.doc_id not in docs_to_run:
            continue
        for att, att_dict in doc.att_dicts.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                continue

            att_dict["value_file_ref"] = att_dict["new_value_file_ref"]

    calc_graph_doc = doc_objs[auth_data["calc_graph_id"]]
    iterator = sims_time_range_end_iter(
        calc_graph_doc=calc_graph_doc,
        is_calc_graph_run=calc_graph_doc.doc_id in docs_to_run,
    )
    for (sim_iter_num, time_ranges_key, time_range), sim_params in iterator:

        for doc_to_run in docs_to_run:
            doc = doc_objs[doc_to_run]

            upstream_failure = False
            attributes_to_run = (
                list(doc_data[doc_to_run].keys())
                if attributes_to_run is None
                else attributes_to_run
            )
            # Run all the attributes associate with this doc
            for att, att_dict in doc.att_dicts.items():
                if not (attributes_to_run is None or att in attributes_to_run):
                    continue
                if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                    continue
                if sim_iter_num not in att_dict["sim_iter_nums"]:
                    continue
                if time_ranges_key not in att_dict["time_ranges_keys"]:
                    continue

                logging.info(f"Running {doc.full_name}-{att}")
                print(f"Running {doc.full_name} {att}")
                # Set all the arguments to the function to run
                for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                    input_doc = doc_objs[input_doc_id]

                    if input_doc.failures():
                        output = failed_output(
                            "Upstream failure from "
                            f"{input_doc.full_name}: {sorted(input_doc.failures())}"
                        )
                        doc.add_output(att, output)
                        upstream_failure = True
                        print("upstream failure", input_doc.failures())
                        break

                if upstream_failure:
                    break

                # Convert the functionv string to a callable function
                func, output = create_function(
                    function_name=att_dict["function_name"],
                    function_header=att_dict["function_header"],
                    function_string=att_dict["function_string"],
                    allowed_modules=get_known_types(),
                    global_vars=kwargs.get("globals", {}),
                )
                doc.add_output(att, output)

                if upstream_failure or output["failed"] or not func:
                    print(f"Skipping {doc.full_name}-{att}")
                    doc.send_output(att, caller=kwargs.get("caller"))
                    continue

                logging.info(f"Running {doc.full_name}: {att}")

                runner_kwargs = {}
                for var_name, input_doc_id in att_dict["var_name_to_id"].items():
                    input_doc = doc_objs[input_doc_id]
                    runner_kwargs[var_name] = input_doc
                    for input_att, input_att_dict in input_doc.att_dicts.items():
                        if not input_att_dict.get("generator", False):
                            continue
                        if input_att_dict.get("chunked", False):
                            doc_objs[input_doc_id][input_att] = input_doc.get_iterator(
                                input_att,
                                sim_iter_nums=[sim_iter_num],
                                time_ranges_keys=[time_ranges_key],
                                time_range=time_range,
                            )
                        else:
                            value = input_doc.get_iterator(
                                input_att,
                                sim_iter_nums=[sim_iter_num],
                                time_ranges_keys=[time_ranges_key],
                                time_range=time_range,
                            )
                            try:
                                doc_objs[input_doc_id][input_att] = next(value)
                            except StopIteration:
                                doc_objs[input_doc_id][input_att] = None

                runner_kwargs["sim_iter_num"] = sim_iter_num
                runner_kwargs["time_ranges_key"] = time_ranges_key
                runner_kwargs["time_range"] = time_range

                failed = False
                if att_dict["chunked"]:
                    run_generator = run_with_generator(
                        func, runner_kwargs, att_dict["value_type"]
                    )
                    for chun_num, run_output_chunk in enumerate(run_generator):
                        doc.add_output(att, run_output_chunk)
                        if run_output_chunk["failed"]:
                            failed = True
                            break

                        doc.upload_chunk(
                            att=att,
                            sim_iter_num=sim_iter_num,
                            time_ranges_key=time_ranges_key,
                            time_range=time_range,
                            chunk_num=chun_num,
                            value_chunk=run_output_chunk["value"],
                        )

                else:
                    output = run_with_expected_type(
                        func, runner_kwargs, att_dict["value_type"]
                    )
                    doc.add_output(att, output)
                    if not output["failed"]:
                        doc.upload_chunk(
                            att=att,
                            sim_iter_num=sim_iter_num,
                            time_ranges_key=time_ranges_key,
                            time_range=time_range,
                            chunk_num=0,
                            value_chunk=output["value"],
                        )
                    else:
                        failed = True
                # If it succeeds then switch to using the new value file/clear old
                # values.
                if not failed:
                    doc.finalize_value_update(att)

    for doc_to_run in docs_to_run:
        doc = doc_objs[doc_to_run]

        upstream_failure = False
        attributes_to_run = (
            list(doc_data[doc_to_run].keys())
            if attributes_to_run is None
            else attributes_to_run
        )
        # Run all the attributes associate with this doc
        for att, att_dict in doc.att_dicts.items():
            if not (attributes_to_run is None or att in attributes_to_run):
                continue
            if not att_dict.get("runnable", False) or att_dict.get("empty", False):
                continue

            doc.send_output(att, caller=kwargs.get("caller"))
    logging.info("Cleaning up connections")
    # cleanup any connections
    for doc in doc_objs.values():
        for cleanup in doc.cleanups.values():
            try:
                cleanup()
            except Exception:
                continue


def sims_time_ranges_iter(
    calc_graph_doc: DocObj,
    sim_iter_nums: List[Text] = None,
    time_ranges_keys: List[Text] = None,
    is_calc_graph_run: bool = False,
):
    dt_max = datetime.datetime.max
    if is_calc_graph_run:
        sims = {"0": {}}
        all_time_ranges = {}
    else:
        try:
            _, sims = next(calc_graph_doc.get_iterator("sims"))
        except StopIteration:
            sims = {"0": {}}
        try:
            __, all_time_ranges = next(calc_graph_doc.get_iterator("all_time_ranges"))
        except StopIteration:
            all_time_ranges = {
                "__WHOLE__": [
                    (datetime.datetime.min, dt_max.replace(microsecond=999000))
                ]
            }

    # NOTE: Must contain sims[0] AND __WHOLE__ if user removed them put them back in.
    if not sims:
        sims = [{}]
    # mongo only supports upto milliseconds, so to be consistent knock off the micro
    # seconds here.

    all_time_ranges["__WHOLE__"] = [
        (datetime.datetime.min, dt_max.replace(microsecond=999000))
    ]
    for sim_iter_num, sim_params in enumerate(sims):
        if sim_iter_nums and sim_iter_num not in sim_iter_nums:
            continue
        for time_ranges_key, time_ranges in all_time_ranges.items():
            if time_ranges_keys and time_ranges_key not in time_ranges_keys:
                continue
            for time_range in time_ranges:
                yield (sim_iter_num, time_ranges_key, time_range), sim_params


def sims_time_range_end_iter(
    calc_graph_doc: DocObj,
    sim_iter_nums: List[str] = None,
    time_ranges_keys: List[str] = None,
    is_calc_graph_run: bool = False,
):

    dt_max = datetime.datetime.max
    if is_calc_graph_run:
        sims = {"0": {}}
        all_time_ranges = {}
    else:
        try:
            _, sims = next(calc_graph_doc.get_iterator("sims"))
        except StopIteration:
            sims = {"0": {}}
        try:
            __, all_time_ranges = next(calc_graph_doc.get_iterator("all_time_ranges"))
        except StopIteration:
            all_time_ranges = {
                "__WHOLE__": [
                    (datetime.datetime.min, dt_max.replace(microsecond=999000))
                ]
            }

    if not sims:
        sims = [{}]
    # Normalize datetime.max → millisecond precision
    dt_max = datetime.datetime.max.replace(microsecond=999000)
    all_time_ranges["__WHOLE__"] = [(datetime.datetime.min, dt_max)]

    results = []
    for sim_iter_num, sim_params in enumerate(sims):
        if sim_iter_nums and sim_iter_num not in sim_iter_nums:
            continue
        for time_ranges_key, time_ranges in all_time_ranges.items():
            if time_ranges_keys and time_ranges_key not in time_ranges_keys:
                continue
            for time_range in time_ranges:
                results.append(
                    ((sim_iter_num, time_ranges_key, time_range), sim_params)
                )

    # Sort by (sim_iter_num, end, start, time_ranges_key)
    results.sort(
        key=lambda x: (
            x[0][0],  # sim_iter_num
            x[0][2][1],  # time_range_end
            x[0][2][0],  # time_range_start
            x[0][1],  # time_ranges_key
        )
    )

    for item in results:
        yield item

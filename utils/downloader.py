from typing import Dict, Text, Optional, Tuple, List, Any, Set
import json
import hashlib
import os
from itertools import islice
from google.cloud.firestore_v1 import FieldFilter, And
import logging
from utils.datetime_utils import normalize_datetime

CACHE_DIR = "/tmp/cache"
MAX_CACHE_BYTES = 2 * 1024**3


def init_cache_size():
    global _current_cache_size
    os.makedirs(CACHE_DIR, exist_ok=True)
    _current_cache_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f))
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".bin")
    )


init_cache_size()


def pull_inputs_from_firestore(
    fs_db,
    auth_data: Dict[Text, Any],
    key: Tuple[str, str, str, str],
    doc_ids: List[str],
) -> Dict[Tuple[str, str], bytes]:
    """
    Pull all Firestore value_file_blocks for a given sim/time range combination
    across multiple docs and attributes, returning the latest versions.

    Args:
        fs_db: Firestore client.
        auth_data:
        sim_iter_num: Simulation iteration identifier.
        time_ranges_key: The key identifying the time range group.
        doc_ids: List of document IDs to fetch blocks for.
        ref_dict: Reference metadata for determining which attributes belong to each doc.

    Returns:
        Dict mapping (input_full_name, attribute_name) -> bytes (_value_chunk).
    """
    user_id = auth_data.get("user_id")
    calc_graph_id = auth_data.get("calc_graph_id")
    data_dict = {}
    sim_iter_num, (time_range_start, time_range_end), time_ranges_key = key

    for doc_batch in batched(doc_ids, 30):
        query = (
            fs_db.collection_group("value_file_block")
            .where(filter=FieldFilter("user_id", "==", user_id))
            .where(filter=FieldFilter("calc_graph_id", "==", calc_graph_id))
            .where(filter=FieldFilter("doc_id", "in", doc_batch))
            .where(filter=FieldFilter("sim_iter_num", "==", sim_iter_num))
            .where(filter=FieldFilter("time_range_start", "==", time_range_start))
            .where(filter=FieldFilter("time_range_end", "==", time_range_end))
            .where(filter=FieldFilter("time_ranges_key", "==", time_ranges_key))
        )
        latest_versions = {}
        for doc_snap in query.stream():
            data = doc_snap.to_dict()
            for k in ["time_range_start", "time_range_end", "version"]:
                data[k] = normalize_datetime(data[k])
            full_name = data.get("full_name")
            version = normalize_datetime(data.get("version"))
            att = data.get("attribute_name")
            run_key = (
                data.get("sim_iter_num"),
                (data.get("time_range_start"), data.get("time_range_end")),
                data.get("time_ranges_key"),
                full_name,
                att,
            )
            if (full_name, att) in latest_versions and version <= latest_versions[
                (full_name, att)
            ]:
                continue

            latest_versions[(full_name, att)] = version
            data_dict[(full_name, att)] = data.get("_value_chunk")
            save_to_disk(run_key, 0, data_dict[(full_name, att)], MAX_CACHE_BYTES)

    return data_dict


def batched(iterable, size=30):
    """Yield successive batches of length <= size."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def prefetch(
    fs_db,
    auth_data: Dict[Text, Any],
    docs_to_run: List[str],
    ref_dict: Dict[str, Dict[str, Dict[str, Any]]],
    doc_id_to_full_name: Dict[str, str],
    sim_iter_nums: Optional[List[str]] = None,
    time_ranges_keys: Optional[List[str]] = None,
) -> None:
    """
    Prefetch and locally cache latest Firestore value_file_blocks for all input nodes
    of the given documents, filtered by sim_iter_nums and time_ranges_keys.

    This respects Firestore’s 30-value limit for each 'in' filter by batching over
    doc_ids, attribute_names, sim_iter_nums, and time_ranges_keys.

    Args:
        fs_db: Firestore client
        user_id: Firestore user ID
        calc_graph_id: Firestore calc graph ID
        doc_to_runs: The doc_ids to prefetch inputs for
        ref_dict: Reference dictionary describing dependencies
        doc_id_to_full_name: Mapping of doc_id -> full_name
        sim_iter_nums: Optional list of simulation iteration identifiers
        time_ranges_keys: Optional list of time range keys
    """
    user_id = auth_data.get("user_id")
    calc_graph_id = auth_data.get("calc_graph_id")

    # ---- Collect all input document IDs and attributes ----
    input_doc_ids: Set[str] = set()
    input_atts: Set[str] = set()
    inputs: Set[Tuple[str, str]] = set()

    for doc_id in docs_to_run:
        full_name = doc_id_to_full_name.get(doc_id)

        for att, meta in ref_dict[full_name].items():
            for input_info in meta.get("inputs", []):
                input_doc_id = input_info.get("doc_id")
                input_full_name = doc_id_to_full_name.get(input_doc_id)
                input_att = input_info.get("attribute_name")
                if input_doc_id and input_att:
                    input_doc_ids.add(input_doc_id)
                    input_atts.add(input_att)
                    inputs.add((input_full_name, input_att))

    input_doc_ids -= set(docs_to_run)
    if not inputs:
        return

    latest_versions = {}
    for doc_batch in batched(sorted(input_doc_ids), 30):
        for sim_iter_num in sim_iter_nums:
            for time_ranges_key in time_ranges_keys:

                query = fs_db.collection_group("value_file_block").where(
                    filter=And(
                        filters=[
                            FieldFilter("user_id", "==", user_id),
                            FieldFilter("calc_graph_id", "==", calc_graph_id),
                            FieldFilter("doc_id", "in", doc_batch),
                            FieldFilter("sim_iter_num", "==", sim_iter_num),
                            FieldFilter("time_ranges_key", "==", time_ranges_key),
                        ]
                    )
                )

                for doc_snap in query.stream():
                    data = doc_snap.to_dict()
                    for k in ["time_range_start", "time_range_end", "version"]:
                        data[k] = normalize_datetime(data[k])
                    input_full_name = data.get("full_name")
                    input_att = data.get("attribute_name")

                    _input_key = (
                        data.get("sim_iter_num"),
                        (data.get("time_range_start"), data.get("time_range_end")),
                        data.get("time_ranges_key"),
                        input_full_name,
                        input_att,
                    )

                    if (input_full_name, input_att) not in inputs:
                        continue
                    if (
                        _input_key in latest_versions
                        and data.get("version") <= latest_versions[_input_key]
                    ):
                        continue
                    latest_versions[_input_key] = data.get("version")

                    if _get_cache_size() >= MAX_CACHE_BYTES:
                        return

                    # Cache locally (latest version first)
                    save_to_disk(
                        _input_key,
                        0,
                        data["_value_chunk"],
                        MAX_CACHE_BYTES,
                    )


def cached_stream_subgraph_by_key(
    fs_db,
    auth_data: Dict[Text, Any],
    full_space: List[Tuple],
    docs_to_run: List[Text],
    ref_dict: Dict[str, Dict[str, Dict[str, Any]]],
    sim_iter_nums: Optional[List[str]] = None,
    time_ranges_keys: Optional[List[str]] = None,
    doc_id_to_full_name: Optional[Dict[Text, Text]] = None,
):
    """
    Iterate through cached data first, then stream from Firestore and cache results.
    """

    run_key_iterator = get_run_key_iterator(
        full_space=full_space,
        docs_to_run=docs_to_run,
        ref_dict=ref_dict,
        time_ranges_keys=time_ranges_keys,
        sim_iter_nums=sim_iter_nums,
        doc_id_to_full_name=doc_id_to_full_name,
    )

    for run_key in run_key_iterator:
        sim_iter, (tr_start, tr_end), tr_key, full_name, att = run_key
        data_dict = {}
        not_in_cache = set()
        for input_dict in ref_dict[full_name][att]["inputs"]:
            input_full_name = input_dict["full_name"]
            input_att = input_dict["attribute_name"]
            input_doc_id = input_dict["doc_id"]
            _input_key = (
                sim_iter,
                (tr_start, tr_end),
                tr_key,
                input_full_name,
                input_att,
            )
            path = key_to_filename(_input_key, 0)
            if os.path.exists(path):
                data_dict[(input_full_name, input_att)] = load_from_disk(path)
            else:
                not_in_cache.add(input_doc_id)

        if not_in_cache:

            data_dict.update(
                pull_inputs_from_firestore(
                    fs_db=fs_db,
                    auth_data=auth_data,
                    key=_input_key[:-2],
                    doc_ids=list(not_in_cache),
                )
            )

        yield run_key, data_dict


def key_to_filename(run_key, chunk_num):
    key = tuple(list(run_key) + [chunk_num])
    key_hash = hashlib.md5(json.dumps(key, default=str).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key_hash}.json")


def load_from_disk(path):
    """Read raw bytes from disk."""
    with open(path, "r") as f:
        return f.read()


def _get_cache_size():
    global _current_cache_size
    return _current_cache_size


def save_to_disk(run_key, chunk_num, _value_chunk, max_cache_bytes):
    """Write raw bytes to disk and enforce max cache size."""
    path = key_to_filename(run_key, chunk_num)
    with open(path, "w") as f:
        f.write(_value_chunk)
    global _current_cache_size
    _current_cache_size += len(_value_chunk)
    _enforce_max_cache_size(max_cache_bytes)


def _enforce_max_cache_size(max_cache_bytes):
    """Evict oldest files until total size <= max_cache_bytes."""
    total = _get_cache_size()
    if total <= max_cache_bytes:
        return

    files = [
        (os.path.join(CACHE_DIR, f), os.path.getmtime(os.path.join(CACHE_DIR, f)))
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".bin")
    ]
    files.sort(key=lambda x: x[1])  # oldest first

    for path, _ in files:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        total = _get_cache_size()
        if total <= max_cache_bytes:
            break


def get_run_key_iterator(
    full_space: List[Tuple],
    docs_to_run: List[Text],
    sim_iter_nums: List[str] = None,
    time_ranges_keys: List[str] = None,
    ref_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    doc_id_to_full_name: Optional[Dict[Text, Text]] = None,
):
    for sim_iter_num, time_range, time_ranges_key in full_space:
        if sim_iter_nums and sim_iter_num not in sim_iter_nums:
            continue
        if time_ranges_keys and time_ranges_key not in time_ranges_keys:
            continue
        for doc_id in docs_to_run:
            full_name = doc_id_to_full_name[doc_id]
            for att in ref_dict[full_name]:
                yield sim_iter_num, time_range, time_ranges_key, full_name, att

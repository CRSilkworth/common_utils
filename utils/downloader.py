from typing import Dict, Text, Optional, Tuple, List, Any, Set
import json
import hashlib
from utils.serialize_utils import serialize_value
from itertools import islice
from google.cloud.firestore_v1 import FieldFilter, And
import logging
from utils.datetime_utils import to_datetimes

_cache = {}
_cache_order = []  # list for FIFO eviction
_cache_size = 0  # total bytes
MAX_CACHE_BYTES = 500_000_000  # adjust


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
        clone_num: Simulation iteration identifier.
        doc_ids: List of document IDs to fetch blocks for.
        ref_dict: Reference metadata for determining which attributes belong to each doc.

    Returns:
        Dict mapping (input_full_name, attribute_name) -> bytes (_value_chunk).
    """
    user_id = auth_data.get("user_id")
    calc_graph_id = auth_data.get("calc_graph_id")
    data_dict = {}
    clone_num, (time_range_start, time_range_end) = key

    for doc_batch in batched(doc_ids, 30):
        query = (
            fs_db.collection_group("value_file_block")
            .where(filter=FieldFilter("user_id", "==", user_id))
            .where(filter=FieldFilter("calc_graph_id", "==", calc_graph_id))
            .where(filter=FieldFilter("doc_id", "in", doc_batch))
            .where(filter=FieldFilter("clone_num", "==", clone_num))
            .where(filter=FieldFilter("time_range_start", "==", time_range_start))
            .where(filter=FieldFilter("time_range_end", "==", time_range_end))
        )
        latest_versions = {}
        for doc_snap in query.stream():
            data = doc_snap.to_dict()
            for k in ["time_range_start", "time_range_end", "version"]:
                data[k] = to_datetimes(data[k])
            full_name = data.get("full_name")
            version = to_datetimes(data.get("version"))
            att = data.get("attribute_name")
            run_key = (
                data.get("clone_num"),
                (data.get("time_range_start"), data.get("time_range_end")),
                full_name,
                att,
            )
            if (full_name, att) in latest_versions and version <= latest_versions[
                (full_name, att)
            ]:
                continue

            latest_versions[(full_name, att)] = version
            data_dict[(full_name, att)] = data.get("_value_chunk")
            save_to_memory(run_key, 0, data_dict[(full_name, att)])

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
    clone_nums: Optional[List[str]] = None,
) -> None:
    """
    Prefetch and locally cache latest Firestore value_file_blocks for all input nodes
    of the given documents, filtered by clone_nums.

    This respects Firestore’s 30-value limit for each 'in' filter by batching over
    doc_ids, attribute_names, clone_nums.

    Args:
        fs_db: Firestore client
        user_id: Firestore user ID
        calc_graph_id: Firestore calc graph ID
        doc_to_runs: The doc_ids to prefetch inputs for
        ref_dict: Reference dictionary describing dependencies
        doc_id_to_full_name: Mapping of doc_id -> full_name
        clone_nums: Optional list of simulation iteration identifiers
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
        for clone_num in clone_nums:

            query = fs_db.collection_group("value_file_block").where(
                filter=And(
                    filters=[
                        FieldFilter("user_id", "==", user_id),
                        FieldFilter("calc_graph_id", "==", calc_graph_id),
                        FieldFilter("doc_id", "in", doc_batch),
                        FieldFilter("clone_num", "==", clone_num),
                    ]
                )
            )

            for doc_snap in query.stream():
                data = doc_snap.to_dict()
                for k in ["time_range_start", "time_range_end", "version"]:
                    data[k] = to_datetimes(data[k])
                input_full_name = data.get("full_name")
                input_att = data.get("attribute_name")

                _input_key = (
                    data.get("clone_num"),
                    (data.get("time_range_start"), data.get("time_range_end")),
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

                # Cache locally (latest version first)
                save_to_memory(_input_key, 0, data["_value_chunk"])


def cached_stream_subgraph_by_key(
    fs_db,
    auth_data: Dict[Text, Any],
    full_space: List[Tuple],
    docs_to_run: List[Text],
    ref_dict: Dict[str, Dict[str, Dict[str, Any]]],
    clone_nums: Optional[List[str]] = None,
    doc_id_to_full_name: Optional[Dict[Text, Text]] = None,
    doc_full_name_to_id: Optional[Dict[Text, Text]] = None,
):
    """
    Iterate through cached data first, then stream from Firestore and cache results.
    """

    run_key_iterator = get_run_key_iterator(
        full_space=full_space,
        docs_to_run=docs_to_run,
        ref_dict=ref_dict,
        clone_nums=clone_nums,
        doc_id_to_full_name=doc_id_to_full_name,
    )

    for run_key in run_key_iterator:
        sim_iter, (tr_start, tr_end), full_name, att = run_key
        data_dict = {}
        not_in_cache = set()
        for input_dict in ref_dict[full_name][att]["inputs"]:
            input_full_name = input_dict["full_name"]
            input_att = input_dict["attribute_name"]
            _input_key = (
                sim_iter,
                (tr_start, tr_end),
                input_full_name,
                input_att,
            )

            if (_input_key, 0) in _cache:
                data_dict[(input_full_name, input_att)] = _cache.get((_input_key, 0))
            else:
                not_in_cache.add((_input_key, 0))

        if not_in_cache:
            not_found_ids = set([doc_full_name_to_id[k[0][2]] for k in not_in_cache])
            data_dict.update(
                pull_inputs_from_firestore(
                    fs_db=fs_db,
                    auth_data=auth_data,
                    key=_input_key[:-2],
                    doc_ids=sorted(not_found_ids),
                )
            )
            for _input_key, chunk_num in not_in_cache:
                _, _, input_full_name, input_att = _input_key
                if (input_full_name, input_att) in data_dict:
                    continue
                save_to_memory(_input_key, chunk_num, serialize_value(None))

        yield run_key, data_dict


def save_to_memory(run_key, chunk_num, chunk_bytes):
    """chunk_bytes MUST be bytes."""
    global _cache_size

    key = (run_key, chunk_num)
    size = len(chunk_bytes)

    # Remove previous value if exists
    if key in _cache:
        old = _cache.pop(key)
        _cache_size -= len(old)
        _cache_order.remove(key)

    # Evict until room
    while _cache_size + size > MAX_CACHE_BYTES and _cache_order:
        oldest = _cache_order.pop(0)
        old_bytes = _cache.pop(oldest)
        _cache_size -= len(old_bytes)

    # Insert
    _cache[key] = chunk_bytes
    _cache_order.append(key)
    _cache_size += size


def get_run_key_iterator(
    full_space: List[Tuple],
    docs_to_run: List[Text],
    clone_nums: List[str] = None,
    ref_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    doc_id_to_full_name: Optional[Dict[Text, Text]] = None,
):
    for clone_num, time_range in full_space:
        if clone_nums and clone_num not in clone_nums:
            continue
        for doc_id in docs_to_run:
            full_name = doc_id_to_full_name[doc_id]
            for att in ref_dict[full_name]:
                yield clone_num, time_range, full_name, att

from typing import Dict, Text, Optional, Tuple
import requests
import json
import binascii
from operator import itemgetter
from itertools import groupby
import datetime
import hashlib
import os

CACHE_DIR = "/tmp/cache"
MAX_CACHE_BYTES = 2 * 1024**3


def stream_subgraph_by_key(
    auth_data, ref_dict, sim_iter_nums, time_ranges_keys, start_key=None
):
    """Your original unmodified streamer (no caching, no limits)."""
    data = {
        "auth_data": auth_data,
        "ref_dict": ref_dict,
        "time_ranges_keys": list(time_ranges_keys) if time_ranges_keys else None,
        "sim_iter_nums": list(sim_iter_nums) if sim_iter_nums else None,
        "start_key": start_key,
    }
    resp = requests.post(
        f"{auth_data['dash_app_url']}/stream-by-key", json=data, stream=True
    )

    for line in resp.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        batch = json.loads(line.strip())
        batch_data = bytes.fromhex(batch["batch_data"])
        index_map = batch["index_map"]

        data_dict = {}
        current_key = None
        for index_key, loc in index_map.items():
            (
                sim_iter,
                tr_key,
                start_iso,
                end_iso,
                full_name,
                att,
                _,
                input_full_name,
                input_att,
            ) = json.loads(index_key)
            tr_start = datetime.datetime.fromisoformat(start_iso)
            tr_end = datetime.datetime.fromisoformat(end_iso)
            run_key = (sim_iter, (tr_start, tr_end), tr_key, full_name, att)
            offset, length = loc["offset"], loc["length"]
            block_bytes = batch_data[offset : offset + length]  # noqa: E203

            if run_key != current_key:
                if current_key is not None:
                    yield current_key, data_dict
                current_key = run_key
                data_dict = {}

            data_dict[(input_full_name, input_att)] = block_bytes

        if data_dict:
            if None in data_dict:
                data_dict = {}
            yield current_key, data_dict


def _key_to_filename(key):
    key_hash = hashlib.md5(json.dumps(key, default=str).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key_hash}.bin")


_current_cache_size = 0  # module global


def _get_cache_size():
    global _current_cache_size
    return _current_cache_size


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


def save_bytes_to_disk(key, block_bytes, max_cache_bytes):
    """Write raw bytes to disk and enforce max cache size."""
    path = _key_to_filename(key)
    with open(path, "wb") as f:
        f.write(block_bytes)
    global _current_cache_size
    _current_cache_size += len(block_bytes)
    _enforce_max_cache_size(max_cache_bytes)


def _load_bytes_from_disk(path):
    """Read raw bytes from disk."""
    with open(path, "rb") as f:
        return f.read()


def prefetch_subgraph(
    auth_data,
    ref_dict,
    sim_iter_nums,
    time_ranges_keys,
    max_cache_bytes=MAX_CACHE_BYTES,
):
    """
    Pull as much as possible from the server up to max_cache_bytes.
    Stops once the total cache size exceeds that limit.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    for run_key, data_dict in stream_subgraph_by_key(
        auth_data, ref_dict, sim_iter_nums, time_ranges_keys
    ):

        sim_iter, (tr_start, tr_end), tr_key, _, _ = run_key
        start_iso = tr_start.isoformat()
        end_iso = tr_end.isoformat()
        for (input_full_name, input_att), block_bytes in data_dict.items():

            input_key = (
                sim_iter,
                tr_key,
                start_iso,
                end_iso,
                input_full_name,
                input_att,
            )
            if _get_cache_size() + len(block_bytes) > max_cache_bytes:
                return run_key
            print("saving", input_key)
            save_bytes_to_disk(input_key, block_bytes, max_cache_bytes)

    return None


def cached_stream_subgraph_by_key(
    auth_data,
    run_key_iterator,
    ref_dict,
    sim_iter_nums,
    time_ranges_keys,
    start_key: Optional[Tuple] = None,
    max_cache_bytes=MAX_CACHE_BYTES,
):
    """
    Iterate through cached batches first; if not found, pull from the server and cache.
    Automatically enforces cache size after each write.
    """
    # seen_keys = set()

    for run_key in run_key_iterator:
        sim_iter, (tr_start, tr_end), tr_key, full_name, att = run_key
        start_iso = tr_start.isoformat()
        end_iso = tr_end.isoformat()
        if run_key == start_key:
            break
        data_dict = {}
        for input_dict in ref_dict[full_name][att]["inputs"]:
            input_full_name = input_dict["full_name"]
            input_att = input_dict["attribute_name"]
            input_key = (
                sim_iter,
                tr_key,
                start_iso,
                end_iso,
                input_full_name,
                input_att,
            )
            path = _key_to_filename(input_key)
            if os.path.exists(path):
                block_bytes = _load_bytes_from_disk(path)
            else:
                raise ValueError(
                    "Something went wrong and the cached file is missing for"
                    f" {input_key}"
                )
            data_dict[(input_full_name, input_att)] = block_bytes
            # seen_keys.add(json.dumps(key, default=str))
            yield run_key, data_dict

    if start_key:
        start_key = (
            start_key[0],
            start_key[1][0].isoformat(),
            start_key[1][1].isoformat(),
            start_key[2],
            start_key[3],
            start_key[4],
        )
        for run_key, data_dict in stream_subgraph_by_key(
            auth_data, ref_dict, sim_iter_nums, time_ranges_keys, start_key=start_key
        ):
            sim_iter, (tr_start, tr_end), tr_key, full_name, att = run_key
            start_iso = tr_start.isoformat()
            end_iso = tr_end.isoformat()
            for input_dict in ref_dict[full_name][att]["inputs"]:
                input_full_name = input_dict["full_name"]
                input_att = input_dict["attribute_name"]
                input_key = (
                    sim_iter,
                    tr_key,
                    start_iso,
                    end_iso,
                    input_full_name,
                    input_att,
                )
                path = _key_to_filename(input_key)
                if os.path.exists(path):
                    data_dict[input_key] = _load_bytes_from_disk(path)
                else:
                    save_bytes_to_disk(input_key, data_dict[input_key], max_cache_bytes)
            yield run_key, data_dict


class BatchDownloader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        value_file_ref: Text,
        doc_id: Text,
        attribute_name: Text,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range_start: Optional[datetime.datetime] = None,
        time_range_end: Optional[datetime.datetime] = None,
        chunked: bool = False,
    ):
        self.auth_data = auth_data
        self.value_file_ref = value_file_ref
        self.chunked = chunked
        self.doc_id = doc_id
        self.attribute_name = attribute_name
        self.sim_iter_nums = sim_iter_nums
        self.time_ranges_keys = time_ranges_keys
        self.time_range_start = time_range_start
        self.time_range_end = time_range_end

    def flat_iterator(self):
        data = {
            "auth_data": self.auth_data,
            "doc_id": self.doc_id,
            "attribute_name": self.attribute_name,
            "value_file_ref": self.value_file_ref,
            "sim_iter_nums": self.sim_iter_nums,
            "time_ranges_keys": self.time_ranges_keys,
            "time_range_start": (
                self.time_range_start.isoformat() if self.time_range_start else None
            ),
            "time_range_end": (
                self.time_range_end.isoformat() if self.time_range_end else None
            ),
        }
        resp = requests.post(
            f"{self.auth_data['dash_app_url']}/stream-batches", json=data, stream=True
        )
        for chunk in resp.iter_content(chunk_size=None):
            for line in chunk.splitlines():
                if not line.strip():
                    continue
                try:
                    batch = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    print(f"Bad line: {line!r}")
                    continue
                batch_data = binascii.unhexlify(batch["batch_data"])
                for key, loc in batch["index_map"].items():
                    offset, length = loc["offset"], loc["length"]
                    (
                        sim_iter_num,
                        time_ranges_key,
                        time_range_start,
                        time_range_end,
                        chunk_num,
                    ) = json.loads(key)
                    chunk = batch_data[offset : offset + length]  # noqa: E203
                    _value_chunk = chunk.decode("utf-8")
                    yield (
                        sim_iter_num,
                        time_ranges_key,
                        (
                            datetime.datetime.fromisoformat(time_range_start),
                            datetime.datetime.fromisoformat(time_range_end),
                        ),
                        chunk_num,
                        _value_chunk,
                    )

    def __iter__(self):
        flat = self.flat_iterator()
        for (sim_id, coll, tr), group in groupby(flat, key=itemgetter(0, 1, 2)):
            if self.chunked:

                def chunk_gen(g=group):
                    for _, _, _, chunk_num, data in g:
                        yield chunk_num, data

                yield (sim_id, coll, tr), chunk_gen()
            else:
                # If not chunked there is only one element in the group
                _, _, _, _, data = next(group)
                yield (sim_id, coll, tr), data

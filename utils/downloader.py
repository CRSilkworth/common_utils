from typing import Dict, Text, Optional, Tuple, List
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Fetch using stream=False but parse as NDJSON lines."""
    data = {
        "auth_data": auth_data,
        "ref_dict": ref_dict,
        "time_ranges_keys": list(time_ranges_keys) if time_ranges_keys else None,
        "sim_iter_nums": list(sim_iter_nums) if sim_iter_nums else None,
        "start_key": start_key,
    }

    url = f"{auth_data['dash_app_url']}/stream-by-key"

    try:
        resp = requests.post(url, json=data, stream=False)
        resp.raise_for_status()
        # Treat response as lines (even though stream=False)
        for line in resp.content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            batch = json.loads(line)
            yield from _process_batch(batch)
    except Exception as e:
        print(f"FAILED!!! FALLBACK TO STREAMING {e}")
        # fallback to proper streaming
        resp = requests.post(url, json=data, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            batch = json.loads(line.strip())
            yield from _process_batch(batch)


def _process_batch(batch):
    """Turn a single batch dict into run_key -> data_dict items."""
    batch_data = batch["batch_data"]
    index_map = batch["index_map"]

    data_dict = {}
    current_key = None

    for index_key, loc in index_map.items():
        (
            sim_iter,
            start_iso,
            end_iso,
            tr_key,
            full_name,
            att,
            chunk_num,
            input_full_name,
            input_att,
        ) = json.loads(index_key)

        if chunk_num > 0:
            continue  # skip chunked pieces

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
        yield current_key, data_dict


def key_to_filename(run_key, chunk_num):
    key = tuple(list(run_key) + [chunk_num])
    key_hash = hashlib.md5(json.dumps(key, default=str).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key_hash}.bin")


_current_cache_size = 0


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


def save_bytes_to_disk(run_key, chunk_num, block_bytes, max_cache_bytes):
    """Write raw bytes to disk and enforce max cache size."""
    path = key_to_filename(run_key, chunk_num)
    with open(path, "wb") as f:
        f.write(block_bytes)
    global _current_cache_size
    _current_cache_size += len(block_bytes)
    _enforce_max_cache_size(max_cache_bytes)


def load_bytes_from_disk(path):
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
    for run_key, data_dict in parallel_stream_subgraph_by_key(
        auth_data, ref_dict, sim_iter_nums, time_ranges_keys, max_workers=8
    ):

        sim_iter, (tr_start, tr_end), tr_key, _, _ = run_key
        start_iso = tr_start.isoformat()
        end_iso = tr_end.isoformat()
        for (input_full_name, input_att), block_bytes in data_dict.items():
            input_key = (
                sim_iter,
                start_iso,
                end_iso,
                tr_key,
                input_full_name,
                input_att,
            )
            if _get_cache_size() + len(block_bytes) > max_cache_bytes:
                return run_key
            save_bytes_to_disk(input_key, 0, block_bytes, max_cache_bytes)

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
                start_iso,
                end_iso,
                tr_key,
                input_full_name,
                input_att,
            )
            path = key_to_filename(input_key, 0)
            if os.path.exists(path):
                data_dict[(input_full_name, input_att)] = load_bytes_from_disk(path)
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
                    start_iso,
                    end_iso,
                    tr_key,
                    input_full_name,
                    input_att,
                )
                path = key_to_filename(input_key, 0)
                if os.path.exists(path):
                    data_dict[input_key] = load_bytes_from_disk(path)
                elif input_key in data_dict:
                    save_bytes_to_disk(
                        input_key, 0, data_dict[input_key], max_cache_bytes
                    )
            yield run_key, data_dict


def _fetch_single(
    auth_data: Dict, ref_dict: Dict, sim_iter_num, time_ranges_key, start_key=None
):
    """
    Fetch a single request for one sim_iter_num and one time_ranges_key.
    Yields raw batches.
    """
    data = {
        "auth_data": auth_data,
        "ref_dict": ref_dict,
        "time_ranges_keys": [time_ranges_key] if time_ranges_key else None,
        "sim_iter_nums": [sim_iter_num] if sim_iter_num else None,
        "start_key": start_key,
    }
    url = f"{auth_data['dash_app_url']}/stream-by-key"

    resp = requests.post(url, json=data, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        yield json.loads(line)


def parallel_stream_subgraph_by_key(
    auth_data: Dict,
    ref_dict: Dict,
    sim_iter_nums: List[Text],
    time_ranges_keys: List[Text],
    max_workers: int = 8,
):
    """
    Parallelize fetching all sim_iter_num x time_ranges_key requests independently.
    Yields processed batches as they arrive.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sim_iter_num in sim_iter_nums:
            for tr_key in time_ranges_keys:
                future = executor.submit(
                    _fetch_single, auth_data, ref_dict, sim_iter_num, tr_key
                )
                futures[future] = (sim_iter_num, tr_key)

        # as each request completes, yield its batches
        for future in as_completed(futures):
            sim_iter_num, tr_key = futures[future]
            try:
                for (
                    batch
                ) in future.result():  # iterate over batches yielded by _fetch_single
                    yield from _process_batch(batch)
            except Exception as e:
                print(f"Error fetching sim_iter={sim_iter_num}, tr_key={tr_key}: {e}")


class BatchDownloader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        value_file_ref: Text,
        doc_id: Text,
        full_name: Text,
        attribute_name: Text,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range_start: Optional[datetime.datetime] = None,
        time_range_end: Optional[datetime.datetime] = None,
        full_space: Optional[List[Tuple]] = None,
        chunked: bool = False,
        use_cache: bool = True,
    ):
        self.auth_data = auth_data
        self.value_file_ref = value_file_ref
        self.chunked = chunked
        self.doc_id = doc_id
        self.full_name = full_name
        self.attribute_name = attribute_name
        self.sim_iter_nums = sim_iter_nums
        self.time_ranges_keys = time_ranges_keys
        self.time_range_start = time_range_start
        self.time_range_end = time_range_end
        self.use_cache = use_cache
        if use_cache and full_space is None:
            raise ValueError("Must provide full space when use_cache=True")
        self.full_space = full_space

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
                batch_data = batch["batch_data"]
                for key, loc in batch["index_map"].items():
                    offset, length = loc["offset"], loc["length"]
                    (
                        sim_iter_num,
                        time_range_start,
                        time_range_end,
                        time_ranges_key,
                        full_name,
                        att,
                        chunk_num,
                    ) = json.loads(key)
                    chunk = batch_data[offset : offset + length]  # noqa: E203
                    _value_chunk = chunk.decode("utf-8")
                    yield (
                        sim_iter_num,
                        (
                            datetime.datetime.fromisoformat(time_range_start),
                            datetime.datetime.fromisoformat(time_range_end),
                        ),
                        time_ranges_key,
                        full_name,
                        att,
                        chunk_num,
                        _value_chunk,
                    )

    def cache_iterator(self):
        for sim_iter_num, time_range, time_ranges_key in self.full_space:

            _run_key = (
                sim_iter_num,
                time_range[0].isoformat(),
                time_range[1].isoformat(),
                time_ranges_key,
                self.full_name,
                self.attribute_name,
            )
            chunk_num = 0
            path = key_to_filename(_run_key, chunk_num)
            while os.path.exists(path):
                _value_chunk = load_bytes_from_disk(path).decode("utf-8")
                yield (
                    sim_iter_num,
                    time_range,
                    time_ranges_key,
                    self.full_name,
                    self.attribute_name,
                    chunk_num,
                    _value_chunk,
                )
                chunk_num += 1
                path = key_to_filename(_run_key, chunk_num)

    def merged_iterator(self):
        """
        Yield items in the order defined by full_space.
        Prefer cached data if available; otherwise pull sequentially
        from the flat iterator (which must be ordered the same way).
        Saves streamed chunks to cache as it goes.
        """
        flat_iter = self.flat_iterator()
        next_flat = None
        for key in self.full_space:
            sim_iter_num, time_range, time_ranges_key = key
            if (
                self.sim_iter_nums is not None
                and sim_iter_num not in self.sim_iter_nums
            ):
                continue
            if (
                self.time_ranges_keys is not None
                and time_ranges_key not in self.time_ranges_keys
            ):
                continue

            if (
                self.time_range_start is not None
                and self.time_range_start > time_range[0]
            ):
                continue
            if self.time_range_end is not None and self.time_range_end < time_range[1]:
                continue

            run_key = (*key, self.full_name, self.attribute_name)

            _run_key = (
                sim_iter_num,
                time_range[0].isoformat(),
                time_range[1].isoformat(),
                time_ranges_key,
                self.full_name,
                self.attribute_name,
            )
            key_with_none = (*run_key, 0, None)
            path = key_to_filename(_run_key, 0)

            chunk_num = 0
            found_cached = False
            while True:
                path = key_to_filename(_run_key, chunk_num)
                if not os.path.exists(path):
                    break
                found_cached = True
                _value_chunk = load_bytes_from_disk(path).decode("utf-8")
                yield (*run_key, chunk_num, _value_chunk)
                chunk_num += 1

            if found_cached:
                continue

            if next_flat is None:
                try:
                    next_flat = next(flat_iter)
                except StopIteration:
                    next_flat = None

            if next_flat is not None:

                f_key = next_flat[:-2]
                if f_key == run_key:
                    block_bytes = next_flat[-1].encode("utf-8")
                    save_bytes_to_disk(_run_key, 0, block_bytes, MAX_CACHE_BYTES)

                    yield next_flat

                    next_flat = None
                else:
                    yield key_with_none
            else:
                yield key_with_none

    def __iter__(self):
        if self.use_cache:
            flat = self.merged_iterator()
        else:
            flat = self.flat_iterator()
        for (sim_id, tr, coll, fn, att), group in groupby(
            flat, key=itemgetter(0, 1, 2, 3, 4)
        ):
            if self.chunked:

                def chunk_gen(g=group):
                    for _, _, _, _, _, chunk_num, data in g:
                        yield chunk_num, data

                yield (sim_id, tr, coll, fn, att), chunk_gen()
            else:
                # If not chunked there is only one element in the group
                _, _, _, _, _, _, data = next(group)
                yield (sim_id, tr, coll, fn, att), data

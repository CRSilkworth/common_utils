from typing import Dict, Text, Optional
import requests
import json
import binascii
from operator import itemgetter
from itertools import groupby
import datetime
import io


def stream_subgraph_by_key(auth_data, value_file_ref_groups):
    data = {
        "auth_data": auth_data,
        "value_file_ref_groups": value_file_ref_groups,
    }
    resp = requests.post(
        f"{auth_data['dash_app_url']}/stream-by-key", json=data, stream=True
    )

    # wrap response in a text buffer to read line by line
    print(resp.raw, encoding="utf-8")
    for line in io.TextIOWrapper(resp.raw, encoding="utf-8"):
        batch = json.loads(line.strip())
        batch_data = bytes.fromhex(batch["batch_data"])
        index_map = batch["index_map"]

        data_dict = {}
        current_key = None
        for index_key, loc in index_map.items():
            sim_iter, tr_key, start_iso, end_iso, _, vf_id, group_idx = json.loads(
                index_key
            )
            tr_start = datetime.datetime.fromisoformat(start_iso)
            tr_end = datetime.datetime.fromisoformat(end_iso)
            key = (sim_iter, (tr_start, tr_end), tr_key, group_idx)

            offset, length = loc["offset"], loc["length"]
            block_bytes = batch_data[offset : offset + length]  # noqa: E203

            if key != current_key:
                if current_key is not None:
                    yield current_key, data_dict
                current_key = key
                data_dict = {}

            data_dict[vf_id] = block_bytes

        # yield the last key in this batch
        if data_dict:
            yield current_key, data_dict


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

    # def nested_iterator(self):
    #     flat = self.flat_iterator()
    #     for sim_iter_num, sim_param_group in groupby(flat, key=itemgetter(0)):

    #         def tr_name_gen(group=sim_param_group):
    #             for collection_name, collection_group in groupby(
    #                 group, key=itemgetter(1)
    #             ):

    #                 def tr_gen(group=collection_group):
    #                     for time_range, time_range_group in groupby(
    #                         group, key=itemgetter(2)
    #                     ):

    #                         def idx_gen(group=time_range_group):
    #                             for chunk_num, chunk in group:
    #                                 self.last_seen["chunk_num"] = chunk_num
    #                                 yield chunk_num, chunk

    #                         to_yield = idx_gen() if self.chunked else next(idx_gen())
    #                         yield time_range, to_yield

    #                 yield collection_name, tr_gen()

    #         yield sim_iter_num, tr_name_gen()

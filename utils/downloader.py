from typing import Dict, Text, Optional
import requests
import json
import binascii
from operator import itemgetter
from itertools import groupby


class BatchDownloader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        value_file_ref: Text,
        sim_param_keys: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range_start: Optional[Text] = None,
        time_range_end: Optional[Text] = None,
        chunked: bool = False,
    ):
        self.auth_data = auth_data
        self.value_file_ref = value_file_ref
        self.chunked = chunked
        self.sim_param_keys = sim_param_keys
        self.time_ranges_keys = time_ranges_keys
        self.time_range_start = time_range_start
        self.time_range_end = time_range_end

    def flat_iterator(self):
        data = {
            "auth_data": self.auth_data,
            "value_file_ref": str(self.value_file_ref),
            "sim_param_keys": self.sim_param_keys,
            "time_ranges_keys": self.time_ranges_keys,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
        }
        resp = requests.get(
            f"{self.auth_data['dash_app_url']}/stream-batches", json=data, stream=True
        )
        for line in resp.iter_lines():
            if not line:
                continue
            batch = json.loads(line)
            batch_data = binascii.unhexlify(batch["batch_data"])
            for key, loc in batch["index_map"].items():
                offset, length = loc["offset"], loc["length"]
                (
                    sim_param_key,
                    time_ranges_key,
                    time_range_start,
                    time_range_end,
                    chunk_num,
                ) = json.loads(key)
                chunk = batch_data[offset : offset + length]  # noqa: E203
                _value_chunk = chunk.decode("utf-8")
                yield (
                    sim_param_key,
                    time_ranges_key,
                    (time_range_start, time_range_end),
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
                _, _, _, _, data = next(group)
                yield (sim_id, coll, tr), data

    # def nested_iterator(self):
    #     flat = self.flat_iterator()
    #     for sim_param_key, sim_param_group in groupby(flat, key=itemgetter(0)):

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

    #         yield sim_param_key, tr_name_gen()

from typing import Dict, Text
import io
import requests
import json


class BatchUploader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        value_file_ref: Text,
        max_batch_bytes: int = 1e7,
    ):
        self.auth_data = auth_data
        self.max_batch_bytes = max_batch_bytes
        self.value_file_ref = value_file_ref
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        self.size = 0

    def add_chunk(
        self,
        sim_param_key,
        time_ranges_key,
        time_range,
        chunk_num,
        _value_chunk,
    ):
        data = _value_chunk.encode("utf-8")
        offset = self.buffer.tell()
        self.buffer.write(data)
        key = json.dumps(
            [
                sim_param_key,
                time_ranges_key,
                time_range[0],
                time_range[1],
                chunk_num,
            ]
        )
        length = len(data)
        self.index_map[key] = {"offset": offset, "length": length}

        self.item_count += 1
        self.size += length

        if self.buffer.tell() >= self.max_batch_bytes:
            self.flush_batch()

    def flush_batch(self):
        if self.item_count == 0:
            return
        self.buffer.seek(0)
        # send entire batch in one request
        files = {"batch_file": self.buffer}
        resp = requests.post(
            f"{self.auth_data['dash_app_url']}/upload-batch",
            files=files,
            data={
                "payload": json.dumps(
                    {
                        "index_map": self.index_map,
                        "auth_data": self.auth_data,
                        "value_file_ref": self.value_file_ref,
                    }
                )
            },
        )
        assert resp.ok, resp.text
        # reset buffer
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0

    def finalize(self):
        self.flush_batch()

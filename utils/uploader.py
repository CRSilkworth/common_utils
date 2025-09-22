from typing import Dict, Text, Optional
from utils.type_utils import TimeRange
import io
import requests
import json


class BatchUploader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        value_file_ref: Text,
        old_value_file_ref: Optional[Text] = None,
        max_batch_bytes: int = 1e7,
        run_at: Optional[Text] = None,
    ):
        self.auth_data = auth_data
        self.max_batch_bytes = max_batch_bytes
        self.value_file_ref = value_file_ref
        self.old_value_file_ref = old_value_file_ref
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        self.size = 0
        self.run_at = run_at

    def add_chunk(
        self,
        sim_iter_num: Text,
        time_ranges_key: Text,
        time_range: TimeRange,
        chunk_num: int,
        _value_chunk: Text,
        preview: Optional[Text] = None,
        _schema: Optional[Text] = None,
        overriden: bool = False,
    ):
        data = _value_chunk.encode("utf-8")
        offset = self.buffer.tell()
        self.buffer.write(data)
        key = json.dumps(
            [
                sim_iter_num,
                time_ranges_key,
                (time_range[0].isoformat() if time_range[0] is not None else None),
                (time_range[1].isoformat() if time_range[1] is not None else None),
                chunk_num,
            ]
        )
        length = len(data)
        self.index_map[key] = {
            "offset": offset,
            "length": length,
            "preview": preview,
            "_schema": _schema,
            "overriden": overriden,
        }

        self.item_count += 1
        self.size += length

        if length > self.max_batch_bytes:
            return (
                False,
                f"_value chunk ({length}) larger than max_batch_bytes:"
                f" ({self.max_batch_bytes})",
            )

        if self.buffer.tell() >= self.max_batch_bytes:
            return self.flush_batch()
        return True, ""

    def flush_batch(self):
        if self.item_count == 0:
            return
        self.buffer.seek(0)
        # send entire batch in one request
        files = {"batch_file": self.buffer}
        try:
            resp = requests.post(
                f"{self.auth_data['dash_app_url']}/upload-batch",
                files=files,
                data={
                    "payload": json.dumps(
                        {
                            "run_at": self.run_at,
                            "index_map": self.index_map,
                            "auth_data": self.auth_data,
                            "value_file_ref": self.value_file_ref,
                            "old_value_file_ref": self.old_value_file_ref,
                        }
                    )
                },
            )
        except requests.RequestException as e:
            return False, str(e)

        # reset buffer
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        return resp.ok, resp.text

    def finalize(self):
        self.flush_batch()

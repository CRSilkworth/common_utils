from typing import Dict, Text, Optional, Tuple
import io
import requests
import json


class BatchUploader:
    def __init__(
        self,
        auth_data: Dict[Text, Text],
        max_batch_bytes: int = 1e7,
        run_at: Optional[Text] = None,
    ):
        self.auth_data = auth_data
        self.max_batch_bytes = max_batch_bytes
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        self.size = 0
        self.run_at = run_at

    def add_chunk(
        self,
        _run_key: Tuple[Text],
        chunk_num: int,
        _value_chunk: Text,
        new_value_file_ref: Text,
        preview: Text = "",
        _schema: Text = "",
        overriden: bool = False,
        old_value_file_ref: Text = "",
    ):
        if overriden and not old_value_file_ref:
            raise ValueError("Must supply old_value_file_ref when overriden")
        data = _value_chunk.encode("utf-8")
        offset = self.buffer.tell()
        self.buffer.write(data)
        key = json.dumps(list(_run_key) + [chunk_num])
        length = len(data)
        self.index_map[key] = {
            "offset": offset,
            "length": length,
            "new_value_file_ref": new_value_file_ref,
            "preview": preview,
            "_schema": _schema,
            "overriden": overriden,
            "old_value_file_ref": old_value_file_ref,
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

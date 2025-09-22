import websocket
import json
import datetime
import io
from utils.preview_utils import value_to_preview
from utils.type_utils import describe_json_schema
from utils.serialize_utils import attempt_serialize


class DataClient:
    def __init__(self, auth_data, max_batch_bytes=5 * 1024 * 1024):
        base_url = (
            auth_data["dash_app_url"].replace("http://", "").replace("https://", "")
        )
        base_url = f"ws://{base_url}"
        self.upload_url = f"{base_url}/ws-stream"
        self.auth_data = auth_data

        self.stream_url = f"{base_url}/ws-stream"
        self.ws_stream = websocket.WebSocket()
        self.ws_stream.connect(self.stream_url)

        self.upload_url = f"{base_url}/ws-upload"
        self.ws_upload = websocket.WebSocket()
        self.ws_upload.connect(self.upload_url)

        # batching state
        self.max_batch_bytes = max_batch_bytes
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        self.size = 0
        self.run_at = datetime.datetime.utcnow().isoformat()

    def stream_subgraph_by_key(self, value_file_ref_groups):
        self.ws_stream.send(
            json.dumps(
                {
                    "auth_data": self.auth_data,
                    "value_file_ref_groups": value_file_ref_groups,
                }
            )
        )

        current_key = None
        data_dict = {}

        while True:
            msg = json.loads(self.ws_stream.recv())

            if msg["type"] == "batch":
                batch_data = bytes.fromhex(msg["batch_data"])
                index_map = msg["index_map"]

                for index_key, loc in index_map.items():
                    (
                        sim_iter,
                        tr_key,
                        start_iso,
                        end_iso,
                        chunk_num,
                        vf_id,
                        group_idx,
                    ) = json.loads(index_key)
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

            elif msg["type"] == "done":
                if data_dict:
                    yield current_key, data_dict
                break

            elif msg["type"] == "error":
                raise RuntimeError(f"Server error: {msg['message']}")

    def upload_batch(self, value_file_ref, batch_data, index_map):
        self.ws_upload.send(
            json.dumps(
                {
                    "auth_data": self.auth_data,
                    "batch_data": batch_data.hex(),
                    "index_map": index_map,
                }
            )
        )
        ack = json.loads(self.ws_upload.recv())
        if ack.get("type") != "upload_ack":
            raise RuntimeError(f"Upload failed: {ack}")
        return ack

    def add_chunk(
        self,
        value_file_ref,
        value_type,
        sim_iter_num,
        time_ranges_key,
        time_range,
        chunk_num,
        value_chunk,
        overriden=False,
        old_value_file_ref=None,
    ):

        if not overriden:
            preview = value_to_preview(value_chunk)
            _schema = json.dumps(describe_json_schema(value_chunk))
            _value_chunk, _ = attempt_serialize(
                value_chunk,
                value_type,
            )
        else:
            preview = ""
            _schema = ""
            _value_chunk = ""

        data = _value_chunk.encode("utf-8")
        offset = self.buffer.tell()
        self.buffer.write(data)
        key = json.dumps(
            [
                sim_iter_num,
                time_ranges_key,
                (time_range[0].isoformat() if time_range[0] else None),
                (time_range[1].isoformat() if time_range[1] else None),
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
            "value_file_ref": value_file_ref,
            "old_value_file_ref": old_value_file_ref,
        }

        self.item_count += 1
        self.size += length

        if length > self.max_batch_bytes:
            return False, f"Chunk too large ({length} > {self.max_batch_bytes})"

        if self.buffer.tell() >= self.max_batch_bytes:
            return self.flush_batch()

        return True, ""

    def flush_batch(self):
        if self.item_count == 0:
            return True, ""

        self.buffer.seek(0)
        batch_data = self.buffer.read()

        self.upload_batch(
            batch_data=batch_data,
            index_map=self.index_map,
        )

        # reset buffer
        self.buffer = io.BytesIO()
        self.index_map = {}
        self.item_count = 0
        return True, ""

    def finalize(self):
        return self.flush_batch()

    def close(self):
        self.ws_stream.close()
        self.ws_upload.close()

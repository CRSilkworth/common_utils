from typing import Text, Iterable, Any, Optional, Dict, List, Tuple
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.type_utils import TimeRange, describe_json_schema
from utils.function_utils import create_function
from utils.preview_utils import value_to_preview
from utils.type_utils import get_known_types
from utils.datetime_utils import convert_timestamps
from utils.serialize_utils import encode_obj
import json
import requests
import os
import datetime
import inspect
from utils.downloader import (
    save_to_disk,
    key_to_filename,
    load_from_disk,
    MAX_CACHE_BYTES,
)
from operator import itemgetter
from itertools import groupby
import logging
from utils.datetime_utils import normalize_datetime


class Attribute:
    def __init__(
        self,
        name: Text,
        auth_data: Dict[Text, Any],
        doc_id: Text,
        value_type: Any,
        doc_full_name: Text,
    ):
        self.name = name
        self.auth_data = auth_data
        self.doc_id = doc_id
        self._val = None
        self.doc_full_name = doc_full_name
        self.value_type = value_type
        self.sim_iter_num = None
        self.time_ranges_key = None
        self.time_range = None
        self.context_key = (None, None, None)
        self.outputs = {}
        self.cleanup = None
        self.runnable = False
        self.full_space = None
        self.run_key = None

    def _set_context(self, **kwargs):
        self.sim_iter_num = kwargs.get("sim_iter_num", None)
        self.time_ranges_key = kwargs.get("time_ranges_key", None)
        self.time_range = kwargs.get("time_range", None)
        self.context_key = (self.sim_iter_num, self.time_ranges_key, self.time_range)

        if not self.sim_iter_num or not self.time_ranges_key or not self.time_range:
            self.run_key = None
            return

        self.run_key = (
            self.sim_iter_num,
            self.time_range,
            self.time_ranges_key,
            self.doc_full_name,
            self.name,
        )

    @property
    def val(self) -> Any:
        return self._val

    def _set_val(self, val: Any, serialized: bool = False):
        if serialized:
            val, output, cleanups = attempt_deserialize(val, self.value_type)
            self._add_output(output)
            self.cleanup = cleanups[0] if cleanups else None

        self._val = val

    def _set_full_space(self, full_space: List[Tuple]):
        self.full_space = full_space

    def _add_output(self, output: Dict[Text, Any]):
        if not output:
            return
        self.outputs.setdefault(self.context_key, [])
        self.outputs[self.context_key].append(output)

    def _get_output(self) -> Dict[Text, Any]:
        output = {
            "failed": [],
            "combined_output": [],
            "stdout_output": [],
            "stderr_output": [],
        }
        for context_key, outputs in self.outputs.items():
            stdout = []
            stderr = []
            combined = []
            for _output in outputs:
                output["failed"].append(_output["failed"])
                combined.append(_output["combined_output"].strip())
                stdout.append(_output["stdout_output"].strip())
                stderr.append(_output["stderr_output"].strip())
            combined = "\n".join(combined).strip()
            stdout = "\n".join(stdout).strip()
            stderr = "\n".join(stderr).strip()
            if combined:
                output["combined_output"].append(
                    f"While running {context_key}:\n{combined}"
                )
            if stdout:
                output["stdout_output"].append(
                    f"While running {context_key}:\n{stdout}"
                )
            if stderr:
                output["stderr_output"].append(
                    f"While running {context_key}:\n{stderr}"
                )

        output["failed"] = any(output["failed"])
        output["combined_output"] = "\n".join(output["combined_output"])
        output["stderr_output"] = "\n".join(output["stderr_output"])
        output["stdout_output"] = "\n".join(output["stdout_output"])
        return output

    def _clear_output(self):
        self.outputs = {}

    def _send_output(
        self,
        caller: Optional[Text] = None,
    ):
        doc_id = self.doc_id

        # Send the attribute result back to the backend
        data = {
            "docs_to_run": [doc_id],
            "outputs": {doc_id: {self.name: self._get_output()}},
            "caller": caller,
            "auth_data": self.auth_data,
            "run_completed": False,
            "run_output": {"failed": False, "message": ""},
        }
        requests.post(
            os.path.join(self.auth_data["dash_app_url"], "job-result"),
            json=data,
            headers={"Authorization": f"Bearer {self.auth_data['token']}"},
        )


class RunnableAttribute(Attribute):
    def __init__(
        self,
        name: Text,
        fs_db: Any,
        auth_data: Dict[Text, Text],
        doc_id: Text,
        doc_full_name: Text,
        value_type: Any,
        var_name_to_id: Dict[Text, Text],
        sim_iter_nums: List[Text],
        time_ranges_keys: List[Text],
        function_name: Text,
        function_header: Text,
        function_string: Text,
        new_version: Optional[Text] = None,
        old_version: Optional[Text] = None,
        chunked: bool = False,
        no_function_body: bool = False,
        overrides: Optional[List[Tuple]] = None,
        global_vars: Dict[Text, Any] = None,
        deleted: bool = False,
    ):

        super().__init__(
            name=name,
            auth_data=auth_data,
            doc_id=doc_id,
            doc_full_name=doc_full_name,
            value_type=value_type,
        )
        self.auth_data = auth_data
        self.new_version = (
            datetime.datetime.fromisoformat(new_version) if new_version else None
        )
        self.old_version = (
            datetime.datetime.fromisoformat(old_version) if old_version else None
        )
        self._val = None
        self.chunked = chunked
        self.sim_iter_num = None
        self.time_ranges_key = None
        self.time_range = None
        self.outputs = {}
        self.runnable = True

        self.overrides = overrides or []
        overrides = []
        for override in self.overrides:
            override[1] = (
                convert_timestamps(override[1][0]),
                convert_timestamps(override[1][1]),
            )
            overrides.append(override)
        self.overrides = overrides

        self.no_function_body = no_function_body
        self.function_name = function_name
        self.function_header = function_header
        self.function_string = function_string
        self.sim_iter_nums = sim_iter_nums
        self.time_ranges_keys = time_ranges_keys
        self.var_name_to_id = var_name_to_id

        self.deleted = deleted
        self.fs_db = fs_db
        self.func, output = create_function(
            function_name=self.function_name,
            function_header=self.function_header,
            function_string=self.function_string,
            allowed_modules=get_known_types(),
            global_vars=global_vars if global_vars is not None else {},
        )
        self._add_output(output)

    def _deserialize(self, iterator: Iterable):
        if self.chunked:
            for key, _value in iterator:

                def value_chunk_gen(_value=_value):
                    for chunk_num, _value_chunk in enumerate(_value):
                        path = key_to_filename(key, chunk_num)
                        if os.path.exists(path):
                            _value_chunk = load_from_disk(path)
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )

                        yield value_chunk

                yield key, value_chunk_gen(), {}
        else:
            for key, _value in iterator:
                path = key_to_filename(key, 0)
                if os.path.exists(path):
                    _value = load_from_disk(path)
                value, output, _ = attempt_deserialize(_value, self.value_type)
                yield key, value, output

    def _get_collection(self):
        return (
            self.fs_db.collection("user")
            .document(self.auth_data.get("user_id"))
            .collection("calc_graph")
            .document(self.auth_data.get("calc_graph_id"))
            .collection("doc")
            .document(self.doc_id)
            .collection("attribute")
            .document(self.name)
            .collection("value_file_block")
        )

    def _delete_value_file_blocks(self, version: Optional[Text] = None):
        query = self._get_collection()
        if version:
            query = query.where("version", "==", version)

        docs = query.stream()
        batch = self.fs_db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()

    def _upload_chunk(
        self,
        run_key: Tuple[Text],
        value_chunk: Any,
        chunk_num: int = 0,
        overriden: bool = False,
    ):
        logging.warning(f"OVERRIDEN: {overriden}")
        if overriden:
            try:
                _, value_chunk = next(
                    self.get_iterator(
                        sim_iter_nums=[self.sim_iter_num],
                        time_ranges_keys=[self.time_ranges_key],
                        time_range=self.time_range,
                        version=self.old_version,
                        use_cache=False,
                    )
                )
                logging.warning(f"Overriden value: {type(value_chunk)}, {value_chunk}")
            except StopIteration:
                logging.warning(
                    f"Overriden value not found: {self.run_key}, {self.old_version}"
                )
        preview = value_to_preview(value_chunk)
        _schema = json.dumps(describe_json_schema(value_chunk))
        logging.warning(("value_chunk", type(value_chunk), self.value_type))
        _value_chunk, output = attempt_serialize(value_chunk, self.value_type)
        # _value_chunk = encode_obj(value_chunk)

        logging.warning(("_value_chunk", type(_value_chunk)))
        self._add_output(output)
        if output.get("failed", False):
            return
        preview = value_to_preview(value_chunk)
        _schema = json.dumps(describe_json_schema(value_chunk))
        logging.warning(("_value_chunk", type(_value_chunk)))
        save_to_disk(run_key, chunk_num, _value_chunk, MAX_CACHE_BYTES)

        # Test size limit
        if len(_value_chunk) > MAX_CACHE_BYTES:
            raise ValueError(
                f"Chunk size {len(_value_chunk)} exceeds max batch size "
                f"{self.max_batch_bytes}"
            )

        doc_data = {
            "user_id": self.auth_data["user_id"],
            "calc_graph_id": self.auth_data["calc_graph_id"],
            "doc_id": self.doc_id,
            "attribute_name": self.name,
            "full_name": self.doc_full_name,
            "sim_iter_num": run_key[0],
            "time_range_start": run_key[1][0],
            "time_range_end": run_key[1][1],
            "time_ranges_key": run_key[2],
            "chunk_num": chunk_num,
            "version": self.new_version,
            "_value_chunk": _value_chunk,
            "preview": preview,
            "_schema": _schema,
            "overriden": overriden,
        }
        version_key = (
            f"{doc_data['sim_iter_num']}_"
            f"{doc_data['time_range_start'].isoformat()}_"
            f"{doc_data['time_range_end'].isoformat()}_"
            f"{doc_data['time_ranges_key']}_"
            f"{doc_data['chunk_num']}_"
            f"{doc_data['version']}"
        )
        # logging.warning(("upload", doc_data))
        # Write directly to Firestore
        chunk_ref = self._get_collection().document(version_key)
        chunk_ref.set(doc_data)
        # for d in self.fs_db.collection_group("value_file_block").stream():
        #     logging.warning(("upload row", d.to_dict()))

    def _get_output(self):
        output = super()._get_output()
        output["version"] = self.new_version.isoformat() if self.new_version else None
        return output

    def _finalize(self):
        """
        Deduplicate and update value_file_blocks in Firestore:
        - Deletes duplicates, keeping only the highest version per
        (sim_iter_num, time_range_start, time_range_end, time_ranges_key)
        - Deletes any docs not in current sim_iter_nums or time_ranges_keys
        - Updates remaining docs to `new_version` if they’re not already at that version.
        """

        batch = self.fs_db.batch()
        batch_count = 0
        total_deleted = 0
        total_updated = 0

        # Step 1: Collect all docs, group by key
        blocks_by_key = {}
        to_delete = []

        for snap in self._get_collection().stream():
            data = snap.to_dict()
            for k in ["time_range_start", "time_range_end", "version"]:
                data[k] = normalize_datetime(data[k])
            block_key = (
                f"{data['sim_iter_num']}_"
                f"{data['time_range_start'].isoformat()}_"
                f"{data['time_range_end'].isoformat()}_"
                f"{data['time_ranges_key']}"
            )

            # Skip docs not in allowed sim_iter_nums / time_ranges_keys
            if (
                self.sim_iter_nums
                and data.get("sim_iter_num") not in self.sim_iter_nums
            ) or (
                self.time_ranges_keys
                and data.get("time_ranges_key") not in self.time_ranges_keys
            ):

                batch.delete(snap.reference)
                total_deleted += 1
                batch_count += 1
                continue

            # Track highest version per block key
            current_version = normalize_datetime(data.get("version"))

            if block_key in blocks_by_key:
                existing = blocks_by_key[block_key]
                existing["version"] = normalize_datetime(existing["version"])

                if current_version > existing["version"]:
                    to_delete.append(existing["ref"])
                    blocks_by_key[block_key] = {
                        "ref": snap.reference,
                        "version": current_version,
                    }
                else:
                    # Current doc is older → delete it
                    to_delete.append(snap.reference)
            else:
                blocks_by_key[block_key] = {
                    "ref": snap.reference,
                    "version": current_version,
                }

            # Commit batch periodically
            if batch_count >= 400:
                batch.commit()
                batch = self.fs_db.batch()
                batch_count = 0

        # Commit any initial deletions
        if batch_count:
            batch.commit()
            batch = self.fs_db.batch()
            batch_count = 0

        # Delete duplicates
        for ref in to_delete:
            batch.delete(ref)
            total_deleted += 1
            batch_count += 1
            if batch_count >= 400:
                batch.commit()
                batch = self.fs_db.batch()
                batch_count = 0

        if batch_count:
            batch.commit()
            batch = self.fs_db.batch()
            batch_count = 0

        # Step 2: Update remaining docs to new_version if needed
        versions_to_delete = set()
        for block in blocks_by_key.values():
            ref = block["ref"]
            data = ref.get().to_dict()
            old_version = normalize_datetime(data.get("version"))
            if old_version != self.new_version:
                batch.update(ref, {"version": self.new_version})
                total_updated += 1
                batch_count += 1
                versions_to_delete.add(old_version)

            if batch_count >= 400:
                batch.commit()
                batch = self.fs_db.batch()
                batch_count = 0

        if batch_count:
            batch.commit()
            batch = self.fs_db.batch()
            batch_count = 0

        # Step 3: Delete any remaining old versions
        if versions_to_delete:
            query = self._get_collection().where(
                "version", "in", list(versions_to_delete)
            )
            for doc_ref in query.stream():
                batch.delete(doc_ref.reference)
                batch_count += 1
                if batch_count >= 400:
                    batch.commit()
                    batch = self.fs_db.batch()
                    batch_count = 0

            if batch_count:
                batch.commit()

        logging.warning(
            f"✅ Finalized Firestore blocks: {total_deleted} deleted, {total_updated} updated to version {self.new_version}."
        )

    def time_series(
        self,
        sim_iter_num: Optional[int] = None,
        time_ranges_key: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ) -> Iterable[Tuple[TimeRange, Any]]:
        """
        Get an iterator of the full time series of values (or section of it) of the
        attribute (up to this point in time)
        Args:
            sim_iter_num: Which simulation to pull the time series from
                (defaults to the current simulation)
            time_ranges_key: Which time range collection to pull the data from
                (defaults to the current time range)
            time_range: max and min time range of the data. Defaults to full time
                series (up to this point).
        Returns:
            iterator of 2-tuples:
                time_range: The time range at which that value was computed.
                value: The value at that time range.
        """

        sim_iter_num = sim_iter_num or self.sim_iter_num
        if sim_iter_num not in self.sim_iter_nums:
            raise ValueError(
                f"sim_iter_num {sim_iter_num} not in {self.doc_full_name}-{self.name}"
                f" list of sim_iter_nums: {self.sim_iter_nums}"
            )
        time_ranges_key = time_ranges_key or self.time_ranges_key
        if time_ranges_key not in self.time_ranges_keys:
            raise ValueError(
                f"time_ranges_key {time_ranges_key} not in "
                f"{self.doc_full_name}-{self.name} list of time_ranges_keys:"
                f" {self.time_ranges_keys}"
            )

        time_range = time_range or (datetime.datetime.min, datetime.datetime.max)
        # Only take data that has been 'completed' already
        if time_range[1] > self.time_range[0]:
            time_range = (time_range[0], self.time_range[0])
        iterator = self.get_iterator(
            sim_iter_nums=[sim_iter_num],
            time_ranges_keys=[time_ranges_key],
            time_range=time_range,
        )
        for (_, time_range, _, _, _), data in iterator:
            yield (time_range, data)

    def sims(
        self,
        time_ranges_key: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ) -> Iterable[Tuple[int, Any]]:
        """
        Get an iterator of the data of all the simulations up until this one.
        Note that there is no restriction on selecting a time_range that includes
        several time ranges of data. May get multiple values corresponding to a single
        sim. Use time_range with caution or leave blank.
        Args:
            time_ranges_key: Which time range collection to pull the data from
                (defaults to the current time range)
            time_range: max and min time range of the data. Defaults to the current one.
        Returns:
            iterator of 2-tuples:
                sim_iter_num: The simulation number that value was computed at
                value: The value for that simulation number.
        """
        time_ranges_key = time_ranges_key or self.time_ranges_key
        if time_ranges_key not in self.time_ranges_keys:
            raise ValueError(
                f"time_ranges_key {time_ranges_key} not in "
                f"{self.doc_full_name}-{self.name} list of time_ranges_keys:"
                f" {self.time_ranges_keys}"
            )

        # NOTE: No restriction on taking multiple time ranges
        time_range = time_range or self.time_range

        # Get all previous sims
        sim_iter_nums = list(range(self.sim_iter_num))

        iterator = self.get_iterator(
            sim_iter_nums=sim_iter_nums,
            time_ranges_keys=[time_ranges_key],
            time_range=time_range,
        )
        for (sim_iter_num, _, _, _, _), data in iterator:
            yield (sim_iter_num, data)

    def get_iterator(
        self,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        use_cache: bool = True,
        version: Optional[Text] = None,
    ) -> Iterable[Tuple[Tuple[int, Text, TimeRange], Any]]:
        """
        Get an iterator of some slice of the data computed up until this point.
        Args:
            sim_iter_nums: Which simulations to pull the data from
                (defaults to all simulations)
            time_ranges_key: Which time range collections to pull the data from
                (defaults to all time ranges collections)
            time_range: max and min time range of the data. Defaults to full time range.
        Returns:
            iterator of 2-tuples:
                context_key: The (sim_iter_num, time_ranges_key, and time_range) the
                    data was computed under.
                value: The value at that context key.
        """
        time_range = time_range if time_range else (None, None)

        if sim_iter_nums is None:
            sim_iter_nums = self.sim_iter_nums
        else:
            sim_iter_nums = sorted(set(self.sim_iter_nums) & set(sim_iter_nums))

        if time_ranges_keys is None:
            time_ranges_keys = self.time_ranges_keys
        else:
            time_ranges_keys = sorted(
                set(self.time_ranges_keys) & set(time_ranges_keys)
            )

        if use_cache:
            iterator = self.merged_iterator(
                sim_iter_nums=sim_iter_nums,
                time_ranges_keys=time_ranges_keys,
                time_range=time_range,
                version=version,
            )
        else:
            iterator = self.firestore_iterator(
                sim_iter_nums=sim_iter_nums,
                time_ranges_keys=time_ranges_keys,
                time_range=time_range,
                version=version,
            )

        for key, group in groupby(iterator, key=itemgetter(0, 1, 2, 3, 4)):
            if self.chunked:

                def value_chunk_gen(g=group):
                    for _, _, _, _, _, chunk_num, _value_chunk in g:
                        path = key_to_filename(key, chunk_num)
                        if os.path.exists(path):
                            _value_chunk = load_from_disk(path)
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )

                        yield value_chunk

                yield key, value_chunk_gen()
            else:
                # If not chunked there is only one element in the group
                _, _, _, _, _, _, _value = next(group)
                logging.warning(f"ITER_VALUE: {_value}")
                path = key_to_filename(key, 0)
                if os.path.exists(path):
                    _value = load_from_disk(path)
                    logging.warning(f"LOADED FROM DISK: {_value}")
                value, output, _ = attempt_deserialize(_value, self.value_type)
                if output.get("failed", False):
                    raise ValueError(
                        f"Deserialization failed: {output['combined_output']}"
                    )
                yield key, value

    def firestore_iterator(
        self,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        version: Optional[Text] = None,
    ):
        """
        Iterate over Firestore value_file_block docs instead of streaming from backend.
        """
        collection_ref = self._get_collection()
        time_range_start = time_range[0] if time_range else None
        time_range_end = time_range[1] if time_range else None

        query = collection_ref
        # TODO: Take out one of the 'in' statements and loop.
        if sim_iter_nums:
            query = query.where("sim_iter_num", "in", sim_iter_nums[:30])
        if time_ranges_keys:
            query = query.where("time_ranges_key", "in", time_ranges_keys[:30])
        if time_range_start:
            query = query.where("time_range_start", ">=", time_range_start)
        if time_range_end:
            query = query.where("time_range_end", "<=", time_range_end)
        if version:
            query = query.where("version", "==", version)

        query = (
            query.order_by("sim_iter_num")
            .order_by("time_range_start")
            .order_by("time_range_end")
            .order_by("time_ranges_key")
            .order_by("chunk_num")
        )
        for doc in query.stream():
            data = doc.to_dict()
            for k in ["time_range_start", "time_range_end", "version"]:
                data[k] = normalize_datetime(data[k])
            full_name = self.doc_full_name
            att = self.name
            chunk_num = data.get("chunk_num", 0)
            _value_chunk = data["_value_chunk"]

            tr = (
                data.get("time_range_start"),
                data.get("time_range_end"),
            )

            yield (
                data.get("sim_iter_num"),
                tr,
                data.get("time_ranges_key"),
                full_name,
                att,
                chunk_num,
                _value_chunk,
            )

    def merged_iterator(
        self,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        version: Optional[Text] = None,
    ):
        """
        Yield items in the order defined by full_space.
        Prefer cached data if available; otherwise pull sequentially
        from the flat iterator (which must be ordered the same way).
        Saves streamed chunks to cache as it goes.
        """
        fire_iter = None
        next_flat = None
        time_range_start = time_range[0] if time_range else None
        time_range_end = time_range[1] if time_range else None
        for key in self.full_space:
            (s_num, tr, tr_key) = key
            if sim_iter_nums is not None and s_num not in sim_iter_nums:
                continue
            if time_ranges_keys is not None and tr_key not in time_ranges_keys:
                continue

            if time_range_start is not None and time_range_start > tr[0]:
                continue
            if time_range_end is not None and time_range_end < tr[1]:
                continue

            run_key = (
                s_num,
                tr,
                tr_key,
                self.doc_full_name,
                self.name,
            )
            key_with_none = (*run_key, 0, None)
            path = key_to_filename(run_key, 0)

            chunk_num = 0
            found_cached = False
            while True:
                path = key_to_filename(run_key, chunk_num)
                if not os.path.exists(path):
                    break
                found_cached = True
                _value_chunk = load_from_disk(path)
                yield (*run_key, chunk_num, _value_chunk)
                chunk_num += 1

            if found_cached:
                continue

            if next_flat is None:
                try:
                    if fire_iter is None:
                        fire_iter = self.firestore_iterator(
                            sim_iter_nums=sim_iter_nums,
                            time_ranges_keys=time_ranges_keys,
                            time_range=time_range,
                            version=version,
                        )
                    next_flat = next(fire_iter)
                except StopIteration:
                    next_flat = None

            if next_flat is not None:

                f_key = next_flat[:-2]
                if f_key == run_key:
                    _value_chunk = next_flat[-1]
                    save_to_disk(run_key, 0, _value_chunk, MAX_CACHE_BYTES)

                    yield next_flat

                    next_flat = None
                else:
                    yield key_with_none
            else:
                yield key_with_none

    def get_first_val(
        self,
        sim_iter_num: int,
        time_ranges_key: Text,
        time_range: Optional[TimeRange] = None,
    ) -> Any:
        """
        Get the first value of returned by that query (up to that point)
        Args:
            Args:
            sim_iter_num: Which simulation to pull the value from
            time_ranges_key: Which time range collection to pull the data from
            time_range: max and min time range of the data. Defaults to full time
                series (up to this point).
        Returns:
            value: The first value retrieved by the query
        """
        iterator = self.get_iterator(
            sim_iter_nums=[sim_iter_num],
            time_ranges_keys=[time_ranges_key],
            time_range=time_range,
        )
        try:
            _, value = next(iterator)
            return value
        except StopIteration:
            return None

    @classmethod
    def get_method_documentation(cls) -> Text:
        methods = [cls.time_series, cls.sims, cls.get_first_val, cls.get_iterator]
        r_str = []
        for method in methods:
            sig = inspect.signature(method)
            header = f"def {method.__name__}{sig}:"
            doc = inspect.getdoc(method)
            r_str.append(header + "\n" + '"""' + (doc or "") + '"""\n...')
        r_str = "\n\n".join(r_str)
        return r_str

from typing import Text, Iterable, Any, Optional, Dict, List, Tuple
from utils.serialize_utils import (
    attempt_deserialize,
    attempt_serialize,
    build_category_maps,
    flatten_structure,
    unflatten_structure,
)
from utils.type_utils import TimeRange, TimeRanges
from utils.json_schema_utils import describe_json_schema
from utils.function_utils import create_function
from utils.preview_utils import value_to_preview
from utils.type_utils import get_known_types
from utils.datetime_utils import to_isos, to_datetimes, datetime_min, datetime_max
import json
import requests
import os
import datetime
import inspect
from utils.downloader import (
    save_to_memory,
    _cache,
    MAX_CACHE_BYTES,
)
from operator import itemgetter
from itertools import groupby
import logging
from utils.doc_obj import DocObj

TABDPT_URL = "http://tabdpt-service.default.svc.cluster.local:6789"


class Attribute:
    def __init__(
        self,
        name: Text,
        auth_data: Dict[Text, Any],
        doc_obj: DocObj,
        value_type: Any,
    ):
        self.doc_obj = doc_obj
        self.name = name
        self.auth_data = auth_data
        self.doc_id = self.doc_obj.doc_id
        self._val = None
        self._pred_val = None
        self.doc_full_name = self.doc_obj.full_name
        self.value_type = value_type
        self.cur_clone_num = None
        self.cur_time_range = None
        self.context_key = (None, None, None)
        self.outputs = {}
        self.cleanup = None
        self.runnable = False
        self.full_space = None
        self.run_key = None

    def _set_context(self, **kwargs):
        self.cur_clone_num = kwargs.get("clone_num", None)
        self.cur_time_range = kwargs.get("time_range", None)
        self.context_key = (self.cur_clone_num, self.cur_time_range)

        self._val = None
        self._pred_val = None

        if not self.cur_clone_num or not self.cur_time_range:
            self.run_key = None
            return

        self.run_key = (
            self.cur_clone_num,
            self.cur_time_range,
            self.doc_full_name,
            self.name,
        )

    def val(self, *args, **kwargs) -> Any:
        return self._val

    def _set_val(self, val: Any, serialized: bool = False):
        if serialized:
            val, output, cleanups = attempt_deserialize(val, self.value_type)
            self._add_output(output)
            self.cleanup = cleanups[0] if cleanups else None

        self._val = val

    def _set_full_space(
        self, full_space: List[Tuple], clone_nums: List[int], time_ranges: TimeRanges
    ):
        self.full_space = full_space
        self.full_clone_nums = clone_nums
        self.full_time_ranges = time_ranges

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
        output = self._get_output()
        data = {
            "docs_to_run": [doc_id],
            "outputs": {doc_id: {self.name: output}},
            "caller": caller,
            "auth_data": self.auth_data,
            "run_completed": False,
            "run_output": {"failed": output.get("failed", False), "message": ""},
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
        doc_obj: DocObj,
        value_type: Any,
        var_name_to_id: Dict[Text, Text],
        valid_clone_nums: List[int],
        valid_time_range: TimeRange,
        function_name: Text,
        function_header: Text,
        function_string: Text,
        predict_function_string: Text = "",
        predict_from: Optional[Text] = None,
        predict_type: Optional[Text] = None,
        new_version: Optional[Text] = None,
        old_version: Optional[Text] = None,
        chunked: bool = False,
        no_function_body: bool = False,
        locks: Optional[List[Tuple]] = None,
        global_vars: Dict[Text, Any] = None,
        deleted: bool = False,
    ):

        super().__init__(
            name=name,
            auth_data=auth_data,
            doc_obj=doc_obj,
            value_type=value_type,
        )
        self.auth_data = auth_data
        self.new_version = to_datetimes(new_version) if new_version else None
        self.old_version = to_datetimes(old_version) if old_version else None
        self._val = None
        self.chunked = chunked
        self.cur_clone_num = None
        self.cur_time_range = None
        self.outputs = {}
        self.runnable = True

        self.locks = locks or []
        locks = []
        for lock in self.locks:
            lock[1] = (
                to_isos(lock[1][0]),
                to_isos(lock[1][1]),
            )
            locks.append(lock)
        self.locks = locks

        self.no_function_body = no_function_body
        self.function_name = function_name
        self.function_header = function_header
        self.function_string = function_string
        self.predict_type = predict_type
        self.predict_function_string = predict_function_string
        self.predict_from = to_datetimes(predict_from) if predict_from else None
        self.valid_clone_nums = valid_clone_nums
        self.valid_time_range = valid_time_range
        self.var_name_to_id = var_name_to_id

        self.full_clone_nums = None
        self.full_time_ranges = None

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
        self.pred_func, output = create_function(
            function_name=self.function_name,
            function_header=self.function_header,
            function_string=self.predict_function_string,
            allowed_modules=get_known_types(),
            global_vars=global_vars if global_vars is not None else {},
        )
        self._add_output(output)

    def _deserialize(self, iterator: Iterable):
        if self.chunked:
            for key, _value in iterator:

                def value_chunk_gen(_value=_value):
                    for chunk_num, _value_chunk in enumerate(_value):
                        _value_chunk = _cache.get((key, chunk_num), _value_chunk)
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )

                        yield value_chunk

                yield key, value_chunk_gen(), {}
        else:
            for key, _value in iterator:
                _value = _cache.get((key, 0), _value)
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
        lock_value: bool = False,
    ):
        if lock_value:
            try:
                value_chunk = None
                for _, value_chunk_iter in self.get_iterator(
                    clone_nums=[self.cur_clone_num],
                    time_range=self.cur_time_range,
                    lock_value=lock_value,
                    # version=self.old_version,
                    use_cache=False,
                ):
                    if value_chunk is None:
                        value_chunk = value_chunk_iter
                        continue

            except StopIteration:
                logging.warning(
                    f"Overriden value not found: {self.run_key}, {self.old_version}"
                )
        preview = value_to_preview(value_chunk)
        _value_chunk, output = attempt_serialize(value_chunk, self.value_type)
        # _value_chunk = encode_obj(value_chunk)

        self._add_output(output)
        if output.get("failed", False):
            return
        preview = value_to_preview(value_chunk)
        schema_hash, defs = describe_json_schema(value_chunk)
        _schema = json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$ref": f"#/$defs/{schema_hash}",
                "$defs": defs,
            }
        )
        save_to_memory(run_key, chunk_num, _value_chunk)

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
            "clone_num": run_key[0],
            "time_range_start": run_key[1][0],
            "time_range_end": run_key[1][1],
            "chunk_num": chunk_num,
            "version": self.new_version,
            "_value_chunk": _value_chunk,
            "preview": preview,
            "_schema": _schema,
            "lock_value": lock_value,
        }
        version_key = (
            f"{doc_data['clone_num']}_"
            f"{to_isos(doc_data['time_range_start'])}_"
            f"{to_isos(doc_data['time_range_end'])}_"
            f"{doc_data['chunk_num']}_"
            f"{to_isos(doc_data['version'])}"
        )
        # Write directly to Firestore
        chunk_ref = self._get_collection().document(version_key)
        chunk_ref.set(doc_data)

    def _get_output(self):
        output = super()._get_output()
        output["version"] = to_isos(self.new_version) if self.new_version else None
        return output

    def _finalize(self):
        """
        Deduplicate and update value_file_blocks in Firestore:
        - Deletes duplicates, keeping only the highest version per
        (clone_num, time_range_start, time_range_end)
        - Deletes any docs not in current clone_nums
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
                data[k] = to_datetimes(data[k])
            block_key = (
                f"{data['clone_num']}_"
                f"{to_isos(data['time_range_start'])}_"
                f"{to_isos(data['time_range_end'])}_"
            )

            # Skip docs not in allowed clone_nums
            if (
                self.valid_clone_nums
                and data.get("clone_num") not in self.valid_clone_nums
            ):

                batch.delete(snap.reference)
                total_deleted += 1
                batch_count += 1
                continue

            # Skip docs not in allowed time_range
            if self.valid_time_range and (
                data.get("time_range_start") < self.valid_time_range[0]
                or data.get("time_range_end") > self.valid_time_range[1]
            ):

                batch.delete(snap.reference)
                total_deleted += 1
                batch_count += 1
                continue

            # Track highest version per block key
            current_version = to_datetimes(data.get("version"))

            if block_key in blocks_by_key:
                existing = blocks_by_key[block_key]
                existing["version"] = to_datetimes(existing["version"])

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
            old_version = to_datetimes(data.get("version"))
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

    def get_func(self):
        if not self.predict_from:
            return self.func
        if self.cur_time_range[0] >= self.predict_from:
            return self.pred_func
        return self.func

    def predict(
        self,
        train_data_domain: Dict[Text, Any],
        predict_type: Text = "regressor",
        window_size: int = 100,
    ):
        clone_num = train_data_domain.get("clone_num", None)
        time_range = train_data_domain.get("time_range", None)

        # 1. Build variable → list of time series values
        train_time_series = {
            "self": [
                v
                for _, v in self.time_series(
                    clone_num=clone_num, _time_range=time_range
                )
            ]
        }
        query_time_series = {"self": [v for _, v in self.time_series()]}

        for input_doc_id in self.var_name_to_id.values():
            full_name = self.doc_obj.doc_id_to_full_name[input_doc_id]
            doc = self.doc_obj.doc_objs[full_name]
            if not hasattr(doc, "data") or input_doc_id == self.doc_id:
                continue
            train_time_series[full_name] = [
                v
                for _, v in doc.data.time_series(
                    clone_num=clone_num, _time_range=time_range
                )
            ]

            query_time_series[full_name] = [v for _, v in doc.data.time_series()]
        if all([not v for v in train_time_series.values()]):
            raise ValueError(
                f"Cannot make prediction when {self.doc_obj.full_name}.{self.name} has no data:"
            )
        # 2. Build category maps
        cat_map, inv_cat_map = build_category_maps(
            list(train_time_series.values()) + list(query_time_series.values())
        )

        all_train_flats = {}
        all_specs = {}
        for full_name, ts in train_time_series.items():
            all_train_flats[full_name] = []
            all_specs[full_name] = []
            for val in ts:
                flat, spec = flatten_structure(val, cat_map)
                all_train_flats[full_name].append(flat)
                all_specs[full_name].append(spec)
        all_query_flats = {}
        for full_name, ts in query_time_series.items():
            all_query_flats[full_name] = []
            for val in ts:
                flat, spec = flatten_structure(val, cat_map)
                all_query_flats[full_name].append(flat)

        payload = {
            "train_data": all_train_flats,
            "query_data": all_query_flats,
            "window_size": window_size,
        }
        resp = requests.post(
            f"{TABDPT_URL}/time_series/{predict_type}",
            json=payload,
        )
        resp.raise_for_status()

        pred = resp.json()
        pred_vector = pred["predictions"][0]  # list of floats

        # use the spec corresponding to the last timestamp of "self"
        last_spec = all_specs["self"][-1]

        unflattened, _ = unflatten_structure(
            pred_vector, last_spec, idx=0, inv_map=inv_cat_map
        )

        return unflattened

    def time_series(
        self,
        clone_num: Optional[int] = None,
        mode: Text = "full",
        _time_range: Optional[TimeRange] = None,
    ) -> Iterable[Tuple[TimeRange, Any]]:
        """
        Get an iterator of the full time series of values (or section of it) of the
        attribute (up to this point in time)
        Args:
            clone_num: Which simulation to pull the time series from
                (defaults to the current simulation)

            mode: What selection of time ranges use to build the time series.
                "full" will use all of the time ranges, even when there is
                no corresponding value. "valid" only uses the time ranges
                that this attribute has set as valid.
            _time_range: (NOTE: Do not use. For internal use only)
                max and min time range of the data. Defaults to full time
                series (up to this point).
        Returns:
            iterator of 2-tuples:
                time_range: The time range at which that value was computed.
                value: The value at that time range.
        """

        if mode == "full":
            # remove before/after time ranges
            time_ranges = self.full_time_ranges[1:-1]
        elif mode == "valid":
            time_ranges = []
            # remove before/after time ranges
            for time_range in self.full_time_ranges[1:-1]:
                if (
                    time_range[0] >= self.valid_time_range[0]
                    and time_range[1] <= self.valid_time_range[1]
                ):
                    time_ranges.append(time_range)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        clone_num = clone_num if clone_num is not None else self.cur_clone_num
        if clone_num not in self.valid_clone_nums:
            raise ValueError(
                f"clone_num {clone_num} not in {self.doc_full_name}-{self.name}"
                f" list of clone_nums: {self.valid_clone_nums}"
            )

        iterator_tr = _time_range if _time_range else (datetime_min, datetime_max)

        # Only take data that has been 'completed' already
        if iterator_tr[1] > self.cur_time_range[0]:
            iterator_tr = (iterator_tr[0], self.cur_time_range[0])

        iterator = self.get_iterator(
            clone_nums=[clone_num],
            time_range=iterator_tr,
        )

        if time_ranges:
            exhausted = False
            att_time_range = None
            for time_range in time_ranges:
                while att_time_range is None or att_time_range[0] < time_range[0]:
                    try:
                        (_, att_time_range, _, _), data = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break

                if att_time_range == time_range:
                    yield (time_range, data)

                if exhausted:
                    break

        else:
            for (_, time_range, _, _), data in iterator:
                yield (time_range, data)

    def clones(
        self, time_range: Optional[TimeRange] = None
    ) -> Iterable[Tuple[int, Any]]:
        """
        Get an iterator of the data of all the simulations up until this one.
        Note that there is no restriction on selecting a time_range that includes
        several time ranges of data. May get multiple values corresponding to a single
        sim. Use time_range with caution or leave blank.
        Args:
            time_range: max and min time range of the data. Defaults to the current one.
        Returns:
            iterator of 2-tuples:
                clone_num: The simulation number that value was computed at
                value: The value for that simulation number.
        """

        # NOTE: No restriction on taking multiple time ranges
        time_range = time_range or self.cur_time_range

        # Get all previous clones
        clone_nums = list(range(self.cur_clone_num))

        iterator = self.get_iterator(
            clone_nums=clone_nums,
            time_range=time_range,
        )
        for (clone_num, _, _, _), data in iterator:
            yield (clone_num, data)

    def get_iterator(
        self,
        clone_nums: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        use_cache: bool = True,
        lock_value: Optional[bool] = None,
        version: Optional[Text] = None,
    ) -> Iterable[Tuple[Tuple[int, Text, TimeRange], Any]]:
        """
        Get an iterator of some slice of the data computed up until this point.
        Args:
            clone_nums: Which simulations to pull the data from
                (defaults to all simulations)
            time_range: max and min time range of the data. Defaults to full time range.
        Returns:
            iterator of 2-tuples:
                context_key: The (clone_num, and time_range) the
                    data was computed under.
                value: The value at that context key.
        """
        time_range = time_range if time_range else self.valid_time_range
        time_range = to_datetimes(time_range)
        if clone_nums is None:
            clone_nums = self.valid_clone_nums

        else:
            clone_nums = sorted(set(self.valid_clone_nums) & set(clone_nums))

        if use_cache:
            iterator = self.merged_iterator(
                clone_nums=clone_nums,
                time_range=time_range,
                lock_value=lock_value,
                version=version,
            )
        else:
            iterator = self.firestore_iterator(
                clone_nums=clone_nums,
                time_range=time_range,
                lock_value=lock_value,
                version=version,
            )

        for key, group in groupby(iterator, key=itemgetter(0, 1, 2, 3)):
            if self.chunked:

                def value_chunk_gen(g=group):
                    for _, _, _, _, chunk_num, _value_chunk in g:
                        _value_chunk = _cache.get((key, chunk_num), _value_chunk)
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )

                        yield value_chunk

                yield key, value_chunk_gen()
            else:
                # If not chunked there is only one element in the group
                _, _, _, _, _, _value = next(group)
                _value = _cache.get((key, 0), _value)
                value, output, _ = attempt_deserialize(_value, self.value_type)
                if output.get("failed", False):
                    raise ValueError(
                        f"Deserialization failed: {output['combined_output']}"
                    )
                yield key, value

    def firestore_iterator(
        self,
        clone_nums: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        lock_value: Optional[bool] = None,
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
        if clone_nums:
            query = query.where("clone_num", "in", clone_nums[:30])

        if time_range_start:
            query = query.where("time_range_start", ">=", time_range_start)
        if time_range_end:
            query = query.where("time_range_end", "<=", time_range_end)
        if version:
            query = query.where("version", "==", version)
        if lock_value is not None:
            query = query.where("lock_value", "==", lock_value)

        query = (
            query.order_by("clone_num")
            .order_by("time_range_start")
            .order_by("time_range_end")
            .order_by("chunk_num")
        )
        for doc in query.stream():
            data = doc.to_dict()
            for k in ["time_range_start", "time_range_end", "version"]:
                data[k] = to_datetimes(data[k])
            full_name = self.doc_full_name
            att = self.name
            chunk_num = data.get("chunk_num", 0)
            _value_chunk = data["_value_chunk"]

            tr = to_datetimes(
                (
                    data.get("time_range_start"),
                    data.get("time_range_end"),
                )
            )

            yield (
                data.get("clone_num"),
                tr,
                full_name,
                att,
                chunk_num,
                _value_chunk,
            )

    def merged_iterator(
        self,
        clone_nums: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
        lock_value: Optional[bool] = None,
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
            (s_num, tr) = key
            if clone_nums is not None and s_num not in clone_nums:
                continue
            if time_range_start is not None and time_range_start > tr[0]:
                continue
            if time_range_end is not None and time_range_end < tr[1]:
                continue

            run_key = (
                s_num,
                tr,
                self.doc_full_name,
                self.name,
            )
            key_with_none = (*run_key, 0, None)

            chunk_num = 0
            found_cached = False
            while True:
                if (run_key, chunk_num) not in _cache:
                    break
                found_cached = True
                _value_chunk = _cache.get((run_key, chunk_num))
                yield (*run_key, chunk_num, _value_chunk)
                chunk_num += 1

            if found_cached:
                continue

            if next_flat is None:
                try:
                    if fire_iter is None:

                        fire_iter = self.firestore_iterator(
                            clone_nums=clone_nums,
                            time_range=time_range,
                            lock_value=lock_value,
                            version=version,
                        )
                    next_flat = next(fire_iter)
                except StopIteration:
                    next_flat = None

            if next_flat is not None:

                f_key = next_flat[:-2]
                if f_key == run_key:
                    _value_chunk = next_flat[-1]
                    save_to_memory(run_key, 0, _value_chunk)

                    yield next_flat

                    next_flat = None
                else:
                    yield key_with_none
            else:
                yield key_with_none

    def val(
        self,
        clone_num: Optional[int] = None,
        _time_range: Optional[TimeRange] = None,
    ) -> Any:
        """
        Get the most recently computed value (default) or get the first value
        returned by that query corresponding to kwargs (up to that point)
        Args:
            Args:
            clone_num: Which simulation to pull the value from
            time_range: (NOTE: Do not use. For internal use only)
            max and min time range of the data. Defaults to full time
                series (up to this point).
        Returns:
            value: The first value retrieved by the query
        """
        if clone_num is None and _time_range is None and self._val is not None:
            return self._val

        iterator = self.get_iterator(
            clone_nums=[clone_num] if clone_num is not None else None,
            time_range=_time_range,
        )
        try:
            _, value = next(iterator)
            return value
        except StopIteration:
            return None

    @classmethod
    def get_method_documentation(cls) -> Text:
        methods = [cls.time_series, cls.clones, cls.val, cls.get_iterator]
        r_str = []
        for method in methods:
            sig = inspect.signature(method)
            header = f"def {method.__name__}{sig}:"
            doc = inspect.getdoc(method)
            r_str.append(header + "\n" + '"""' + (doc or "") + '"""\n...')
        r_str = "\n\n".join(r_str)
        return r_str

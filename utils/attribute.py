from typing import Text, Iterable, Any, Optional, Dict, List, Tuple
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.type_utils import TimeRange, describe_json_schema
from utils.function_utils import create_function
from utils.downloader import BatchDownloader
from utils.uploader import BatchUploader
from utils.preview_utils import value_to_preview
from utils.type_utils import get_known_types
import logging
import json
import requests
import os
import datetime
import inspect


class Attribute:
    def __init__(
        self,
        name: Text,
        auth_data: Dict[Text, Any],
        doc_id: Text,
        value_type: Any,
    ):
        self.name = name
        self.auth_data = auth_data
        self.doc_id = doc_id
        self._val = None
        self.value_type = value_type
        self.sim_iter_num = None
        self.time_ranges_key = None
        self.time_range = None
        self.context_key = (None, None, None)
        self.outputs = {}
        self.cleanup = None
        self.runnable = False

    def _set_context(self, **kwargs):
        self.sim_iter_num = kwargs.get("sim_iter_num", None)
        self.time_ranges_key = kwargs.get("time_ranges_key", None)
        self.time_range = kwargs.get("time_range", None)
        self.context_key = (self.sim_iter_num, self.time_ranges_key, self.time_range)

    @property
    def val(self) -> Any:
        return self._val

    def _set_val(self, val: Any, serialized: bool = False):
        if serialized:
            val, output, cleanups = attempt_deserialize(val, self.value_type)
            self._add_output(output)
            self.cleanup = cleanups[0] if cleanups else None

        self._val = val

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
                if _output["failed"]:
                    print(
                        "fail",
                        self.name,
                        context_key,
                        _output["combined_output"].strip(),
                    )
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
        output["new_value_file_ref"] = getattr(self, "value_file_ref", None)
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
        auth_data: Dict[Text, Text],
        doc_id: Text,
        value_type: Any,
        value_file_ref: Text,
        var_name_to_id: Dict[Text, Text],
        sim_iter_nums: List[Text],
        time_ranges_keys: List[Text],
        function_name: Text,
        function_header: Text,
        function_string: Text,
        chunked: bool = False,
        old_value_file_ref: Optional[Text] = None,
        no_function_body: bool = False,
        overrides: Optional[List[Tuple]] = None,
        global_vars: Dict[Text, Any] = None,
    ):

        super().__init__(
            name=name,
            auth_data=auth_data,
            doc_id=doc_id,
            value_type=value_type,
        )
        self.auth_data = auth_data
        self.value_file_ref = value_file_ref
        self._val = None
        self.chunked = chunked
        self.sim_iter_num = None
        self.time_ranges_key = None
        self.time_range = None
        self.outputs = {}
        self.runnable = True
        self.overrides = overrides or []
        self.no_function_body = no_function_body
        self.function_name = function_name
        self.function_header = function_header
        self.function_string = function_string
        self.sim_iter_nums = sim_iter_nums
        self.time_ranges_keys = time_ranges_keys
        self.var_name_to_id = var_name_to_id
        self.uploader = BatchUploader(
            auth_data=auth_data,
            value_file_ref=value_file_ref,
            old_value_file_ref=old_value_file_ref,
        )
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
                    for _value_chunk in _value:
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )
                        self._add_output(output)
                        yield value_chunk

                yield key, value_chunk_gen()
        else:
            for key, _value in iterator:
                value, output, _ = attempt_deserialize(_value, self.value_type)
                if output:
                    print(key, output)
                self._add_output(output)
                yield key, value

    def _upload_chunk(self, value_chunk, chunk_num: int = 0, overriden: bool = False):
        if not overriden:
            preview = value_to_preview(value_chunk)
            _schema = json.dumps(describe_json_schema(value_chunk))
            _value_chunk, output = attempt_serialize(value_chunk, self.value_type)
            self._add_output(output)
        else:
            preview = ""
            _schema = ""
            _value_chunk = ""

        success, message = self.uploader.add_chunk(
            sim_iter_num=self.sim_iter_num,
            time_ranges_key=self.time_ranges_key,
            time_range=self.time_range,
            _value_chunk=_value_chunk,
            chunk_num=chunk_num,
            preview=preview,
            _schema=_schema,
            overriden=overriden,
        )
        output = {
            "failed": not success,
            "combined_output": message,
            "stderr_output": message,
            "stdout_output": "",
        }
        self._add_output(output)

    def _flush(self):
        self.uploader.flush_batch()

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
        time_ranges_key = time_ranges_key or self.time_ranges_key
        time_range = time_range or (datetime.datetime.min, datetime.datetime.max)

        # Only take data that has been 'completed' already
        if time_range[1] > self.time_range[0]:
            time_range = (time_range[0], self.time_range[0])
        iterator = self.get_iterator(
            sim_iter_nums=[sim_iter_num],
            time_ranges_keys=[time_ranges_key],
            time_range=time_range,
        )
        for (sim_iter_num, time_ranges_key, time_range), data in iterator:
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

        # NOTE: No restriction on taking multiple time ranges
        time_range = time_range or self.time_range

        # Get all previous sims
        sim_iter_nums = list(range(self.sim_iter_num))

        iterator = self.get_iterator(
            sim_iter_nums=sim_iter_nums,
            time_ranges_keys=[time_ranges_key],
            time_range=time_range,
        )
        for (sim_iter_num, time_ranges_key, time_range), data in iterator:
            yield (sim_iter_num, data)

    def get_iterator(
        self,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
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

        downloader = BatchDownloader(
            auth_data=self.auth_data,
            doc_id=self.doc_id,
            attribute_name=self.name,
            sim_iter_nums=sim_iter_nums,
            value_file_ref=self.value_file_ref,
            time_ranges_keys=time_ranges_keys,
            time_range_start=time_range[0],
            time_range_end=time_range[1],
            chunked=self.chunked,
        )

        return self._deserialize(iterator=downloader)

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
            key, value = next(iterator)
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

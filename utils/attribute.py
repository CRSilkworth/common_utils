from typing import Text, Iterable, Any, Optional, Dict, List, Tuple
from utils.serialize_utils import attempt_deserialize
from utils.type_utils import TimeRange
from utils.downloader import BatchDownloader
import logging


class Attribute:
    def __init__(
        self,
        name: Text,
        auth_data: Dict[Text, Text],
        doc_id: Text,
        value_type: Any,
        value_file_ref: Text,
        chunked: bool = False,
    ):
        self.name = name
        self.auth_data = auth_data
        self.doc_id = doc_id
        self.value_file_ref = value_file_ref
        self._val = None
        self.value_type = value_type
        self.chunked = chunked
        self.sim_iter_num = None
        self.time_ranges_key = None
        self.time_range = None

    def set_context(self, **kwargs):
        self.sim_iter_num = kwargs.get("sim_iter_num", None)
        self.time_ranges_key = kwargs.get("time_ranges_key", None)
        self.time_range = kwargs.get("time_range", None)

    def deserialize(self, iterator: Iterable):
        if self.chunked:
            for key, _value in iterator:

                def value_chunk_gen(_value=_value):
                    for _value_chunk in _value:
                        value_chunk, _, _ = attempt_deserialize(
                            _value_chunk, self.value_type
                        )
                        if output:
                            logging.warning(output["combined_output"])
                        yield value_chunk

                yield key, value_chunk_gen()
        else:
            for key, _value in iterator:
                value, output, _ = attempt_deserialize(_value, self.value_type)
                if output:
                    logging.warning(output["combined_output"])
                yield key, value

    @property
    def val(self) -> Any:
        return self._val

    def _set_val(self, val: Any):
        self._val = val

    def time_series(
        self,
        sim_iter_num: Optional[int] = None,
        time_ranges_key: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ) -> List[Tuple[TimeRange, Any]]:
        sim_iter_num = sim_iter_num or self.sim_iter_num
        time_ranges_key = time_ranges_key or self.time_ranges_key
        time_range = time_range or self.time_range

        # Only take data that has been 'completed' already
        if time_range[1] >= self.time_range[0]:
            time_range[1] = self.time_range[0]

        return self.get_iterator(
            sim_iter_num=sim_iter_num,
            time_ranges_key=time_ranges_key,
            time_range=time_range,
        )

    def sims(
        self,
        time_ranges_key: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ) -> List[Tuple[TimeRange, Any]]:
        sim_iter_num = sim_iter_num or self.sim_iter_num
        time_ranges_key = time_ranges_key or self.time_ranges_key
        time_range = time_range or self.time_range

        # Only take data that has been 'completed' already
        sim_iter_nums = list(range(self.sim_iter_num))

        return self.get_iterator(
            sim_iter_num=sim_iter_num,
            time_ranges_key=time_ranges_key,
            time_range=time_range,
        )

    def get_iterator(
        self,
        sim_iter_nums: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ):
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

        return self.deserialize(
            iterator=downloader, value_type=self.value_type, chunked=self.chunked
        )

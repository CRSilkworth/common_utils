from typing import Dict, Text, Any, Set, Optional, Tuple, Iterable

from utils.datetime_utils import to_datetimes
from utils.serialize_utils import serialize_value
from utils.type_utils import deserialize_typehint, TimeRange
import copy


class DocObj(dict):
    def __init__(
        self,
        doc_id: Text,
        full_name: Text,
        doc_dict: Dict[Text, Any],
        auth_data: Dict[Text, Any],
        fs_db: Any,
        global_vars: Optional[Dict[Text, Any]] = None,
    ):
        from utils.attribute import RunnableAttribute, Attribute

        super().__init__()
        self.doc_id = doc_id
        self.full_name = full_name
        self.auth_data = auth_data
        self.fs_db = fs_db
        self.cleanups = {}
        self.outputs = {}
        self.uploaders = {}
        self.doc_objs = None
        self.doc_id_to_full_name = None
        self.attributes: Dict[Text, Attribute] = {}
        self.doc_dict = copy.deepcopy(doc_dict)
        for att, att_dict in self.doc_dict.items():
            if att == "full_name":
                continue
            if isinstance(att_dict, str):
                value_type = str
                _value = serialize_value(att_dict)
                runnable = False
            elif isinstance(att_dict, dict):

                value_type = deserialize_typehint(att_dict["_value_type"])
                att_dict["value_type"] = value_type
                _value = att_dict.get("_value", None)
                runnable = att_dict.get("runnable", False)

            if runnable:

                valid_time_range = (
                    to_datetimes(att_dict["time_range"][0]),
                    to_datetimes(att_dict["time_range"][1]),
                )

                self.attributes[att] = RunnableAttribute(
                    name=att,
                    auth_data=auth_data,
                    doc_obj=self,
                    fs_db=self.fs_db,
                    value_type=value_type,
                    new_version=att_dict.get("new_version"),
                    old_version=att_dict.get("old_version"),
                    chunked=att_dict["chunked"],
                    var_name_to_id=att_dict.get("var_name_to_id"),
                    function_name=att_dict.get("function_name"),
                    function_header=att_dict.get("function_header"),
                    function_string=att_dict.get("function_string"),
                    no_function_body=att_dict.get("empty", False),
                    valid_clone_nums=att_dict["clone_nums"],
                    valid_time_range=valid_time_range,
                    predict_from=att_dict.get("predict_from"),
                    predict_function_string=att_dict.get("predict_function_string", ""),
                    predict_type=att_dict.get("predict_type"),
                    locks=att_dict["locks"],
                    global_vars=global_vars,
                )
            else:
                # value_type in (QuickBooks, DBConnection, Files, str)
                self.attributes[att] = Attribute(
                    name=att,
                    auth_data=auth_data,
                    doc_obj=self,
                    value_type=value_type,
                )
                self.attributes[att]._set_val(val=_value, serialized=True)

    def failures(self) -> Set[Text]:
        failures = set()
        for att in self.attributes:
            if self.attributes[att]._get_output().get("failed", False):
                failures.add(att)
        return failures

    @property
    def att_dicts(self):
        return {k: v for k, v in self.doc_dict.items() if isinstance(v, dict)}

    def __getattr__(self, att: Text):
        if att in self.attributes:
            return self.attributes[att]

        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute {att!r}"
        )

    def val(
        self,
        clone_num: Optional[int] = None,
        _time_range: Optional[TimeRange] = None,
    ) -> Any:
        """
        Short cut to the data attribute's val method

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
        return self.data.val(clone_num=clone_num, _time_range=_time_range)

    def time_series(
        self,
        clone_num: Optional[int] = None,
        _time_range: Optional[TimeRange] = None,
    ) -> Iterable[Tuple[TimeRange, Any]]:
        """
        Short cut to the data attribute's time_series method

        Get an iterator of the full time series of values (or section of it) of the
        attribute (up to this point in time)
        Args:
            clone_num: Which simulation to pull the time series from
                (defaults to the current simulation)
            _time_range: (NOTE: Do not use. internal use only). specify a slice
            of the time_series. Defaults to full.
        Returns:
            iterator of 2-tuples:
                time_range: The time range at which that value was computed.
                value: The value at that time range.
        """
        return self.data.time_series(clone_num=clone_num, _time_range=_time_range)

    def clones(
        self, time_range: Optional[TimeRange] = None
    ) -> Iterable[Tuple[int, Any]]:
        """
        Short cut to the data attribute's clones method

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
        return self.data.clones(time_range=time_range)

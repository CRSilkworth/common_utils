from typing import Dict, Text, Any, Set, Optional
from utils.serialize_utils import serialize_value
from utils.type_utils import deserialize_typehint
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
                    sim_iter_nums=att_dict["sim_iter_nums"],
                    time_ranges_keys=att_dict["time_ranges_keys"],
                    overrides=att_dict["overrides"],
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

from typing import Dict, Text, Any, Optional, Iterable, Set
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.type_utils import deserialize_typehint, ModelDict, TimeRange
from utils.downloader import BatchDownloader
from utils.uploader import BatchUploader
from pymongo.mongo_client import MongoClient as PyMongoClient
from psycopg2.extensions import connection as Psycopg2Connection
from google.cloud.bigquery import Client as BigQueryClient
from quickbooks import QuickBooks
import copy
import requests
import os


class DocObj(dict):
    def __init__(
        self,
        doc_dict: Dict[Text, Any],
        auth_data: Dict[Text, Any],
    ):
        super().__init__()
        self.auth_data = auth_data
        self.cleanups = {}
        self.outputs = {}
        self.uploaders = {}
        self.doc_dict = copy.deepcopy(doc_dict)
        for att, att_dict in doc_dict.items():
            if isinstance(att_dict, str):
                self[att] = att_dict
            if isinstance(att_dict, dict):
                self.uploaders[att] = BatchUploader(
                    auth_data=auth_data,
                    value_file_ref=att_dict["new_value_file_ref"],
                )
                att_dict["local_type"] = deserialize_typehint(att_dict["_local_type"])
                local_rep, output, _ = attempt_deserialize(
                    att_dict["_local_rep"], att_dict["local_type"]
                )
                self.add_output(att, output)
                if output.get("failed", False):
                    continue

                att_dict["local_rep"] = local_rep

                att_dict["value_type"] = deserialize_typehint(att_dict["_value_type"])
                if att_dict["value_type"] is (QuickBooks):
                    value, _, __ = attempt_deserialize(None, att_dict["value_type"])
                    value.auth_data = auth_data
                    att_dict["value"] = value

                elif att_dict["value_type"] in (
                    ModelDict,
                    PyMongoClient,
                    Psycopg2Connection,
                    BigQueryClient,
                ):
                    value, output, cleanups = attempt_deserialize(
                        att_dict["_value"], att_dict["value_type"]
                    )
                    self.outputs[att] = output
                    if output.get("failed", False):
                        self.failures.add(att)
                        continue

                    if cleanups:
                        self.cleanups[att] = cleanups[0]
                    att_dict["value"] = value

    def add_output(self, att: Text, output: Dict[Text, Any]):
        if not output:
            return
        self.outputs.setdefault(att, [])
        self.outputs[att].append(output)

    def get_output(self, att: Text, **context) -> Dict[Text, Any]:
        combined = {
            "failed": False,
            "combined_output": "",
            "stdout_output": "",
            "stderr_output": "",
        }
        for output in self.outputs[att]:
            combined = {
                "failed": combined["failed"] or output["failed"],
                "combined_output": combined["combined_output"]
                + output["combined_output"],
                "stderr_output": combined["stderr_output"] + output["stderr_output"],
                "stdout_output": combined["stdout_output"] + output["stdout_output"],
            }
        if context:
            context = "\n".join([f"{k}={v}" for k, v in context.items()])
            for key in combined:
                if key == "failed":
                    continue
                combined[key] = f"When running with {context}"
        return combined

    def send_output(
        self,
        att: Text,
        caller: Optional[Text] = None,
        context: Optional[Dict[Text, Any]] = None,
    ):
        doc_id = self.doc_dict["doc_id"]
        context = context if context else {}

        # Send the attribute result back to the backend
        data = {
            "docs_to_run": [doc_id],
            "outputs": {doc_id: {att: self.get_output(att, **context)}},
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

    def failures(self) -> Set[bool]:
        failures = set()
        for att in self.att_dicts:
            if self.get_output(att).get("failed"):
                failures.add(att)
        return failures

    @property
    def att_dicts(self):
        return {k: v for k, v in self.doc_dict.items() if isinstance(v, dict)}

    def time_series(
        self,
        attribute_name: Text,
        time_ranges_key: Optional[Text] = "__WHOLE__",
        sim_param_key: Optional[Text] = "__TRUE__",
    ):
        iterator = self.get_iterator(
            attribute_name,
            sim_param_keys=[sim_param_key],
            time_ranges_keyss=[time_ranges_key],
        )
        for (_, __, time_range), value in iterator:
            yield (time_range, value)

    def sims(
        self,
        attribute_name: Text,
        time_ranges_key: Optional[Text] = "__WHOLE__",
        time_range_start: Optional[Text] = None,
        time_range_end: Optional[Text] = None,
    ):
        iterator = self.get_iterator(
            attribute_name,
            time_ranges_keyss=[time_ranges_key],
            time_range_start=time_range_start,
            time_range_end=time_range_end,
        )
        for (sim_param_key, __, time_range), value in iterator:
            yield (sim_param_key, time_range, value)

    def get_iterator(
        self,
        att: Text,
        sim_param_keys: Optional[Text] = None,
        time_ranges_keys: Optional[Text] = None,
        time_range: Optional[TimeRange] = None,
    ):
        time_range = time_range if time_range else (None, None)
        if att not in self.doc_dict:
            raise AttributeError(
                f"DocObj has no attribute '{att}'. Available attributes are "
                f"{sorted(self.doc_dict)}"
            )
        if isinstance(self.doc_dict[att], dict):
            if "value" in self.doc_dict[att]:
                return self.doc_dict[att]["value"]
            att_dict = self.att_dicts[att]
            downloader = BatchDownloader(
                auth_data=self.auth_data,
                sim_param_keys=sim_param_keys,
                value_file_ref=self.doc_dict[att]["value_file_ref"],
                time_ranges_keys=time_ranges_keys,
                time_range_start=time_range[0],
                time_range_end=time_range[1],
                chunked=att_dict["chunked"],
            )

            return self.deserialize(
                att=att,
                iterator=downloader,
                value_type=att_dict["value_type"],
                chunked=att_dict["chunked"],
            )

    def finalize_value_update(self, att: Text):
        self.uploaders[att].flush_batch()

        self.doc_dict[att]["value_file_ref"] = self.doc_dict[att]["new_value_file_ref"]

        # Just in case someone tries to upload it to the . have it throw an error
        del self.doc_dict[att]["new_value_file_ref"]

    def upload_chunk(
        self,
        att: Text,
        sim_param_key: Text,
        time_ranges_key: Text,
        time_range: TimeRange,
        chunk_num: int,
        value_chunk: Any,
    ):
        uploader = self.uploaders[att]
        _value_chunk, output = attempt_serialize(
            value_chunk, self.att_dicts[att]["value_type"]
        )
        self.add_output(att, output)

        uploader.add_chunk(
            sim_param_key=sim_param_key,
            time_ranges_key=time_ranges_key,
            time_range=time_range,
            chunk_num=chunk_num,
            value_chunk=_value_chunk,
        )

    def deserialize(
        self, att: Text, iterator: Iterable, value_type: Any, chunked=False
    ):
        for key, _value in iterator:
            if chunked:

                def value_chunk_gen(_value=_value):
                    for _value_chunk in _value:
                        value_chunk, output, _ = attempt_deserialize(
                            _value_chunk, value_type
                        )

                        if output:
                            self.add_output(att, output)
                            break
                        yield value_chunk

                yield key, value_chunk_gen()
            else:
                value, output, _ = attempt_deserialize(_value, value_type)

                if output:
                    self.add_output(att, output)
                    break
                yield key, value

    def __getattr__(self, att: Text):

        if att not in self.doc_dict:
            raise AttributeError(
                f"DocObj has no attribute '{att}'. Available attributes are "
                f"{sorted(self.doc_dict)}"
            )

        return self[att]

    def __setattr__(self, att, value):
        if att not in ("doc_dict", "auth_data"):
            raise AttributeError(f"Tried to set {att}, but attributes are read only")
        self[att] = value

    def __delattr__(self, att):
        try:
            del self[att]
        except KeyError:
            raise AttributeError(f"'{self.var_name}' has no attribute '{att}'")

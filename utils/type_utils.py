from typing import (
    Text,
    TypeVar,
    Dict,
    Union,
    List,
    Tuple,
    Optional,
    TypedDict,
    get_args,
    get_origin,
    ForwardRef,
    Any,
    Iterable,
    Hashable,
)
from collections.abc import Iterable as abc_Iterable
import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence
import builtins
import typing
import typing_extensions
import sqlite3
import plotly.graph_objects as go
import torch
from quickbooks import QuickBooks
import re
import datetime
from pymongo.mongo_client import MongoClient as PyMongoClient
from psycopg2.extensions import connection as Psycopg2Connection
from google.cloud.bigquery import Client as BigQueryClient
from utils.json_schema_utils import describe_json_schema
import pymongo
import json
import logging

TextOrint = TypeVar("TextOrint", Text, int)


class PositionDict(TypedDict):
    x: int
    y: int


Position = Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], PositionDict]


class File(TypedDict):
    file_content: Text
    file_name: Text
    file_date: float


Files = List[File]


ElementType = Dict[str, Union[int, str, Dict[str, TextOrint]]]
ElementDataType = Dict[Text, Optional[TextOrint]]


Allowed = Union[
    Dict[typing.Hashable, "Allowed"],
    List["Allowed"],
    Tuple["Allowed", ...],
    int,
    float,
    bytes,
    str,
    np.ndarray,
    np.generic,
    pd.DataFrame,
    pd.Period,
    pd.Series,
    datetime.datetime,
    type(None),
]


class ExportSpec(TypedDict):
    value_type: Any
    start_cell: Text


ExportConfig = Dict[Union[Text, int], ExportSpec]
ExportablePrimitive = Union[int, float, bytes, str, type(None)]

Exportable = Union[
    ExportablePrimitive,
    List[ExportablePrimitive],
    np.ndarray,
    np.generic,
    pd.DataFrame,
    pd.Period,
    pd.Series,
]
DBConnection = Union[PyMongoClient, Psycopg2Connection, BigQueryClient]
ModelDict = Dict[Text, Union[torch.nn.Module, Text, Dict[Text, Any]]]
TimeRange = Union[Tuple[datetime.datetime, datetime.datetime]]
TimeRanges = typing.Iterable[TimeRange]
AllTimeRanges = typing.Dict[Text, TimeRanges]
SimParams = typing.Dict[Text, typing.Hashable]
AllSimParams = List[SimParams]


class GCSPath(str):
    def __new__(cls, value: str):
        if not value.startswith("gs://"):
            raise ValueError("GCSPath must start with 'gs://'")
        return super().__new__(cls, value)

    @property
    def bucket(self) -> str:
        return self.split("/", 3)[2]

    @property
    def path(self) -> str:
        return self.split("/", 3)[3] if "/" in self[5:] else ""


def describe_allowed(obj, defs=None):
    schema_hash, defs = describe_json_schema(obj, defs=defs)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$ref": f"#/$defs/{schema_hash}",
        "$defs": defs,
    }


def is_allowed_type(obj) -> bool:
    if isinstance(
        obj,
        (
            int,
            float,
            bytes,
            str,
            np.ndarray,
            np.generic,
            pd.DataFrame,
            pd.Series,
            pd.Period,
            datetime.datetime,
            type(None),
        ),
    ):
        return True, ""

    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            if not is_allowed_type(k):
                return False, f"key '{k}' ({type(k)}) is not of allowed type"
            if not is_allowed_type(v):
                return False, f"value '{v}' ({type(v)}) is not of allowed type"

        return True, ""

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for k in obj:
            if not is_allowed_type(k):
                return False, f"item '{k}' ({type(k)}) is not of allowed type"

        return True, ""

    return False, f"{obj} of type: {type(obj)} is not of allowed type"


def serialize_typehint(t: type) -> str:
    """Serialize a type hint to a string (prefer registered name over full path)."""
    if isinstance(t, str):
        return t

    if isinstance(t, TypeVar):
        return f"TypeVar({t.__name__})"

    known_types = get_known_types()

    for k, v in known_types.items():
        if v == t:
            return k

    if t in {Hashable, Iterable, Tuple}:
        return f"{t.__module__}.{t.__qualname__}"

    origin = get_origin(t)
    args = get_args(t)

    if origin:
        origin_str = serialize_typehint(origin)
        args_str = ", ".join(serialize_typehint(arg) for arg in args)
        return f"{origin_str}[{args_str}]"

    if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
        if t.__module__ == "builtins":
            return t.__qualname__
        return f"{t.__module__}.{t.__qualname__}"

    return str(t)


def get_known_types(custom_types: Optional[Dict[Text, Any]] = None):
    known_types = {
        # Built-ins
        **vars(builtins),
        # Typing
        **vars(typing),
        **vars(typing_extensions),
        "typing": typing,
        "typing_extensions": typing_extensions,
        # Numpy/Pandas
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        # SQL/DB
        "sqlite3": sqlite3,
        "sqlite3.Connection": sqlite3.Connection,
        "go": go,
        "Figure": go.Figure,
        "plotly": __import__("plotly"),
        "plotly.graph_objs": go,
        "plotly.graph_objs._figure": __import__(
            "plotly.graph_objs._figure", fromlist=["Figure"]
        ),
        "plotly.graph_objs._figure.Figure": go.Figure,
        "utils": __import__("utils"),
        "utils.type_utils": __import__("utils.type_utils"),
        "NoneType": type(None),
        "_NoneType": type(None),
        "utils.type_utils.PositionDict": PositionDict,
        "utils.type_utils.Position": Position,
        "utils.type_utils.DBConnection": DBConnection,
        "utils.type_utils.File": File,
        "utils.type_utils.Files": Files,
        "utils.type_utils.ModelDict": ModelDict,
        "utils.type_utils.AllSimParams": AllSimParams,
        "utils.type_utils.SimParams": SimParams,
        "utils.type_utils.TimeRange": TimeRange,
        "utils.type_utils.TimeRanges": TimeRanges,
        "utils.type_utils.AllTimeRanges": AllTimeRanges,
        "utils.type_utils.Allowed": Allowed,
        "utils.type_utils.GCSPath": GCSPath,
        "typing.Hashable": typing.Hashable,
        "typing.Iterable": typing.Iterable,
        "typing.Tuple": typing.Tuple,
    }

    known_types.update(
        {
            "pymongo": __import__("pymongo"),
            "pymongo.mongo_client": pymongo.mongo_client,
            "pymongo.synchronous.mongo_client": pymongo.mongo_client,
            "pymongo.mongo_client.MongoClient": PyMongoClient,
            "pymongo.synchronous.mongo_client.MongoClient": PyMongoClient,
            "psycopg2": __import__("psycopg2"),
            "psycopg2.extensions.connection": Psycopg2Connection,
            "google": __import__("google"),
            "google.cloud": __import__("google.cloud", fromlist=["bigquery"]),
            "google.cloud.bigquery": __import__(
                "google.cloud.bigquery", fromlist=["client"]
            ),
            "google.cloud.bigquery.client.Client": BigQueryClient,
            "quickbooks": __import__("quickbooks"),
            "quickbooks.QuickBooks": QuickBooks,
            "quickbooks.client.QuickBooks": QuickBooks,
            "torch": torch,
            "torch.nn": torch.nn,
            "nn": torch.nn,
            "torch.nn.Module": torch.nn.Module,
            "torch.nn.modules.module.Module": torch.nn.Module,
        }
    )

    if custom_types:
        known_types.update(custom_types)
    return known_types


def deserialize_typehint(s: str, custom_types: dict = None) -> type:
    """Deserialize a string back into a type hint."""

    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s).__name__}: {s}")

    known_types = get_known_types(custom_types=custom_types)

    if s.startswith("TypeVar(") and s.endswith(")"):
        name = s[len("TypeVar(") : -1]  # noqa: E203

        if name in known_types:
            return known_types[name]
        else:
            raise ValueError(f"Unknown TypeVar {name}")

    try:
        return eval(s, known_types)
    except Exception as e:
        logging.warning(e)
        raise ValueError(f"Cannot deserialize type hint from: {s}") from e


def _resolve_forwardrefs(tp, globalns):
    """Recursively resolve ForwardRefs in a type."""
    if isinstance(tp, ForwardRef):
        ref = tp.__forward_arg__
        if ref in globalns:
            return globalns[ref]
        else:
            raise ValueError(f"Unresolvable ForwardRef: {ref}")

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:
        return tp

    resolved_args = tuple(_resolve_forwardrefs(arg, globalns) for arg in args)
    return origin[resolved_args]


def is_valid_output(value, output_type):
    if output_type == sqlite3.Connection:
        return isinstance(value, sqlite3.Connection), ""
    if output_type == SimParams:
        try:
            if not isinstance(value, dict):
                return False, f"{value} is not an instance of dict"
            for key, v in value.items():
                if not isinstance(key, str):
                    return False, f"{key} is not an instance of str"
                hash(v)
        except TypeError:
            return False, f"{v} is not hashable"

        return True, ""
    if output_type == ExportConfig:
        if not isinstance(value, dict):
            return False, f"{value} is not an instance of dict"
        for key, v in value.items():
            if not isinstance(key, str) and not isinstance(key, int):
                return False, f"{key} is not an instance of str or int"
            is_valid, message = is_valid_output(v, ExportSpec)
            if not is_valid:
                return False, message

        return True, ""
    if output_type == ExportSpec:

        if not isinstance(value, dict):
            return False, f"{value} is not an instance of dict"
        if set(value.keys()) != set("value_type", "start_cell"):
            return (
                False,
                f"{value} must contain keys: 'value_type', 'start_cell' and no others,"
                f" Got {list(value.keys())}",
            )

        is_cell_code = bool(re.compile(r"^[A-Z]+[0-9]+$").match(value["start_cell"]))
        if not is_cell_code:
            return False, f"{value['start_cell']} is not proper cell name"

        return True, ""
    if output_type == AllSimParams:
        if not isinstance(value, list):
            return False, f"{value} is not an instance of list"
        for sim_params in value:
            is_valid, message = is_valid_output(sim_params, output_type=SimParams)
            if not is_valid:
                return False, message

        return True, ""
    if output_type == TimeRanges:
        if not isinstance(value, abc_Iterable):
            return False, f"{value} is not an iterable of time ranges"
        for time_range in value:
            is_valid, message = is_valid_output(time_range, output_type=TimeRange)
            if not is_valid:
                return False, message
    if output_type == TimeRange:
        if not isinstance(value, tuple) or not len(value) == 2:
            return False, f"{value} is not a tuple of len 2"
        if not isinstance(value[0], datetime.datetime) or not isinstance(
            value[1], datetime.datetime
        ):
            return False, f"expected a 2-tuple of datetime objects, got: {value}"

        return True, ""
    if output_type == AllTimeRanges:
        if not isinstance(value, dict):
            return False, f"{value} is not an instance of dict"
        for key, v in value.items():
            if not isinstance(key, str):
                return False, f"{key} is not an instance of str"
            is_valid, message = is_valid_output(v, output_type=TimeRanges)
            if not is_valid:
                return False, message

        return True, ""

    if output_type == ModelDict:
        if not isinstance(value, dict):
            return (False, f"{value} is not a model dict")
        if set(value.keys()) != set(["class_def", "model", "metadata"]):
            return (
                False,
                (
                    f"{value} must contain keys: 'class_def', 'model' and no others,"
                    f" Got {list(value.keys())}"
                ),
            )

        if not isinstance(value["class_def"], str):
            return False, f"'class_def' must be a str, got {type(value['class_def'])}"
        if not isinstance(value["model"], torch.nn.Module):
            return (
                False,
                f"'model' must be of type torch.nn.Module, got {type(value['model'])}",
            )
        if not isinstance(value["metadata"], dict):

            return (
                False,
                f"'metadata' must be a dict, not {type(value['metadata'])}",
            )
        try:
            json.dumps(value["metadata"])
        except TypeError as e:
            return False, f"Metadata must be json serializable. Got {e}"
        return True, ""

    if output_type == torch.nn.Module:
        return isinstance(value, torch.nn.Module), ""

    origin = get_origin(output_type)
    args = get_args(output_type)

    # Normalize dict type aliases
    def is_dict_str_figure(typ):
        return get_origin(typ) in {dict, Dict} and get_args(typ) == (str, go.Figure)

    # Handle Union[Figure, Dict[str, Figure]] robustly
    if origin is Union:
        if go.Figure in args and any(is_dict_str_figure(arg) for arg in args):
            if isinstance(value, go.Figure):
                return True, ""
            if isinstance(value, dict):
                return (
                    all(
                        isinstance(k, str) and isinstance(v, go.Figure)
                        for k, v in value.items()
                    ),
                    "",
                )
            return (
                False,
                "Must return either go.Figure or dictionary of keys str and values"
                f" go.Figure. Got {value}",
            )

    # Fallback
    return is_allowed_type(value)

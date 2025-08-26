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
import hashlib
import builtins
import typing
import typing_extensions
import sqlite3
import plotly.graph_objects as go
import torch
from quickbooks import QuickBooks

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
    type(None),
]

SimParams = typing.Dict[Text, typing.Hashable]
FrozenSimParams = Tuple[Tuple[Text, typing.Hashable], ...]
AllSimParams = typing.Iterable[SimParams]
SimValues = Dict[FrozenSimParams, Allowed]
SimValue = Tuple[SimParams, Allowed]

chunked_type_map = {AllSimParams: SimParams, SimValues: SimValue, Allowed: Allowed}


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


def hash_schema(schema):
    return hashlib.md5(str(schema).encode()).hexdigest()


def describe_json_schema(obj, definitions=None, path="", max_len: int = 32):
    if definitions is None:
        definitions = {}

    import torch

    if isinstance(obj, torch.nn.Module):
        import inspect

        try:
            sig = inspect.signature(obj.__class__.__init__)
            args_info = {
                k: str(v.annotation) if v.annotation != inspect._empty else "Any"
                for k, v in sig.parameters.items()
                if k != "self"
            }
        except Exception:
            args_info = {}

        schema = {
            "x-type": "torch.nn.Module",
            "type": "object",
            "properties": {"class": obj.__class__.__name__, "args": args_info},
        }

    if isinstance(obj, dict):
        # Represent dict as array of [key, value] pairs
        if len(obj) < max_len:
            properties = {}
            required = []
            for k, v in obj.items():
                k = str(k)
                key_schema, definitions = describe_json_schema(
                    k, definitions, f"{path}/key/{k}"
                )
                val_schema, definitions = describe_json_schema(
                    v, definitions, f"{path}/value/{k}"
                )

                properties[k] = {"key": key_schema, "value": val_schema}
                required.append(k)
            schema = {
                "x-type": "dict",
                "type": "object",
                "properties": properties,
                "required": required,
            }
        else:
            # Get the schema from the *first* key and value
            first_key, first_val = next(iter(obj.items()))
            key_schema, definitions = describe_json_schema(
                first_key, definitions, path + "/key"
            )
            val_schema, definitions = describe_json_schema(
                first_val, definitions, path + "/value"
            )

            schema = {
                "x-type": "dict",
                "type": "object",
                "properties": {
                    "keys": key_schema,
                    "values": val_schema,
                    "length": len(obj),
                },
            }

    elif isinstance(obj, list):
        if obj:
            items_schema, definitions = describe_json_schema(
                obj[0], definitions, path + "/items"
            )
            schema = {
                "x-type": "list",
                "type": "array",
                "minItems": len(obj),
                "maxItems": len(obj),
                "items": items_schema,
            }
        else:
            schema = {"x-type": "list", "type": "array", "items": {}}
    elif isinstance(obj, (frozenset, set)):
        x_type = "frozenset" if isinstance(obj, frozenset) else "set"
        if obj:
            items_schema, definitions = describe_json_schema(
                next(iter(obj)), definitions, path + "/items"
            )
            schema = {
                "x-type": x_type,
                "type": "array",
                "minItems": len(obj),
                "maxItems": len(obj),
                "items": items_schema,
                "uniqueItems": True,
            }
        else:
            schema = {
                "x-type": x_type,
                "type": "array",
                "minItems": len(obj),
                "maxItems": len(obj),
                "items": {},
                "uniqueItems": True,
            }

    elif isinstance(obj, np.ndarray):
        schema = {
            "x-type": "np.ndarray",
            "type": "object",
            "properties": {"shape": list(obj.shape), "dtype": str(obj.dtype)},
        }
    elif isinstance(obj, np.generic):
        schema = {
            "x-type": "np.generic",
            "type": "object",
            "properties": {"dtype": str(obj.dtype)},
        }
    elif isinstance(obj, pd.DataFrame):
        schema = {
            "x-type": "pd.dataframe",
            "type": "object",
            "properties": {
                "columns": {col: str(dtype) for col, dtype in obj.dtypes.items()},
                "shape": list(obj.shape),
            },
        }

    elif isinstance(obj, int):
        schema = {"type": "integer"}
    elif isinstance(obj, float):
        schema = {"type": "number"}
    elif isinstance(obj, str):
        schema = {"type": "string"}
    elif isinstance(obj, bytes):
        schema = {"x-type": "bytes", "type": "string"}
    elif obj is None:
        schema = {"type": "null"}
    else:
        schema = {"type": "unknown"}

    # Deduplication
    key = hash_schema(schema)
    if key not in definitions:
        definitions[key] = schema
    return {"$ref": f"#/definitions/{key}"}, definitions


def describe_allowed(obj, definitions=None):
    schema, definitions = describe_json_schema(obj, definitions=definitions)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        **schema,
        "definitions": definitions,
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
            pd.Period,
            type(None),
        ),
    ):
        return True

    elif isinstance(obj, Mapping):
        return all(is_allowed_type(k) and is_allowed_type(v) for k, v in obj.items())

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return all(is_allowed_type(item) for item in obj)

    return False


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
        "utils.type_utils.AllSimParams": AllSimParams,
        "utils.type_utils.SimParams": SimParams,
        "utils.type_utils.FrozenSimParams": FrozenSimParams,
        "utils.type_utils.SimValues": SimValues,
        "utils.type_utils.SimValue": SimValue,
        "utils.type_utils.Allowed": Allowed,
        "utils.type_utils.GCSPath": GCSPath,
        "typing.Hashable": typing.Hashable,
        "typing.Iterable": typing.Iterable,
        "typing.Tuple": typing.Tuple,
    }

    from pymongo.mongo_client import MongoClient as PyMongoClient
    from psycopg2.extensions import connection as Psycopg2Connection
    from google.cloud.bigquery import Client as BigQueryClient
    import torch
    import pymongo

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
        print(e)
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
        return isinstance(value, sqlite3.Connection)
    if output_type == SimParams:
        try:
            if not isinstance(value, dict):
                return False
            for key, value in value.items():
                if not isinstance(key, str):
                    return False
                hash(value)
        except TypeError:
            return False

        return True
    if output_type == AllSimParams:
        if not isinstance(value, abc_Iterable):
            return False
        for sim_params in value:
            if not is_valid_output(sim_params, output_type=SimParams):
                return False

        return True
    if output_type == SimValues:
        if not isinstance(value, dict):
            return False
        for frzn, alwd in value.items():
            if not isinstance(frzn, tuple):
                return False
            try:
                for k, v in frzn:
                    if not isinstance(k, str):
                        return False
                    hash(v)
            except TypeError:
                return False

            if not is_allowed_type(alwd):
                return False

        return True

    if output_type == SimValue:
        if not isinstance(value, tuple) or not len(value) == 2:
            return False
        frzn, alwd = value

        if not is_valid_output(frzn, SimParams):
            return False

        if not is_allowed_type(alwd):
            return False

        return True

    if output_type == torch.nn.Module:
        return isinstance(value, torch.nn.Module)

    origin = get_origin(output_type)
    args = get_args(output_type)

    # Normalize dict type aliases
    def is_dict_str_figure(typ):
        return get_origin(typ) in {dict, Dict} and get_args(typ) == (str, go.Figure)

    # Handle Union[Figure, Dict[str, Figure]] robustly
    if origin is Union:
        if go.Figure in args and any(is_dict_str_figure(arg) for arg in args):
            if isinstance(value, go.Figure):
                return True
            if isinstance(value, dict):
                return all(
                    isinstance(k, str) and isinstance(v, go.Figure)
                    for k, v in value.items()
                )
            return False

    # Fallback
    return is_allowed_type(value)

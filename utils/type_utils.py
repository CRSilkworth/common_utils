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
    FrozenSet,
    Hashable,
)
import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence
import hashlib
import builtins
import typing
import typing_extensions
import sqlite3
import plotly.graph_objects as go

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
    Dict[Union[Text, int], "Allowed"],
    List["Allowed"],
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

AllSimParams = typing.Iterable[Dict[Text, typing.Hashable]]
SimParamKey = FrozenSet[Tuple[Text, typing.Hashable]]
SimValues = Dict[SimParamKey, Allowed]


def hash_schema(schema):
    return hashlib.md5(str(schema).encode()).hexdigest()


def describe_json_schema(obj, definitions=None, path="", with_db: bool = True):
    if definitions is None:
        definitions = {}

    if with_db:
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
                "type": "torch.nn.Module",
                "class": obj.__class__.__name__,
                "args": args_info,
            }

    if isinstance(obj, dict):
        properties = {}
        required = []
        for k, v in obj.items():
            sub_schema, definitions = describe_json_schema(
                v, definitions, path + "/" + k, with_db=with_db
            )
            properties[k] = sub_schema
            required.append(k)
        schema = {"type": "object", "properties": properties, "required": required}

    elif isinstance(obj, list):
        if obj:
            items_schema, definitions = describe_json_schema(
                obj[0], definitions, path + "/items", with_db=with_db
            )
            schema = {"type": "array", "items": items_schema}
        else:
            schema = {"type": "array", "items": {}}

    elif isinstance(obj, np.ndarray):
        schema = {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    elif isinstance(obj, np.generic):
        schema = {"type": "np.generic", "dtype": str(obj.dtype)}
    elif isinstance(obj, pd.DataFrame):
        schema = {
            "type": "dataframe",
            "columns": {col: str(dtype) for col, dtype in obj.dtypes.items()},
            "shape": list(obj.shape),
        }

    elif isinstance(obj, int):
        schema = {"type": "integer"}
    elif isinstance(obj, float):
        schema = {"type": "number"}
    elif isinstance(obj, str):
        schema = {"type": "string"}
    elif isinstance(obj, bytes):
        schema = {"type": "string"}
    elif obj is None:
        schema = {"type": "null"}

    else:
        schema = {"type": "unknown"}

    # Deduplication
    key = hash_schema(schema)
    if key not in definitions:
        definitions[key] = schema
    return {"$ref": f"#/definitions/{key}"}, definitions


def describe_allowed(obj, with_db: bool = True):
    schema, definitions = describe_json_schema(obj, with_db=with_db)
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


def serialize_typehint(t: type, with_db: bool = True) -> str:
    """Serialize a type hint to a string."""
    if isinstance(t, str):
        return t

    # Handle special case: non-subscriptable types
    if t in {Hashable, Iterable}:
        return f"{t.__module__}.{t.__qualname__}"

    if isinstance(t, TypeVar):
        return f"TypeVar({t.__name__})"

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


def get_known_types(
    custom_types: Optional[Dict[Text, Any]] = None, with_db: bool = True
):
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
        "utils.type_utils.PositionDict": PositionDict,
        "NoneType": type(None),
        "utils.type_utils.AllSimParams": AllSimParams,
        "typing.Hashable": Hashable,
        "typing.Iterable": Iterable,
        "Hashable": Hashable,
        "Iterable": Iterable,
    }

    if with_db:
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


def deserialize_typehint(
    s: str, custom_types: dict = None, with_db: bool = True
) -> type:
    """Deserialize a string back into a type hint."""

    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s).__name__}: {s}")

    known_types = get_known_types(custom_types=custom_types, with_db=with_db)

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


def is_valid_output(value, output_type, with_db: bool = True):
    if output_type == sqlite3.Connection:
        return isinstance(value, sqlite3.Connection)
    if output_type == AllSimParams:
        try:
            for sim_params in value:
                if not isinstance(sim_params, dict):
                    return False
                for key, value in sim_params.items():
                    if not isinstance(key, str):
                        return False
                    hash(value)
        except TypeError:
            return False

        return True
    origin = get_origin(output_type)
    args = get_args(output_type)

    if origin in {dict, Dict} and len(args) == 2:
        key_type, val_type = args
        print("in origin")
        if key_type == SimParamKey and val_type == Allowed:
            print("HERE")
            if not isinstance(value, dict):
                print("dict")
                return False
            for frzn, alwd in value.items():
                if not isinstance(frzn, frozenset):
                    print("frozen set")
                    return False
                try:
                    for k, v in frzn:
                        if not isinstance(k, str):
                            print("froz")
                            return False
                        hash(v)
                except TypeError:
                    print("hash")

                    return False

                if not is_allowed_type(alwd):
                    print("allowed")

                    return False

            return True
    if with_db:
        import torch

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

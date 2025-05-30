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
Position = Union[Tuple[Optional[int], Optional[int]], List[Optional[int]]]


class PositionDict(TypedDict):
    x: int
    y: int


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


def hash_schema(schema):
    return hashlib.md5(str(schema).encode()).hexdigest()


def describe_json_schema(obj, definitions=None, path=""):
    if definitions is None:
        definitions = {}

    if isinstance(obj, dict):
        properties = {}
        required = []
        for k, v in obj.items():
            sub_schema, definitions = describe_json_schema(
                v, definitions, path + "/" + k
            )
            properties[k] = sub_schema
            required.append(k)
        schema = {"type": "object", "properties": properties, "required": required}

    elif isinstance(obj, list):
        if obj:
            items_schema, definitions = describe_json_schema(
                obj[0], definitions, path + "/items"
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


def describe_allowed(obj):
    schema, definitions = describe_json_schema(obj)
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
    """Serialize a type hint to a string."""
    if isinstance(t, str):
        return repr(t)

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


def deserialize_typehint(
    s: str, custom_types: dict = None, with_db: bool = True
) -> type:
    """Deserialize a string back into a type hint."""

    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s).__name__}: {s}")

    # Create the base eval namespace
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
    }

    if with_db:
        from pymongo.mongo_client import MongoClient as PyMongoClient
        from psycopg2.extensions import connection as Psycopg2Connection
        from google.cloud.bigquery import Client as BigQueryClient
        import pymongo

        known_types.update(
            {
                "pymongo": __import__("pymongo"),
                "pymongo.mongo_client": pymongo.mongo_client,
                "pymongo.synchronous.mongo_client": pymongo.mongo_client,
                "pymongo.mongo_client.MongoClient": PyMongoClient,
                "pymongo.synchronous.mongo_client.MongoClient": PyMongoClient,
                "psycopg2": __import__("psycopg2"),
                # "psycopg2.extensions": psycopg2.extensions,
                "psycopg2.extensions.connection": Psycopg2Connection,
                "google": __import__("google"),
                "google.cloud": __import__("google.cloud", fromlist=["bigquery"]),
                "google.cloud.bigquery": __import__(
                    "google.cloud.bigquery", fromlist=["client"]
                ),
                # "google.cloud.bigquery.client": google.cloud.bigquery.client,
                "google.cloud.bigquery.client.Client": BigQueryClient,
            }
        )

    if custom_types:
        known_types.update(custom_types)

    if s.startswith("TypeVar(") and s.endswith(")"):
        name = s[len("TypeVar(") : -1]  # noqa: E203

        if name in known_types:
            return known_types[name]
        else:
            raise ValueError(f"Unknown TypeVar {name}")

    try:
        return eval(s, known_types)
    except Exception as e:
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

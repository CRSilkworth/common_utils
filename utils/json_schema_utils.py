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
import re
import datetime
from pymongo.mongo_client import MongoClient as PyMongoClient
from psycopg2.extensions import connection as Psycopg2Connection
from google.cloud.bigquery import Client as BigQueryClient
import pymongo
import json
import inspect
import logging


def torch_schema(cls: torch.nn.Module) -> Dict[Text, Any]:

    try:
        sig = inspect.signature(cls.__init__)
        args_info = {
            k: str(v.annotation) if v.annotation != inspect._empty else "Any"
            for k, v in sig.parameters.items()
            if k != "self"
        }
    except Exception:
        args_info = {}

    return {
        "x-type": "torch.nn.Module",
        "type": "object",
        "properties": {"class": cls.__name__, "args": args_info},
    }


def np_array_schema(shape: Iterable[int], dtype: np.dtype) -> Dict[Text, Any]:
    schema = {
        "x-type": "np.ndarray",
        "type": "object",
        "properties": {"shape": list(shape), "dtype": str(dtype)},
    }
    return schema


def np_generic_schema(dtype: np.dtype) -> Dict[Text, Any]:
    schema = {
        "x-type": "np.generic",
        "type": "object",
        "properties": {"dtype": str(dtype)},
    }
    return schema


def dataframe_schema(dtypes: Dict[Text, Any], shape: Iterable[int]) -> Dict[Text, Any]:
    schema = {
        "x-type": "pd.DataFrame",
        "type": "object",
        "properties": {
            "columns": {col: str(dtype) for col, dtype in dtypes.items()},
            "shape": list(shape),
        },
    }
    return schema


def series_schema(
    index_schema: Dict[Text, Any], val_schema: Dict[Text, Any], length: int
) -> Dict[Text, Any]:
    schema = {
        "x-type": "pd.Series",
        "type": "object",
        "properties": {"values": val_schema, "index": index_schema, "length": length},
    }
    return schema


def time_range_schema_and_defs() -> Dict[Text, Any]:
    return describe_json_schema(
        (datetime.datetime(2025, 1, 1), datetime.datetime(2026, 1, 1))
    )


def dict_schema(
    properties: Dict[Text, Any],
    length: Optional[int] = None,
    required: Optional[List[Text]] = None,
) -> Dict[Text, Any]:
    schema = {
        "x-type": "dict",
        "type": "object",
        "properties": properties,
    }
    if length is not None:
        schema["x-length"] = length
    if required is not None:
        schema["required"] = required

    return schema


def iterable_schema(
    item_schema: Optional[Dict[Text, Any]] = None,
    length: Optional[int] = None,
    unique_items: bool = False,
    x_type: str = "iterable",
) -> Dict[Text, Any]:
    schema = {"x-type": x_type, "type": "array"}
    if item_schema is not None:
        schema["items"] = item_schema
    if length is not None:
        schema["minItems"] = length
        schema["maxItems"] = length
    if unique_items:
        schema["uniqueItems"] = True

    return schema


def describe_json_schema(obj, defs=None, max_len: int = 32):
    if defs is None:
        defs = {}

    import torch

    if isinstance(obj, torch.nn.Module):
        schema = torch_schema(cls=obj.__class__)

    if isinstance(obj, dict):
        # Represent dict as array of [key, value] pairs
        if len(obj) < max_len:
            properties = {}
            required = []
            for k, v in obj.items():
                k = str(k)
                key_schema_hash, defs = describe_json_schema(k, defs)
                val_schema_hash, defs = describe_json_schema(v, defs)
                pair_schema = {
                    "type": "object",
                    "x-type": "key-value-pair",
                    "properties": {
                        "key": {"$ref": f"#/$defs/{key_schema_hash}"},
                        "value": {"$ref": f"#/$defs/{val_schema_hash}"},
                    },
                    "required": ["key", "value"],
                }
                pair_schema_hash = hash_schema(pair_schema)
                defs[pair_schema_hash] = pair_schema
                properties[k] = {"$ref": f"#/$defs/{pair_schema_hash}"}
                required.append(k)

            schema = dict_schema(
                properties=properties,
                length=len(obj),
                required=required,
            )
        else:
            # Get the schema from the *first* key and value
            first_key, first_val = next(iter(obj.items()))
            key_schema_hash, defs = describe_json_schema(first_key, defs)
            val_schema_hash, defs = describe_json_schema(first_val, defs)

            properties = {
                "key_schema": {"$ref": f"#/$defs/{key_schema_hash}"},
                "value_schema": {"$ref": f"#/$defs/{val_schema_hash}"},
            }
            schema = dict_schema(
                properties=properties,
                length=len(obj),
                required=["key_schema", "value_schema"],
            )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        x_type = obj.__class__.__name__
        if len(obj) < max_len:
            items_schema = []
            for _, item in enumerate(obj):
                item_schema, defs = describe_json_schema(item, defs)
                items_schema.append(item_schema)

        else:
            items_schema, defs = describe_json_schema(obj[0], defs)
        schema = iterable_schema(
            item_schema=items_schema,
            length=len(obj),
            x_type=x_type,
            unique_items=isinstance(obj, (set, frozenset)),
        )

    elif isinstance(obj, np.ndarray):
        schema = np_array_schema(shape=obj.shape, dtype=obj.dtype)
    elif isinstance(obj, np.generic):
        schema = np_generic_schema(dtype=obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        schema = dataframe_schema(dtypes=obj.dtypes, shape=obj.shape)
    elif isinstance(obj, pd.Series):
        val_schema, defs = describe_json_schema(obj.values, defs)
        index_schema, defs = describe_json_schema(obj.index.tolist(), defs)
        schema = series_schema(
            val_schema=val_schema, index_schema=index_schema, length=len(obj)
        )
    elif isinstance(obj, pd.Period):
        schema = {"x-type": "pd.Period", "type": "object"}
    elif isinstance(obj, datetime.datetime):
        schema = {"x-type": "datetime", "type": "object"}
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

    schema_hash = hash_schema(schema)
    if schema_hash not in defs:
        defs[schema_hash] = schema

    return schema_hash, defs


def hash_schema(schema):
    return hashlib.md5(str(schema).encode()).hexdigest()

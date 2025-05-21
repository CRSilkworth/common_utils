from typing import Union, Any, Optional
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import sqlite3
import io
from utils.misc_utils import failed_output
from dataclasses import is_dataclass, asdict
import datetime
import traceback


def encode_obj(obj: Any):
    if is_dataclass(obj):
        return {
            "__kind__": "dataclass",
            "class": obj.__class__.__name__,
            "data": encode_obj(asdict(obj)),
        }

    if isinstance(obj, go.Figure):
        return {"__kind__": "PlotlyFigure", "data": encode_obj(obj.to_dict())}

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        for col in df.columns:
            if isinstance(df[col].dtype, pd.PeriodDtype):
                df[col] = df[col].astype(str)
            elif isinstance(df[col].dtype, pd.DatetimeTZDtype):
                df[col] = df[col].astype(str)
        return {"__kind__": "DataFrame", "data": df.to_json(orient="split")}

    elif isinstance(obj, pd.Series):
        s = obj.copy()
        if isinstance(s.dtype, pd.PeriodDtype):
            s = s.astype(str)
        return {"__kind__": "Series", "data": s.to_json(orient="split")}

    elif isinstance(obj, pd.Period):
        return {"__kind__": "Period", "data": str(obj), "freq": obj.freqstr}

    elif isinstance(obj, pd.PeriodIndex):
        return {
            "__kind__": "PeriodIndex",
            "data": [str(p) for p in obj],
            "freq": obj.freqstr,
        }

    elif isinstance(obj, pd.Timestamp):
        return {"__kind__": "Timestamp", "data": obj.isoformat()}

    elif isinstance(obj, datetime.datetime):
        return {"__kind__": "datetime", "data": obj.isoformat()}

    elif isinstance(obj, datetime.date):
        return {"__kind__": "date", "data": obj.isoformat()}

    elif isinstance(obj, datetime.time):
        return {"__kind__": "time", "data": obj.isoformat()}

    elif isinstance(obj, pd.Index):
        return {"__kind__": "Index", "data": obj.tolist()}

    elif isinstance(obj, np.ndarray):
        return {
            "__kind__": "ndarray",
            "data": [encode_obj(x) for x in obj.tolist()],
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }

    elif isinstance(obj, bytes):
        return {"__kind__": "bytes", "data": obj.decode("utf-8")}

    elif isinstance(obj, (list, tuple)):
        return [encode_obj(v) for v in obj]

    elif isinstance(obj, dict):
        return {str(k): encode_obj(v) for k, v in obj.items()}

    else:
        return obj


def decode_obj(obj: Any):
    if isinstance(obj, dict):
        kind = obj.get("__kind__")
        if kind == "PlotlyFigure":
            return go.Figure(decode_obj(obj["data"]))
        elif kind == "DataFrame":
            return pd.read_json(obj["data"], orient="split")
        elif kind == "Series":
            return pd.read_json(obj["data"], orient="split", typ="series")
        elif kind == "Period":
            return pd.Period(obj["data"], freq=obj["freq"])
        elif kind == "PeriodIndex":
            return pd.PeriodIndex(obj["data"], freq=obj["freq"])
        elif kind == "Timestamp":
            return pd.Timestamp(obj["data"])
        elif kind == "datetime":
            return datetime.datetime.fromisoformat(obj["data"])
        elif kind == "date":
            return datetime.date.fromisoformat(obj["data"])
        elif kind == "time":
            return datetime.time.fromisoformat(obj["data"])
        elif kind == "Index":
            return pd.Index(obj["data"])
        elif kind == "ndarray":
            return np.array(
                [decode_obj(x) for x in obj["data"]], dtype=obj["dtype"]
            ).reshape(obj["shape"])
        elif kind == "bytes":
            return obj["data"].encode("utf-8")
        else:
            return {k: decode_obj(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [decode_obj(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(decode_obj(v) for v in obj)
    else:
        return obj


def serialize_value(value: Any, value_type: Optional[Any] = None, with_db: bool = True):
    if value is None:
        return value
    if value_type == sqlite3.Connection:
        buffer = io.BytesIO()
        for line in value.iterdump():
            buffer.write(f"{line}\n".encode("utf-8"))
        buffer.seek(0)

        value = buffer.read()
    return json.dumps(encode_obj(value))


def deserialize_value(value, value_type, with_db: bool = True):
    value = None if value is None else decode_obj(json.loads(value))
    if value_type == sqlite3.Connection and value is not None:
        buffer = io.BytesIO(value)
        conn = sqlite3.connect(":memory:")

        cursor = conn.cursor()

        # Execute dump to restore database
        script = buffer.getvalue().decode("utf-8")
        cursor.executescript(script)
        value = conn

    if with_db:
        from pymongo.mongo_client import MongoClient as PyMongoClient
        from psycopg2.extensions import connection as Psycopg2Connection
        from google.cloud.bigquery import Client as BigQueryClient
        from utils.db_utils import connect_to_biquery, connect_to_mongo, connect_to_sql

        if (
            value_type == Union[PyMongoClient, Psycopg2Connection, BigQueryClient]
            and value is not None
        ):
            if value["db_type"] == "mongo":
                value = connect_to_mongo(value)
            elif value["db_type"] == "bigquery":
                value = connect_to_biquery(value)
            elif value["db_type"] == "sql":
                value = connect_to_sql(value)
            else:
                raise ValueError(f"Unsupported db type:{value['db_type']}")

    return value


def attempt_deserialize(
    value: Any, value_type: Optional[Any] = None, with_db: bool = True
):
    cleanups = []
    output = {}
    try:
        deserialized_dict = deserialize_value(value, value_type, with_db=with_db)
    except Exception:
        message = f"Deserialize failed: {traceback.format_exc()}"
        output = {"value_setter": failed_output(message)}
        return None, output, cleanups

    if with_db:
        from pymongo.mongo_client import MongoClient as PyMongoClient
        from psycopg2.extensions import connection as Psycopg2Connection
        from google.cloud.bigquery import Client as BigQueryClient

        if value_type == Union[PyMongoClient, Psycopg2Connection, BigQueryClient]:
            deserialized_value = deserialized_dict["connection"]
            if deserialized_dict.get("cleanup", None):
                cleanups.append(deserialized_dict["cleanup"])

        else:
            deserialized_value = deserialized_dict
    else:
        deserialized_value = deserialized_dict
    return deserialized_value, output, cleanups


def attempt_serialize(
    value: Any, value_type: Optional[Any] = None, with_db: bool = True
):
    output = {}
    try:
        value = serialize_value(value, value_type=value_type, with_db=with_db)
    except Exception:
        output = failed_output(f"Failed to serialize value: {traceback.format_exc()}")
    return value, output

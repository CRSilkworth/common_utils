from typing import Any
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go


def encode_obj(obj):
    if isinstance(obj, go.Figure):
        return {"__kind__": "PlotlyFigure", "data": obj.to_dict()}
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

    elif isinstance(obj, pd.Index):
        return {"__kind__": "Index", "data": obj.tolist()}

    elif isinstance(obj, np.ndarray):
        return {
            "__kind__": "ndarray",
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }

    elif isinstance(obj, (list, tuple)):
        return [encode_obj(v) for v in obj]

    elif isinstance(obj, dict):
        return {str(k): encode_obj(v) for k, v in obj.items()}

    else:
        return obj


def decode_obj(obj):

    if isinstance(obj, dict):
        kind = obj.get("__kind__")
        if kind == "PlotlyFigure":
            return go.Figure(obj["data"])
        if kind == "DataFrame":
            return pd.read_json(obj["data"], orient="split")
        elif kind == "Series":
            return pd.read_json(obj["data"], orient="split", typ="series")
        elif kind == "Period":
            return pd.Period(obj["data"], freq=obj["freq"])
        elif kind == "PeriodIndex":
            return pd.PeriodIndex(obj["data"], freq=obj["freq"])
        elif kind == "Timestamp":
            return pd.Timestamp(obj["data"])
        elif kind == "Index":
            return pd.Index(obj["data"])
        elif kind == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"])
        else:
            return {k: decode_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_obj(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(decode_obj(v) for v in obj)
    else:
        return obj


def serialize(obj: Any):
    if obj is None:
        return None
    return json.dumps(encode_obj(obj))


def deserialize(encoded_str):
    if encoded_str is None:
        return None
    raw = json.loads(encoded_str)
    return decode_obj(raw)

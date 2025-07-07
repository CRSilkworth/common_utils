import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import sqlite3
import utils.serialize_utils as serialize_utils


def test_encode_decode_basic_types():
    # Test encoding and decoding of basic types
    assert serialize_utils.encode_obj(123) == 123
    assert serialize_utils.encode_obj("test") == "test"
    assert serialize_utils.encode_obj(None) is None


def test_encode_decode_bytes():
    data = b"hello bytes"
    encoded = serialize_utils.encode_obj(data)
    assert encoded["__kind__"] == "bytes"
    decoded = serialize_utils.decode_obj(encoded)
    assert decoded == data


def test_encode_decode_numpy():
    arr = np.array([1, 2, 3])
    encoded = serialize_utils.encode_obj(arr)
    assert encoded["__kind__"] == "ndarray"
    decoded = serialize_utils.decode_obj(encoded)
    np.testing.assert_array_equal(decoded, arr)

    # Test encoding numpy scalar
    scalar = np.int32(10)
    encoded_scalar = serialize_utils.encode_obj(scalar)
    assert isinstance(encoded_scalar, int)


def test_encode_decode_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    encoded_df = serialize_utils.encode_obj(df)
    assert encoded_df["__kind__"] == "DataFrame"
    decoded_df = serialize_utils.decode_obj(encoded_df)
    pd.testing.assert_frame_equal(decoded_df, df)

    s = pd.Series([1, 2, 3])
    encoded_s = serialize_utils.encode_obj(s)
    assert encoded_s["__kind__"] == "Series"
    decoded_s = serialize_utils.decode_obj(encoded_s)
    pd.testing.assert_series_equal(decoded_s, s)

    period = pd.Period("2023-01", freq="M")
    encoded_p = serialize_utils.encode_obj(period)
    assert encoded_p["__kind__"] == "Period"
    decoded_p = serialize_utils.decode_obj(encoded_p)
    assert decoded_p == period

    pi = pd.PeriodIndex(["2023-01", "2023-02"], freq="M")
    encoded_pi = serialize_utils.encode_obj(pi)
    assert encoded_pi["__kind__"] == "PeriodIndex"
    decoded_pi = serialize_utils.decode_obj(encoded_pi)
    assert all(decoded_pi == pi)

    ts = pd.Timestamp("2023-01-01T12:00:00")
    encoded_ts = serialize_utils.encode_obj(ts)
    assert encoded_ts["__kind__"] == "Timestamp"
    decoded_ts = serialize_utils.decode_obj(encoded_ts)
    assert decoded_ts == ts

    idx = pd.Index([1, 2, 3])
    encoded_idx = serialize_utils.encode_obj(idx)
    assert encoded_idx["__kind__"] == "Index"
    decoded_idx = serialize_utils.decode_obj(encoded_idx)
    assert all(decoded_idx == idx)


def test_encode_decode_datetime():
    dt = datetime.datetime(2023, 1, 1, 12, 30)
    encoded = serialize_utils.encode_obj(dt)
    assert encoded["__kind__"] == "datetime"
    decoded = serialize_utils.decode_obj(encoded)
    assert decoded == dt

    date = datetime.date(2023, 1, 1)
    encoded_date = serialize_utils.encode_obj(date)
    assert encoded_date["__kind__"] == "date"
    decoded_date = serialize_utils.decode_obj(encoded_date)
    assert decoded_date == date

    time = datetime.time(12, 30)
    encoded_time = serialize_utils.encode_obj(time)
    assert encoded_time["__kind__"] == "time"
    decoded_time = serialize_utils.decode_obj(encoded_time)
    assert decoded_time == time


def test_encode_decode_tuple_and_list():
    tup = (1, 2, 3)
    encoded_tup = serialize_utils.encode_obj(tup)
    assert encoded_tup["__kind__"] == "tuple"
    decoded_tup = serialize_utils.decode_obj(encoded_tup)
    assert decoded_tup == tup

    lst = [1, 2, 3]
    encoded_lst = serialize_utils.encode_obj(lst)
    assert isinstance(encoded_lst, list)
    decoded_lst = serialize_utils.decode_obj(encoded_lst)
    assert decoded_lst == lst


def test_encode_decode_dict():
    d = {"a": 1, "b": 2}
    encoded = serialize_utils.encode_obj(d)
    assert encoded["__kind__"] == "dict"
    decoded = serialize_utils.decode_obj(encoded)
    assert decoded == d


def test_serialize_deserialize_value_sqlite():
    # Create a sqlite database in memory and serialize it
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
    cur.execute("INSERT INTO test (val) VALUES ('abc')")
    conn.commit()

    serialized = serialize_utils.serialize_value(conn, value_type=sqlite3.Connection)
    assert isinstance(serialized, str)

    deserialized = serialize_utils.deserialize_value(
        serialized, value_type=sqlite3.Connection
    )
    cur2 = deserialized.cursor()
    cur2.execute("SELECT val FROM test")
    res = cur2.fetchone()[0]
    assert res == "abc"


def test_attempt_serialize_deserialize(monkeypatch):
    val = {"key": "value"}
    serialized, output = serialize_utils.attempt_serialize(val)
    assert output == {}
    assert isinstance(serialized, str)

    deserialized, output, cleanups = serialize_utils.attempt_deserialize(serialized)
    assert output == {}
    assert isinstance(deserialized, dict)

    # Test deserialize failure
    def fail_deserialize(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr(serialize_utils, "deserialize_value", fail_deserialize)
    deserialized, output, cleanups = serialize_utils.attempt_deserialize(serialized)
    assert deserialized is None


def test_model_builder_function_string():
    class_def = "class Model:\n    pass"
    func_str = serialize_utils._model_builder_function_string(class_def)
    assert "def model_builder()" in func_str
    assert class_def in func_str


def test_encode_obj_with_db_dict_special(monkeypatch):
    # Prepare a dummy torch mock
    class DummyModel:
        def state_dict(self):
            return {"weights": [1, 2, 3]}

    class DummyTorchModule:
        def __init__(self):
            self.Module = None

        def save(self, obj, buffer):
            buffer.write(b"state")

    # monkeypatch torch
    class DummyTorch:
        def save(self, state_dict, buffer):
            buffer.write(b"dummy")

    # monkeypatch encode_obj to check recursive call
    dummy_model = DummyModel()
    obj = {"class_def": "def Model(): pass", "model": dummy_model}

    monkeypatch.setitem(__import__("builtins").__dict__, "torch", DummyTorch())

    # Actually test encode_obj for this special case
    result = serialize_utils.encode_obj(obj, with_db=True)
    assert result["__kind__"] == "TorchModel"
    assert "class_def" in result["data"]


def test_decode_obj_plotly_figure():
    fig = go.Figure(data=go.Bar(y=[2, 3]))
    encoded = serialize_utils.encode_obj(fig)
    decoded = serialize_utils.decode_obj(encoded)
    assert isinstance(decoded, go.Figure)


def test_decode_obj_unknown_kind():
    obj = {"a": 1, "b": 2}
    decoded = serialize_utils.decode_obj(obj)
    assert decoded == obj


def test_serialize_value_none():
    assert serialize_utils.serialize_value(None) is None


def test_deserialize_value_none():
    assert serialize_utils.deserialize_value(None, None) is None


# Additional test: test that decode_obj returns input as is for non dict/list


def test_decode_obj_non_dict_list():
    val = 123
    assert serialize_utils.decode_obj(val) == val

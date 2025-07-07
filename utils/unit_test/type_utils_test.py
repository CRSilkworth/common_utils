import typing
import numpy as np
import pandas as pd
import pytest
import sqlite3
from utils import type_utils

import plotly.graph_objects as go


def test_hash_schema_consistency():
    schema = {"type": "string"}
    h1 = type_utils.hash_schema(schema)
    h2 = type_utils.hash_schema(schema)
    assert isinstance(h1, str)
    assert h1 == h2


def test_describe_json_schema_basic_types():
    # Test int
    schema, defs = type_utils.describe_json_schema(42)
    assert "$ref" in schema
    # Test str
    schema, defs = type_utils.describe_json_schema("test")
    assert "$ref" in schema
    # Test None
    schema, defs = type_utils.describe_json_schema(None)
    assert "$ref" in schema


def test_describe_json_schema_dict_small_and_large():
    small_dict = {"a": 1, "b": 2}
    schema, defs = type_utils.describe_json_schema(small_dict, max_len=10)
    assert schema["$ref"]

    large_dict = {i: i for i in range(100)}
    schema, defs = type_utils.describe_json_schema(large_dict, max_len=10)
    assert schema["$ref"]


def test_describe_json_schema_list_and_set():
    lst = [1, 2, 3]
    schema, defs = type_utils.describe_json_schema(lst)
    assert schema["$ref"]

    empty_lst = []
    schema, defs = type_utils.describe_json_schema(empty_lst)
    assert schema["$ref"]

    st = {1, 2, 3}
    schema, defs = type_utils.describe_json_schema(st)
    assert schema["$ref"]

    empty_st = set()
    schema, defs = type_utils.describe_json_schema(empty_st)
    assert schema["$ref"]


def test_describe_json_schema_numpy_and_pandas():
    arr = np.array([1, 2, 3])
    schema, defs = type_utils.describe_json_schema(arr)
    assert schema["$ref"]

    scalar = np.int32(5)
    schema, defs = type_utils.describe_json_schema(scalar)
    assert schema["$ref"]

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    schema, defs = type_utils.describe_json_schema(df)
    assert schema["$ref"]


def test_describe_allowed_returns_schema():
    obj = {"a": 1}
    schema = type_utils.describe_allowed(obj)
    assert "$schema" in schema
    assert "definitions" in schema


def test_is_allowed_type_basic():
    assert type_utils.is_allowed_type(123)
    assert type_utils.is_allowed_type(12.3)
    assert type_utils.is_allowed_type(b"abc")
    assert type_utils.is_allowed_type("abc")
    assert type_utils.is_allowed_type(np.array([1]))
    assert type_utils.is_allowed_type(pd.DataFrame({"a": [1]}))
    assert type_utils.is_allowed_type(pd.Period("2023-01"))
    assert type_utils.is_allowed_type(None)


def test_is_allowed_type_mapping_and_sequence():
    d = {"a": 1, "b": 2}
    assert type_utils.is_allowed_type(d)
    lst = [1, 2, 3]
    assert type_utils.is_allowed_type(lst)
    tup = (1, 2, 3)
    assert type_utils.is_allowed_type(tup)
    # Not allowed: set
    assert not type_utils.is_allowed_type({1, 2, 3})
    # Not allowed: object
    assert not type_utils.is_allowed_type(object())


def test_serialize_and_deserialize_typehint_roundtrip():
    t = int
    s = type_utils.serialize_typehint(t)
    t2 = type_utils.deserialize_typehint(s)
    assert t == t2

    s_tv = "TypeVar(TextOrint)"
    # Because TextOrint is a TypeVar from the code, we test deserialize with known_types
    with pytest.raises(ValueError):
        type_utils.deserialize_typehint("TypeVar(NonExistent)")


def test_serialize_typehint_with_generic():
    from typing import List

    s = type_utils.serialize_typehint(List[int])
    assert "List" in s or "list" in s


def test_get_known_types_contains_expected_keys():
    known = type_utils.get_known_types()
    assert "int" in known
    assert "np" in known
    assert "pd" in known
    assert "torch" in known


def test__resolve_forwardrefs_basic():
    T = typing.ForwardRef("int")
    result = type_utils._resolve_forwardrefs(T, {"int": int})
    assert result == int

    with pytest.raises(ValueError):
        type_utils._resolve_forwardrefs(typing.ForwardRef("nonexistent"), {})


def test_is_valid_output_special_cases():
    # sqlite3.Connection
    conn = sqlite3.connect(":memory:")
    assert type_utils.is_valid_output(conn, sqlite3.Connection)

    # AllSimParams
    valid_all_sim_params = [{"param1": 1}, {"param2": "a"}]
    assert type_utils.is_valid_output(valid_all_sim_params, type_utils.AllSimParams)

    invalid_all_sim_params = [{"param1": 1}, [1, 2]]
    assert not type_utils.is_valid_output(
        invalid_all_sim_params, type_utils.AllSimParams
    )

    # SimValues
    valid_sim_values = {
        (("a", 1), ("b", 2)): 123,
    }
    assert type_utils.is_valid_output(valid_sim_values, type_utils.SimValues)

    invalid_sim_values = {
        ((1, 2),): 123,
    }
    assert not type_utils.is_valid_output(invalid_sim_values, type_utils.SimValues)


def test_is_valid_output_with_torch_module(monkeypatch):
    class DummyModule:
        pass

    import types

    class DummyTorchModule:
        pass

    # patch torch.nn.Module
    monkeypatch.setitem(
        type_utils.get_known_types(),
        "torch",
        types.SimpleNamespace(nn=types.SimpleNamespace(Module=DummyTorchModule)),
    )

    # patch to simulate torch.nn.Module
    monkeypatch.setattr(type_utils, "is_allowed_type", lambda obj: True)

    # forcibly test with_db=True and output_type torch.nn.Module
    dummy_module = DummyTorchModule()
    assert type_utils.is_valid_output(dummy_module, DummyTorchModule, with_db=True)


def test_is_valid_output_union_figure_and_dict(monkeypatch):
    # Union with Figure and Dict[str, Figure]
    from typing import Union, Dict

    union_type = Union[go.Figure, Dict[str, go.Figure]]

    fig = go.Figure()
    dict_fig = {"a": fig}
    not_valid = {1: fig}

    assert type_utils.is_valid_output(fig, union_type)
    assert type_utils.is_valid_output(dict_fig, union_type)
    assert not type_utils.is_valid_output(not_valid, union_type)


def test_deserialize_typehint_invalid_type(monkeypatch):
    with pytest.raises(TypeError):
        type_utils.deserialize_typehint(123)

    with pytest.raises(ValueError):
        type_utils.deserialize_typehint("NonExistentType")

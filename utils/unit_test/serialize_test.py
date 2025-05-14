import pytest
from utils.serialize_utils import serialize, deserialize
import numpy as np
import plotly.graph_objects as go
import json


def test_serialize_none():
    assert serialize(None) is None


def test_serialize_basic_types():
    assert serialize(42) == "42"
    assert serialize("Hello World") == '"Hello World"'
    assert serialize(3.14) == "3.14"


def test_serialize_list():
    assert serialize([1, 2, 3]) == "[1, 2, 3]"
    assert serialize(["a", "b", "c"]) == '["a", "b", "c"]'


def test_serialize_dict():
    assert serialize({"key": "value"}) == '{"key": "value"}'
    assert (
        serialize({"number": 1, "list": [1, 2, 3]})
        == '{"number": 1, "list": [1, 2, 3]}'
    )


def test_serialize_numpy_array():
    arr = np.array([1, 2, 3])
    serialized = serialize(arr)
    assert serialized


def test_deserialize_none():
    assert deserialize(None) is None


def test_deserialize_basic_types():
    encoded_str = serialize(42)
    assert deserialize(encoded_str) == 42
    encoded_str = serialize("Hello World")
    assert deserialize(encoded_str) == "Hello World"
    encoded_str = serialize(3.14)
    assert deserialize(encoded_str) == 3.14


def test_deserialize_list():
    encoded_str = serialize([1, 2, 3])
    assert deserialize(encoded_str) == [1, 2, 3]


def test_deserialize_dict():
    encoded_str = serialize({"key": "value"})
    assert deserialize(encoded_str) == {"key": "value"}


def test_deserialize_numpy_array():
    arr = np.array([1, 2, 3])
    serialized = serialize(arr)
    deserialized = deserialize(serialized)
    np.testing.assert_array_equal(deserialized, arr)


@pytest.mark.parametrize(
    "obj",
    [
        np.array([1, 2, 3]),  # NumPy array
        go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])]),  # Plotly Figure
    ],
)
def test_serialize_deserialize(obj):
    """Test that objects can be serialized and deserialized without data loss."""
    encoded = serialize(obj)
    decoded = deserialize(encoded)

    # Check types match
    assert type(decoded) is type(obj), f"Type mismatch: {type(decoded)} != {type(obj)}"

    # Check values match
    if isinstance(obj, np.ndarray):
        np.testing.assert_array_equal(decoded, obj)
    elif isinstance(obj, go.Figure):
        # Remove templates from both figures
        obj.layout.template = None
        decoded.layout.template = None

        # Compare JSON representations in a stable way
        obj_json = json.loads(obj.to_json())  # Convert to dict
        decoded_json = json.loads(decoded.to_json())  # Convert to dict
        assert obj_json == decoded_json, "Plotly Figure mismatch"
    else:
        assert decoded == obj

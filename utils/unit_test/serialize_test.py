import pytest
from utils.serialize_utils import (
    serialize_value,
    deserialize_value,
    encode_obj,
    decode_obj,
)
from utils.type_utils import get_known_types, Allowed

import numpy as np
import plotly.graph_objects as go
import json
import torch
import logging


def test_torch_model_serialization():
    # Define a model class dynamically (simulating what you're storing)
    class_def = """
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)
"""

    known_types = get_known_types()

    # Dynamically build the model
    namespace = {}
    exec(class_def, known_types, namespace)
    model_class = namespace["Model"]
    model = model_class()

    # Run forward pass and get output before serialization
    input_tensor = torch.tensor([[1.0, 2.0]])
    original_output = model(input_tensor).detach().numpy()

    # Encode the model
    model_data = {
        "class_def": class_def,
        "model": model,
    }
    encoded = encode_obj(model_data)

    # Decode the model
    decoded_model = decode_obj(encoded)["model"]

    # Run forward pass and get output after deserialization
    decoded_output = decoded_model(input_tensor).detach().numpy()

    logging.warning("-" * 10)
    logging.warning(decoded_output)
    logging.warning("-" * 10)
    # Assert the outputs are nearly equal
    assert decoded_output.shape == original_output.shape
    assert decoded_output == pytest.approx(original_output, rel=1e-5)


def test_serialize_none():
    assert serialize_value(None, value_type=Allowed) is None


def test_serialize_basic_types():
    assert serialize_value(42, value_type=Allowed) == "42"
    assert serialize_value("Hello World", value_type=Allowed) == '"Hello World"'
    assert serialize_value(3.14, value_type=Allowed) == "3.14"


def test_serialize_list():
    assert serialize_value([1, 2, 3], value_type=Allowed) == "[1, 2, 3]"
    assert serialize_value(["a", "b", "c"], value_type=Allowed) == '["a", "b", "c"]'


def test_serialize_dict():
    assert serialize_value({"key": "value"}, value_type=Allowed) == '{"key": "value"}'
    assert (
        serialize_value({"number": 1, "list": [1, 2, 3]}, value_type=Allowed)
        == '{"number": 1, "list": [1, 2, 3]}'
    )


def test_serialize_numpy_array():
    arr = np.array([1, 2, 3])
    serialized = serialize_value(arr, value_type=Allowed)
    assert serialized


def test_deserialize_none():
    assert deserialize_value(None, value_type=Allowed) is None


def test_deserialize_basic_types():
    encoded_str = serialize_value(42)
    assert deserialize_value(encoded_str, value_type=Allowed) == 42
    encoded_str = serialize_value("Hello World", value_type=Allowed)
    assert deserialize_value(encoded_str, value_type=Allowed) == "Hello World"
    encoded_str = serialize_value(3.14, value_type=Allowed)
    assert deserialize_value(encoded_str, value_type=Allowed) == 3.14


def test_deserialize_list():
    encoded_str = serialize_value([1, 2, 3], value_type=Allowed)
    assert deserialize_value(encoded_str, value_type=Allowed) == [1, 2, 3]


def test_deserialize_dict():
    encoded_str = serialize_value({"key": "value"}, value_type=Allowed)
    assert deserialize_value(encoded_str, value_type=Allowed) == {"key": "value"}


def test_deserialize_numpy_array():
    arr = np.array([1, 2, 3])
    serialized = serialize_value(arr, value_type=Allowed)
    deserialized = deserialize_value(serialized, value_type=Allowed)
    np.testing.assert_array_equal(deserialized, arr)


@pytest.mark.parametrize(
    "obj",
    [
        np.array([1, 2, 3]),  # NumPy array
        go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])]),  # Plotly Figure
    ],
)
def test_serialize_deserialize_value(obj):
    """Test that objects can be serialized and deserialized without data loss."""
    encoded = serialize_value(obj, value_type=Allowed)
    decoded = deserialize_value(encoded, value_type=Allowed)

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

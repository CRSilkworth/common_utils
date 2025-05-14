import pytest
import numpy as np
from utils.numpy_utils import dtype_to_string  # Replace with the actual module name


def test_dtype_to_string():
    # Test cases for various numpy dtypes
    test_cases = {
        np.dtype("int"): "int",
        np.dtype("int8"): "int8",
        np.dtype("int16"): "int16",
        np.dtype("int32"): "int32",
        np.dtype("int64"): "int64",
        np.dtype("uint8"): "uint8",
        np.dtype("uint16"): "uint16",
        np.dtype("uint32"): "uint32",
        np.dtype("uint64"): "uint64",
        np.dtype("float"): "float",
        np.dtype("float16"): "float16",
        np.dtype("float32"): "float32",
        np.dtype("float64"): "float64",
        np.dtype("complex"): "complex",
        np.dtype("complex64"): "complex64",
        np.dtype("complex128"): "complex128",
        np.dtype("bool"): "bool",
        np.dtype("object"): "object",
    }

    for dtype, expected_str in test_cases.items():
        assert dtype_to_string(dtype) == expected_str


if __name__ == "__main__":
    pytest.main()

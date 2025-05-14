import numpy as np


# Define a dictionary mapping dtype names to numpy dtypes
allowed_dtypes = {
    "int": np.dtype("int"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "float": np.dtype("float"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "complex": np.dtype("complex"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
    "bool": np.dtype("bool"),
    "object": np.dtype("object"),
}


def dtype_to_string(dtype: np.dtype) -> str:
    """
    Convert a numpy dtype to its corresponding string representation.

    Args:
        dtype (np.dtype): The numpy dtype to convert.

    Returns:
        str: The string representation of the numpy dtype.
            - "unicode_" for unicode types.
            - "string_" for string types.
            - "object" for object types.
            - The original dtype string for all other types.
    """
    dtype_str = str(dtype)
    dtype_str = dtype_str.replace("<", "")

    # Determine the string representation based on the dtype prefix
    if dtype_str.lstrip("|").startswith("U"):
        return "unicode_"
    elif dtype_str.lstrip("|").startswith("S"):
        return "string_"
    elif dtype_str.lstrip("|").startswith("O"):
        return "object"

    return dtype_str

from typing import Optional, Text, Any
import re
from lxml import etree
import os
import numpy as np
import pandas as pd
import json


def edit_distance(str1: Optional[str], str2: Optional[str]) -> float:
    """
    Calculate the normalized edit distance between two strings.

    The edit distance is the minimum number of operations required to transform
    one string into the other. The normalization is done by dividing the edit
    distance by the sum of the lengths of the two strings.

    Args:
        str1 (Optional[str]): The first string to compare. If None, it's treated as an
            empty string.
        str2 (Optional[str]): The second string to compare. If None, it's treated as an
            empty string.

    Returns:
        float: The normalized edit distance between the two strings.
    """
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    m = len(str1)
    n = len(str2)

    if not m and not n:
        return 0.0

    # Initialize the DP table with zeros
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill the DP table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Min. operations = j (insertions)
            elif j == 0:
                dp[i][j] = i  # Min. operations = i (deletions)
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change needed
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j],  # Remove
                    dp[i - 1][j - 1],  # Replace
                )

    return dp[m][n] / (n + m)


def longest_substring_overlap(
    query_string: Optional[str], test_string: Optional[str]
) -> int:
    """
    Find the length of the longest substring of `query_string` that is also a substring
    of `test_string`.

    Args:
        query_string (Optional[str]): The string to search for substrings. If None,
            it's treated as an empty string.
        test_string (Optional[str]): The string to check against. If None, it's treated
            as an empty string.

    Returns:
        int: The length of the longest overlapping substring.
    """
    query_string = "" if query_string is None else query_string
    test_string = "" if test_string is None else test_string

    max_overlap = 0
    len1 = len(query_string)

    # Check all possible substrings of query_string
    for i in range(len1):
        for j in range(i + 1, len1 + 1):
            substring = query_string[i:j]

            if len(substring) <= max_overlap:
                continue

            if substring in test_string:
                max_overlap = len(substring)
            else:
                break

    return max_overlap


def remove_indent(text: str) -> str:
    """
    Removes the common leading indentation from all lines in the given text.

    Parameters:
    text (str): The input string with possible leading indentation.

    Returns:
    str: The text with the common indentation removed.
    """
    # Split the text into lines
    lines = text.splitlines()

    # Find the common leading whitespace for all non-empty lines
    common_indent = None
    for line in lines:
        stripped = line.lstrip()
        if stripped:  # Only consider non-empty lines
            leading_spaces = len(line) - len(stripped)
            if common_indent is None:
                common_indent = leading_spaces
            else:
                common_indent = min(common_indent, leading_spaces)

    # If all lines are empty or there's no common indent, return the original text
    if common_indent is None or common_indent == 0:
        return text

    # Remove the common leading indentation
    return "\n".join(line[common_indent:] if line.lstrip() else "" for line in lines)


def remove_imports(text: str, only_top_level: bool = False) -> str:
    """
    Removes lines starting with 'import ' from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with lines starting with 'import ' removed.
    """
    if only_top_level:
        no_imports = "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith("import ") and not line.startswith("from ")
        )

        no_imports = no_imports.strip()
    else:
        no_imports = "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith("import ")
            and not line.strip().startswith("from ")
        )

        no_imports = no_imports.strip()
    return no_imports


def remove_outer_def(body_string):
    """
    Extracts and returns the body of a function that returns another function.

    Args:
        code (str): The Python code as a string.

    Returns:
        str: The extracted body of the inner function, or the original code if no match
        is found.
    """
    # Match the structure of a function definition that returns another function
    pattern = r"""
    ^def\s+\w+\(\):\s*
    ([\s\S]+?)\s*
    return\s+(\w+)$
    """
    match = re.search(pattern, body_string, re.VERBOSE | re.MULTILINE)

    if match:
        # Extract the captured inner function body
        outer_body = match.group(1)
        inner_function_name = match.group(2)

        # Look for the inner function definition within the outer body
        inner_function_pattern = rf"""
        ^def\s+{inner_function_name}\(\):\s*
        ([\s\S]+?)$
        """
        inner_match = re.search(
            inner_function_pattern, outer_body, re.VERBOSE | re.MULTILINE
        )
        if inner_match:
            return inner_match.group(1).strip()

    # Return the original code if no transformation is needed
    return body_string.strip()


def xml_to_html(
    xml_string,
    xslt_file: Text = "assets/xml_to_html/ILCD2XLS-0.1.1a/"
    "de/fzk/iai/lca/ilcd/stylesheets/process2html.xsl",
):
    # NOTE: This runs but doesn't really seem to work when rendered.
    xslt_file = os.path.abspath(xslt_file)

    if not os.path.exists(xslt_file):
        raise FileNotFoundError(f"XSLT file not found: {xslt_file}")

    base_path = "/".join(xslt_file.split("/")[:-1])
    # Parse the XML string
    xml_bytes = xml_string.encode("utf-8")
    xml_tree = etree.fromstring(xml_bytes)

    class CustomResolver(etree.Resolver):
        def resolve(self, url, id, context):
            abs_path = os.path.join(base_path, url)
            if os.path.exists(abs_path):
                return self.resolve_filename(abs_path, context)

            return self.resolve_string(b"<dummy></dummy>", context)

    recovering_parser = etree.XMLParser(recover=True)
    recovering_parser.resolvers.add(CustomResolver())
    xslt_tree = etree.parse(xslt_file, parser=recovering_parser)

    # Create an XSLT transformer
    transform = etree.XSLT(xslt_tree)

    # Perform the transformation
    html_tree = transform(xml_tree)
    html_string = etree.tostring(html_tree, pretty_print=True)
    return html_string.decode("utf-8")


def json_converter(obj: object) -> object:
    """Convert NumPy arrays, pandas objects, and other non-serializable
    types to JSON-serializable formats.

    Args:
        obj (object): The object to convert.

    Returns:
        object: A JSON-serializable version of the object.

    Raises:
        TypeError: If the object is not JSON serializable.
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        # Convert NumPy arrays to lists
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dictionary
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        # Convert Series to dictionary
        return {str(k): json_converter(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        # Convert timestamps and timedeltas to string
        return str(obj)
    elif isinstance(obj, (tuple, list)):
        # Recursively convert items in tuples and lists
        return [json_converter(o) for o in obj]
    elif isinstance(obj, dict):
        # Recursively convert dictionary keys and values
        return {str(k): json_converter(v) for k, v in obj.items()}
    elif type(obj) is bytes:
        # Convert bytes to string
        return str(obj)
    elif pd.isna(obj):
        # Convert pandas NA values to None
        return None
    elif isinstance(obj, (str, int, complex, float, bool)):
        return str(obj)
    else:
        return str(obj)
    # raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def data_to_readable_string(data: Any, indent: int = 2) -> str:
    """Convert nested dictionary (with NumPy arrays) to a human-readable string.

    Args:
        data (dict): The data to convert.
        indent (int, optional): The indentation level for pretty printing.
            Defaults to 2.

    Returns:
        str: A JSON string representation of the dictionary.
    """
    # Convert keys to strings and use numpy_converter for non-serializable objects
    return json.dumps(
        convert_keys_to_strings(data), indent=indent, default=json_converter
    )


def convert_keys_to_strings(obj: object) -> object:
    """Recursively convert dictionary keys to strings.

    Args:
        obj (object): The object to process.

    Returns:
        object: The processed object with all dictionary keys as strings.
    """
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_strings(i) for i in obj]
    else:
        return obj

from typing import Optional, Text
import re
import ast
from lxml import etree
import os
from exceptions.parse_exceptions import ParseException
import traceback
import asttokens


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
    Removes the common leading indentation (spaces or tabs) from all non-blank lines in
    a multiline string.Preserves relative indentation.
    """
    lines = text.splitlines()

    # Filter out blank lines
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return text

    # Extract leading whitespace from each non-empty line
    indents = [re.match(r"^[ \t]*", line).group(0) for line in non_empty_lines]

    # Find common prefix of all leading indents
    common_indent = indents[0]
    for indent in indents[1:]:
        # Reduce to the common prefix character-by-character
        i = 0
        while (
            i < len(common_indent) and i < len(indent) and common_indent[i] == indent[i]
        ):
            i += 1
        common_indent = common_indent[:i]

    # Remove the common indent
    dedented_lines = [
        (
            line[len(common_indent) :]  # noqa: E203
            if line.startswith(common_indent)
            else line
        )
        for line in lines
    ]

    return "\n".join(dedented_lines)


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


def extract_function_body(function_string: str) -> str:
    """
    Extracts the body of the function from the given function string.

    Args:
        function_string (str): The function string from which to extract the body.

    Returns:
        str: The extracted function body.
    """

    try:
        atok = asttokens.ASTTokens(function_string, parse=True)
        func_node = atok.tree.body[0]
    except Exception as e:
        print(traceback.format_exc())
        raise ParseException(f"Failed to parse function string: {e}")

    if not isinstance(func_node, ast.FunctionDef):
        raise ParseException("Provided string is not a function definition.")

    # Get the first and last statement in the body
    first_stmt = func_node.body[0]
    last_stmt = func_node.body[-1]

    # Use asttokens to get exact text between those
    start_pos = atok.get_text_range(first_stmt)[0]
    end_pos = atok.get_text_range(last_stmt)[1]

    # Return the body substring (preserving all formatting)
    return function_string[start_pos:end_pos]


def extract_class_def_body(class_string: str) -> str:
    """
    Extracts the body of the class from the given class string.

    Args:
        class_string (str): The class string from which to extract the body.

    Returns:
        str: The extracted class body.
    """
    try:
        # Parse the class string into an AST
        tree = ast.parse(class_string)
    except SyntaxError as e:
        print(traceback.format_exc())
        raise ParseException("Failed to parse class string: {}".format(e.args[0]))

    # Extract the class node
    class_node = tree.body[0]

    # Ensure that it is a class definition
    if isinstance(class_node, ast.ClassDef):
        # Convert the class body into a string
        try:
            body = "\n".join(ast.unparse(stmt) for stmt in class_node.body)
        except Exception as e:
            print(traceback.format_exc())
            raise ParseException("Failed to extract class body: {}".format(e.args[0]))
        return body
    else:
        return ""


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

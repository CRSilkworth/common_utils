from typing import Text, Dict, Any, Callable
from utils.misc_utils import failed_output
import re


def create_function(
    function_name: Text, function_string: Text, allowed_modules: Dict[Text, Any]
) -> Callable:
    """
    Compile and return the function specified in `function_name`.

    Returns:
        Callable: The function specified in `function_name`.

    Raises:
        ValueError: If the function cannot be created or is not callable.
    """
    output = {}
    function = None

    if not function_name:
        output = failed_output("No function defined")
        return function, output

    try:
        body = extract_function_body(function_string=function_string)
        if not body or body.strip() == "pass":
            output = failed_output("")
            return None, output

        bytecode = compile(function_string, filename="<inline code>", mode="exec")
        exec_result = {}
        exec(bytecode, allowed_modules, exec_result)

        function = exec_result.get(function_name)
        if function is None or not callable(function):
            raise ValueError(f"Failed to create python function:\n{function_string}")
    except (ValueError, SyntaxError) as e:
        output = failed_output(
            f"A  error occurred when trying to deserialize the function: {str(e)}"
        )

    return function, output


def extract_function_body(function_string: str) -> str:
    """
    Extracts the body of the function from the given function string.

    Args:
        function_string (str): The function string from which to extract the body.

    Returns:
        str: The extracted function body.
    """
    # Match the function definition line and capture the body
    match = re.search(r"def\s+\w+\(.*\):\n((?:\n|.)*)", function_string)
    if match:
        function_body = match.group(1)
        # Strip the leading indentation (assuming it is uniformly indented)
        lines = function_body.split("\n")
        stripped_lines = []
        for line in lines:
            if line.startswith("\t"):
                line = line[1:]
            elif line.startswith("    "):
                line = line[4:]
            stripped_lines.append(line)

        return "\n".join(stripped_lines).strip()
    return ""

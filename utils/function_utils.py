from typing import Text, Dict, Any, Callable, Optional, Tuple, Union, Generator
from utils.misc_utils import failed_output
from utils.string_utils import remove_indent
import re
import traceback
import io
import contextlib
from utils.type_utils import is_valid_output, chunked_type_map
import logging
from utils.string_utils import remove_imports
import inspect


def create_function(
    function_name: Text,
    function_string: Text,
    allowed_modules: Dict[Text, Any],
    function_header: Text = "",
    header_code: Text = "",
    global_vars: Optional[Dict[Text, Any]] = None,
) -> Callable:
    """
    Compile and return the function specified in `function_name`.

    Returns:
        Callable: The function specified in `function_name`.

    Raises:
        ValueError: If the function cannot be created or is not callable.
    """
    if not global_vars:
        global_vars = {}

    output = {
        "failed": False,
        "value": None,
        "combined_output": "",
        "stdout_output": "",
        "stderr_output": "",
    }
    function = None

    if not function_name:
        return function, output

    try:
        body = extract_function_body(function_string=function_string)
        if not body.strip() or body.strip() == "pass":
            return None, output

        if header_code:
            function_string = (
                function_header
                + "\n\t"
                + header_code.replace("\n", "\n\t")
                + "\n\t"
                + body.replace("\n", "\n\t")
            )

        bytecode = compile(function_string, filename="<inline code>", mode="exec")
        exec_result = {}

        for key, value in global_vars.items():
            allowed_modules[key] = value
        exec(bytecode, allowed_modules, exec_result)

        function = exec_result.get(function_name)
        if function is None or not callable(function):
            raise ValueError(f"Failed to create python function:\n{function_string}")
    except (ValueError, SyntaxError) as e:
        logging.info(traceback.format_exc())
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
    match = re.search(r"def\s+\w+\(.*?\):\n((?:\s+.+\n?)*)", function_string, re.DOTALL)

    if match:
        function_body = match.group(1)
        function_body = remove_indent(function_body)
        # Strip the leading indentation (assuming it is uniformly indented)
        lines = function_body.split("\n")

        return "\n".join(lines).strip()
    return ""


def extract_class_def_body(class_def: str) -> str:
    """
    Extracts the body of the class_def from the given class_def string.

    Args:
        class_def_string (str): The class_def string from which to extract the body.

    Returns:
        str: The extracted class_def body.
    """
    # Match the class_def definition line and capture the body
    match = re.search(r"class\s+\w+\(.*\):\n((?:\n|.)*)", class_def)
    if match:
        class_def_body = match.group(1)
        class_def_body = remove_indent(class_def_body)
        # Strip the leading indentation (assuming it is uniformly indented)
        lines = class_def_body.split("\n")
        stripped_lines = []
        for line in lines:
            if line.startswith("\t"):
                line = line[1:]
            elif line.startswith("    "):
                line = line[4:]
            stripped_lines.append(line)

        return "\n".join(stripped_lines).strip()
    return ""


def capture_output(func: Callable, *args: Any, **kwargs: Any) -> tuple:
    """
    Capture the output and errors of a function execution.

    Args:
        func (Callable): The function to be executed.
        *args (Any): Positional arguments for the function.
        **kwargs (Any): Keyword arguments for the function.

    Returns:
        tuple: A tuple containing the function result, combined output, stdout, stderr,
            and a failure flag.
    """
    # Create pipes for stdout and stderr
    stdout_pipe = io.StringIO()
    stderr_pipe = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_pipe), contextlib.redirect_stderr(
            stderr_pipe
        ):
            result = func(*args, **kwargs)
        failed = False
    except Exception:
        result = None
        failed = True
        # Capture the full traceback and write it to stderr_pipe
        stderr_pipe.write(traceback.format_exc())

    # Collect output and errors
    stdout_output = stdout_pipe.getvalue().strip()
    stderr_output = stderr_pipe.getvalue().strip()

    # Remove the non user define function traceback
    if failed:
        stderr_output = "\n".join(stderr_output.split("\n")[3:])

    # Combine output and errors
    combined_output = ""
    if stdout_output:
        combined_output += "[stdout] " + "\n[stdout] ".join(stdout_output.split("\n"))
    if stderr_output:
        combined_output += "[stderr] " + "\n[stderr] ".join(stderr_output.split("\n"))

    return result, combined_output, stdout_output, stderr_output, failed


def capture_output_generator(
    func: Callable, *args: Any, **kwargs: Any
) -> Tuple[Union[Any, Generator], str, str, str, bool]:
    """
    Capture the output and errors of a function execution.

    Supports both regular functions and generator functions.

    Returns:
        tuple: (result or generator wrapper, combined_output, stdout, stderr, failed)
    """
    stdout_pipe = io.StringIO()
    stderr_pipe = io.StringIO()
    failed = False

    def output() -> str:
        return stdout() + stderr()

    def stdout() -> str:
        return stdout_pipe.getvalue().strip()

    def stderr() -> str:
        return stderr_pipe.getvalue().strip()

    def failed_flag() -> bool:
        return failed

    def wrapper() -> Generator[Any, None, None]:
        nonlocal failed
        try:
            with contextlib.redirect_stdout(stdout_pipe), contextlib.redirect_stderr(
                stderr_pipe
            ):
                gen = func(*args, **kwargs)
                for item in gen:
                    yield item
        except Exception:
            failed = True
            stderr_pipe.write(traceback.format_exc())
            logging.info("stderr", traceback.format_exc())
            print(traceback.format_exc())

    return wrapper(), output, stdout, stderr, failed_flag


def run_with_expected_type(
    func: Callable,
    decoded_kwargs: Dict[Text, Any],
    output_type: Any,
):
    """
    Execute a function that returns and put returned value.

    Args:
        request (RunRequest): The request containing function details
                              and arguments.

    Returns:
        Response: A response with the function result and execution details.
    """

    value, combined_output, stdout_output, stderr_output, failed = capture_output(
        func=func, **decoded_kwargs
    )
    output = {
        "value": None,
        "combined_output": combined_output,
        "stdout_output": stdout_output,
        "stderr_output": stderr_output,
        "failed": failed,
    }
    if failed:
        pass
    elif not is_valid_output(value, output_type=output_type):
        new_error = (
            f"\nExpected output type of {output_type}. {value} is of type "
            f"{type(value).__name__}\n"
        )
        output["value"] = None
        output["failed"] = True
        output["stderr_output"] += new_error
        output["combined_output"] += new_error
    else:
        output["value"] = value

    return output


def run_with_generator(
    func: Callable, decoded_kwargs: Dict[Text, Any], output_type: Any
):
    """
    Execute a function that returns and put returned value.

    Args:
        request (RunRequest): The request containing function details
                              and arguments.

    Returns:
        Response: A response with the function result and execution details.
    """
    chunked_output_type = chunked_type_map[output_type]
    gen, combined, stdout, stderr, fail = capture_output_generator(
        func=func, **decoded_kwargs
    )
    for value_chunk in gen:
        output = {
            "value_chunk": None,
            "combined_output": combined(),
            "stdout_output": stdout(),
            "stderr_output": stderr(),
            "failed": fail(),
        }
        if output["failed"]:
            pass
        elif not is_valid_output(value_chunk, output_type=chunked_output_type):
            new_error = (
                f"\nExpected output type of {chunked_output_type}. {value_chunk} is of"
                f" type {type(value_chunk).__name__}\n"
            )
            output["value_chunk"] = None
            output["failed"] = True
            output["stderr_output"] += new_error
            output["combined_output"] += new_error
        else:
            output["value_chunk"] = value_chunk

        yield output


def body_from_function(function: Callable) -> Text:
    return extract_function_body(
        remove_imports(remove_indent(inspect.getsource(function)), only_top_level=True)
    )

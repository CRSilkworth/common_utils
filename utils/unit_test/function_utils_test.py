from utils import function_utils


def test_extract_function_body():
    func_str = """def foo():\n    return 1\n"""
    body = function_utils.extract_function_body(func_str)
    assert body == "return 1"

    func_str_empty = "def foo():\n    pass\n"
    body_empty = function_utils.extract_function_body(func_str_empty)
    assert body_empty == "pass"

    func_str_no_body = "def foo():\n"
    body_no = function_utils.extract_function_body(func_str_no_body)
    assert body_no == ""


def test_extract_class_def_body():
    class_str = """class Foo(object):\n    def bar(self):\n        return 1\n"""
    body = function_utils.extract_class_def_body(class_str)
    assert "def bar(self):" in body
    assert "return 1" in body

    class_str_empty = "class Foo():\n    pass\n"
    body_empty = function_utils.extract_class_def_body(class_str_empty)
    assert body_empty == "pass"

    class_str_no_body = "class Foo():\n"
    body_no = function_utils.extract_class_def_body(class_str_no_body)
    assert body_no == ""


def test_create_function_success():
    func_str = "def test_func():\n    return 42"
    func, output = function_utils.create_function(
        "test_func", func_str, allowed_modules={}, function_header="", header_code=""
    )
    assert callable(func)
    assert output["failed"] is False
    assert func() == 42


def test_create_function_empty_body():
    func_str = "def test_func():\n    pass"
    func, output = function_utils.create_function(
        "test_func", func_str, allowed_modules={}, function_header="", header_code=""
    )
    assert func is None
    assert output["failed"] is False


def test_create_function_invalid_syntax():
    func_str = "def test_func(:\n    return 42"
    func, output = function_utils.create_function(
        "test_func", func_str, allowed_modules={}, function_header="", header_code=""
    )
    assert func is None


def test_create_function_not_callable():
    # Function string does not define the function_name as a callable
    func_str = "def test_func():\n    return 42"
    func, output = function_utils.create_function(
        "non_existent_func",
        func_str,
        allowed_modules={},
        function_header="",
        header_code="",
    )
    assert func is None
    assert output["failed"] is True


def test_capture_output_success():
    def test_func(x):
        print("hello")
        return x

    result, combined, stdout, stderr, failed = function_utils.capture_output(
        test_func, 5
    )
    assert result == 5
    assert "hello" in stdout
    assert not failed
    assert stderr == ""
    assert "[stdout] hello" in combined


def test_capture_output_exception():
    def test_func():
        raise ValueError("error")

    result, combined, stdout, stderr, failed = function_utils.capture_output(test_func)
    assert result is None
    assert failed
    assert "ValueError" in stderr
    assert "[stderr] " in combined


def test_run_with_expected_type_success(monkeypatch):
    def test_func():
        return 123

    # Patch is_valid_output to True
    monkeypatch.setattr(
        function_utils, "is_valid_output", lambda value, output_type, with_db: True
    )
    output = function_utils.run_with_expected_type(test_func, {}, int)
    assert output["value"] == 123
    assert not output["failed"]


def test_run_with_expected_type_failure_type(monkeypatch):
    def test_func():
        return "string"

    monkeypatch.setattr(
        function_utils, "is_valid_output", lambda value, output_type, with_db: False
    )
    output = function_utils.run_with_expected_type(test_func, {}, int)
    assert output["value"] is None
    assert output["failed"]
    assert "Expected output type" in output["stderr_output"]


def test_run_with_expected_type_failed_execution(monkeypatch):
    def test_func():
        raise Exception("fail")

    output = function_utils.run_with_expected_type(test_func, {}, int)
    assert output["value"] is None
    assert output["failed"]

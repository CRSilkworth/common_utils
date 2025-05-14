import pytest
from utils.function_utils import (
    function_defaults,
    has_arguments,
)


def test_function_defaults_no_defaults():
    def func(a, b):
        pass

    result = function_defaults(func)
    assert result == {}


def test_function_defaults_with_defaults():
    def func(a, b=2, c="default"):
        pass

    expected_defaults = {"b": 2, "c": "default"}
    result = function_defaults(func)
    assert result == expected_defaults


def test_function_defaults_only_some_defaults():
    def func(a, b=2, c="default", d=4):
        pass

    expected_defaults = {"b": 2, "c": "default", "d": 4}
    result = function_defaults(func)
    assert result == expected_defaults


def test_function_defaults_no_args():
    def func():
        pass

    result = function_defaults(func)
    assert result == {}


def test_has_arguments_with_args():
    def func(a, b, *args):
        pass

    result = has_arguments(func)
    assert result is True


def test_has_arguments_without_args():
    def func(a, b):
        pass

    result = has_arguments(func)
    assert result is False


def test_has_arguments_with_kwargs():
    def func(a, b, **kwargs):
        pass

    result = has_arguments(func)
    assert result is False


def test_has_arguments_mixed_args_and_kwargs():
    def func(a, b, *args, **kwargs):
        pass

    result = has_arguments(func)
    assert result is True


if __name__ == "__main__":
    pytest.main()

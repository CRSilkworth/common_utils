import pytest
from exceptions.llm import catch_llm_exceptions
from exceptions.input_exceptions import InputException
from exceptions.quota_exceptions import QuotaException
from exceptions.parse_exceptions import ParseException
import mongoengine as me


def test_catch_input_exception():
    with catch_llm_exceptions() as error_dict:
        raise InputException("Input error occurred")
    assert error_dict["failed"] is True
    assert error_dict["error"] == "('Input error occurred', 'Parsing input failed.')"


def test_catch_validation_error():
    with catch_llm_exceptions() as error_dict:
        raise me.ValidationError("Validation error occurred")
    assert error_dict["failed"] is True
    assert error_dict["error"] == "Validation error occurred"


def test_catch_quota_exception():
    with catch_llm_exceptions() as error_dict:
        raise QuotaException("Quota exceeded")
    assert error_dict["failed"] is True
    assert error_dict["error"] == "Quota exceeded"


def test_catch_parse_exception():
    with catch_llm_exceptions() as error_dict:
        raise ParseException("Parse error occurred")
    assert error_dict["failed"] is True
    assert error_dict["error"] == "Parse error occurred"


def test_catch_general_exception():
    with catch_llm_exceptions() as error_dict:
        raise Exception("Some other error")
    assert error_dict["failed"] is True
    assert error_dict["error"] == "There was an error when attempting to use LLM output"

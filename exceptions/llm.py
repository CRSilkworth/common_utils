from contextlib import contextmanager
import traceback
from exceptions.input_exceptions import InputException
from exceptions.quota_exceptions import QuotaException
from exceptions.parse_exceptions import ParseException
import mongoengine as me
import logging


@contextmanager
def catch_llm_exceptions():
    error_dict = {"failed": False, "error": ""}
    try:
        yield error_dict  # Yield a reference to error_dict
    except InputException as e:
        print(traceback.format_exc())
        error_dict["failed"] = True
        error_dict["error"] = str(e)
    except me.ValidationError as e:
        print(traceback.format_exc())
        error_dict["failed"] = True
        error_dict["error"] = str(e)
    except QuotaException as e:
        print(traceback.format_exc())
        error_dict["failed"] = True
        error_dict["error"] = e.args[0]
    except ParseException as e:
        logging.info("LLM parse failure: {}".format(e.args[0]))
        print(traceback.format_exc())
        error_dict["failed"] = True
        error_dict["error"] = e.args[0]
    except Exception as e:
        logging.info("Save exception: {}".format(e))
        print(traceback.format_exc())
        error_dict["failed"] = True
        error_dict["error"] = "There was an error when attempting to use LLM output"

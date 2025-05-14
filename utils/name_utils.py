import re
from typing import Text, List, Iterable
from exceptions.input_exceptions import InputException
from typeguard import typechecked


@typechecked
def is_valid_component_name(name: Text) -> bool:
    """
    Check whether a name contains only letters, numbers, and underscores.

    Args:
        name (Text): The name to check.

    Returns:
        bool: True if the name conforms to the restrictions, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, name))


@typechecked
def clean_name(name):
    """
    Remove any characters not in unicode characters, letters, numbers, and underscores.

    Args:
        name (Text): The name to check.

    Returns:
        bool: True if the name conforms to the restrictions, False otherwise.
    """
    # Remove any character that is not a Unicode letter, digit, or underscore
    cleaned_name = name.replace(" ", "_")
    cleaned_name = re.sub(r"[^\w\u00A0-\uFFFF]", "", name)
    if not cleaned_name:
        return "_"

    # Ensure the first character is a Unicode letter or underscore
    if cleaned_name and cleaned_name[0].isdigit():
        cleaned_name = "_" + cleaned_name

    return cleaned_name


@typechecked
def is_valid_name(name: Text) -> bool:
    """
    Check whether a name quote-like characters or backslashes or periods.

    Args:
        name (Text): The name to check.

    Returns:
        bool: True if the name conforms to the restrictions, False otherwise.
    """
    return not bool(re.search(r'[\'"\\`“”‘’´.]', name))


@typechecked
def strip_characters(name: str) -> str:
    """
    left/right strip any whitespace and quotelike characters from a name

    Args:
        name (Text): The name to check.

    Returns:
        bool: True if the name conforms to the restrictions, False otherwise.
    """
    return re.sub(r'^[\'"\\`“”‘’´.\s]+|[\'"\\`“”‘’´.\s]+$', "", name)


@typechecked
def clean_full_names(full_names: Iterable[Text]) -> List[Text]:
    cleaned_full_names = []
    for full_name in full_names:
        names = []
        for name in full_name.split("."):
            if not is_valid_name(name):
                raise InputException(
                    "Node names cannot contain, periods, quotes or escape \n"
                    "characters. Got: {}".format(full_name)
                )
            names.append(name.strip())
        cleaned_full_names.append(".".join(names))
    return cleaned_full_names

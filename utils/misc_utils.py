from typing import Text, List, Dict, Any, Literal, Optional, Tuple, Union
from typeguard import typechecked
from utils.type_utils import ElementType


def failed_output(message: Text):
    return {
        "failed": True,
        "value": None,
        "combined_output": message,
        "stdout_output": message,
        "stderr_output": message,
    }


@typechecked
def get_elements_diff(
    new: Optional[List[ElementType]],
    old: Optional[List[ElementType]],
    diff_type: Literal["both", "new", "old"] = "both",
) -> List[Dict[Text, Any]]:
    """
    Compare two lists of elements and return their differences.

    Args:
        new (Optional[List[ElementType]]): The new list of elements.
        old (Optional[List[ElementType]]): The old list of elements.
        diff_type (Literal["both", "new", "old"], optional): The type of differences to
            return.
            - "new": Returns elements in `new` that are not in `old`.
            - "old": Returns elements in `old` that are not in `new`.
            - "both": Returns elements that are different between `new` and `old`.
            Defaults to "both".

    Returns:
        List[Dict[Text, Any]]: A list of dictionaries representing the differences
            between the two lists.
    """
    new = new if new else []
    old = old if old else []

    new_set = set()
    for elem in new:
        elem_list: List[Tuple[Text, Union[Any, Tuple]]] = []
        for key, d in elem.items():
            if isinstance(d, dict):
                elem_list.append((key, tuple(d.items())))
            else:
                elem_list.append((key, d))
        new_set.add(tuple(elem_list))

    old_set = set()
    for elem in old:
        elem_list: List[Tuple[Text, Union[Any, Tuple]]] = []
        for key, d in elem.items():
            if isinstance(d, dict):
                elem_list.append((key, tuple(d.items())))
            else:
                elem_list.append((key, d))
        old_set.add(tuple(elem_list))

    diffs = set()
    if diff_type in ("new", "both"):
        diffs.update(new_set - old_set)

    if diff_type in ("old", "both"):
        diffs.update(old_set - new_set)

    differences: List[Dict[Text, Any]] = []
    for diff in sorted(diffs):
        diff_dict: Dict[Text, Any] = {}
        for key, tup in diff:
            if isinstance(tup, tuple):
                diff_dict[key] = dict(tup)
            else:
                diff_dict[key] = tup
        differences.append(diff_dict)

    return differences

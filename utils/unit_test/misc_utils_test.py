import pytest
from unittest.mock import MagicMock
from utils.misc_utils import get_elements_diff


# Define a mock MongoEngine document for testing
class MockDocument(MagicMock):
    pass


def test_get_elements_diff():
    # Test with both "new" and "old" lists
    new = [{"a": 1, "b": {"c": 2}}, {"d": 3}]
    old = [{"a": 1, "b": {"c": 2}}, {"e": 4}]

    result = get_elements_diff(new, old, "both")
    expected = [{"d": 3}, {"e": 4}]
    assert sorted(result, key=lambda a: a.keys()) == sorted(
        expected, key=lambda a: a.keys()
    )

    # Test with "new" only
    result = get_elements_diff(new, old, "new")
    expected = [{"d": 3}]
    assert result == expected

    # Test with "old" only
    result = get_elements_diff(new, old, "old")
    expected = [{"e": 4}]
    assert result == expected

    # Test with empty lists
    result = get_elements_diff([], [], "both")
    expected = []
    assert result == expected

    # Test with None values
    result = get_elements_diff(None, None, "both")
    expected = []
    assert result == expected

    # Test with None new list
    result = get_elements_diff(None, old, "both")
    expected = old
    assert result == expected

    # Test with None old list
    result = get_elements_diff(new, None, "both")
    expected = new
    assert result == expected


if __name__ == "__main__":
    pytest.main()

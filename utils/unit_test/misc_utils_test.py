import pytest
from utils.misc_utils import get_elements_diff, failed_output


def test_failed_output():
    message = "error occurred"
    output = failed_output(message)
    expected = {
        "failed": True,
        "value": None,
        "combined_output": message,
        "stdout_output": message,
        "stderr_output": message,
    }
    assert output == expected


def test_get_elements_diff():
    new = [{"a": 1, "b": {"c": 2}}, {"d": 3}]
    old = [{"a": 1, "b": {"c": 2}}, {"e": 4}]

    # Test diff_type both
    result = get_elements_diff(new, old, "both")
    expected = [{"d": 3}, {"e": 4}]
    assert sorted(result, key=lambda a: sorted(a.keys())) == sorted(
        expected, key=lambda a: sorted(a.keys())
    )

    # Test diff_type new
    result = get_elements_diff(new, old, "new")
    expected = [{"d": 3}]
    assert result == expected

    # Test diff_type old
    result = get_elements_diff(new, old, "old")
    expected = [{"e": 4}]
    assert result == expected

    # Test with empty lists
    result = get_elements_diff([], [], "both")
    expected = []
    assert result == expected

    # Test with None lists
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

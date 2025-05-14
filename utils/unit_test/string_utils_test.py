import pytest
from utils.string_utils import edit_distance, longest_substring_overlap


def test_edit_distance():
    """
    Tests for the `edit_distance` function.
    """
    # Test with normal strings
    assert edit_distance("kitten", "sitting") == 0.23076923076923078

    # Test with one empty string
    assert edit_distance("kitten", "") == 1.0
    assert edit_distance("", "sitting") == 1.0  # (7/7)

    # Test with both strings empty
    assert edit_distance("", "") == 0.0  # (0/0)

    # Test with both strings None
    assert edit_distance(None, None) == 0.0  # (0/0)

    # Test with one None and one non-empty string
    assert edit_distance(None, "test") == 1.0  # (4/4)

    # Test with two identical strings
    assert edit_distance("hello", "hello") == 0.0  # (0/10)

    # Test with different strings
    assert edit_distance("abcdef", "azced") == 3 / 11


def test_longest_substring_overlap():
    """
    Tests for the `longest_substring_overlap` function.
    """
    # Test with normal strings
    assert longest_substring_overlap("abc", "zabcy") == 3  # "abc"

    # Test with empty strings
    assert longest_substring_overlap("", "") == 0
    assert longest_substring_overlap("", "nonempty") == 0
    assert longest_substring_overlap("nonempty", "") == 0

    # Test with None values
    assert longest_substring_overlap(None, None) == 0
    assert longest_substring_overlap(None, "test") == 0
    assert longest_substring_overlap("test", None) == 0

    # Test with substrings present in one string
    assert longest_substring_overlap("abcd", "abcd") == 4  # "abcd"

    # Test with non-overlapping substrings
    assert longest_substring_overlap("abc", "xyz") == 0

    # Test with overlapping substrings of different lengths
    assert longest_substring_overlap("hello", "he") == 2  # "he"
    assert longest_substring_overlap("hello", "helloo") == 5  # "hell"


if __name__ == "__main__":
    pytest.main()

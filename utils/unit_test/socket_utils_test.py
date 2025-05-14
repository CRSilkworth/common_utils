import pytest
from typing import Dict, Text
from bson.json_util import dumps
from utils.socket_utils import (
    get_graph_id,
    get_id,
    get_operation,
    get_document,
    get,
)


@pytest.fixture
def mock_update() -> Dict[Text, Text]:
    """
    Fixture to provide a sample update dictionary for testing.
    """
    return {
        "data": dumps(
            {
                "fullDocument": {"calc_graph_ref": "graph123"},
                "documentKey": {"_id": "doc456"},
                "operationType": "update",
                "collection": "my_collection",
            }
        )
    }


def test_get_graph_id(mock_update):
    """
    Test for `get_graph_id` function.
    """
    result = get_graph_id(mock_update)
    assert result == "graph123"


def test_get_id(mock_update):
    """
    Test for `get_id` function.
    """
    result = get_id(mock_update)
    assert result == "doc456"


def test_get_operation(mock_update):
    """
    Test for `get_operation` function.
    """
    result = get_operation(mock_update)
    assert result == "update"


def test_get_document(mock_update):
    """
    Test for `get_document` function.
    """
    result = get_document(mock_update)
    expected_document = {"calc_graph_ref": "graph123"}
    assert result == expected_document


def test_get(mock_update):
    """
    Test for `get` function with valid key.
    """
    result = get(mock_update, "collection")
    assert result == "my_collection"


def test_get_invalid_key(mock_update):
    """
    Test for `get` function with invalid key.
    """
    with pytest.raises(KeyError):
        get(mock_update, "non_existent_key")


if __name__ == "__main__":
    pytest.main()

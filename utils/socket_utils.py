from typing import Dict, Any, Text
from bson.json_util import loads


def get_graph_id(update: Dict[Text, Text]) -> Text:
    """
    Extract the graph ID from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Text: The graph ID extracted from the update.
    """
    data = loads(update["data"])
    return str(data["fullDocument"]["calc_graph_ref"])


def get_id(update: Dict[Text, Text]) -> Text:
    """
    Extract the document ID from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Text: The document ID extracted from the update.
    """
    data = loads(update["data"])
    return str(data["documentKey"]["_id"])


def get_collection(update: Dict[Text, Text]) -> Text:
    """
    Extract the document collection from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Text: The document collection extracted from the update.
    """
    data = loads(update["data"])
    return str(data["collection"])


def get_operation(update: Dict[Text, Text]) -> Text:
    """
    Extract the operation type from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Text: The operation type extracted from the update.
    """
    data = loads(update["data"])
    return data["operationType"]


def get_document(update: Dict[Text, Text]) -> Dict[Text, Any]:
    """
    Extract the full document from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Dict[Text, Any]: The full document extracted from the update.
    """
    data = loads(update["data"])
    return data["fullDocument"]


def get_is_deleted(update: Dict[Text, Text]) -> Dict[Text, Any]:
    """
    Extract the is_deleted field from an update dictionary.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.

    Returns:
        Dict[Text, Any]: The is_deleted field extracted from the update.
    """
    data = loads(update["data"])
    return data["fullDocument"]["is_deleted"]


def get(update: Dict[Text, Text], key: Text) -> Any:
    """
    Extract a specific value from an update dictionary based on the provided key.

    Args:
        update (Dict[Text, Text]): A dictionary containing update data in JSON format.
        key (Text): The key of the value to extract from the update.

    Returns:
        Any: The value associated with the provided key.
    """
    data = loads(update["data"])
    return data[key]

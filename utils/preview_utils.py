from utils.type_utils import describe_allowed
from utils.string_utils import data_to_readable_string
import json


def value_to_preview(value, max_len: int = 500):
    preview = data_to_readable_string(value)

    if len(preview) > max_len:
        schema = describe_allowed(value)
        preview = (
            preview[: max_len // 2]
            + "\n...\n"
            + preview[-max_len // 2 :]  # noqa: E203
            + "\n\nValue too large to show. This is it's basic"
            f" format:\n\n"
            f"{json.dumps(schema, indent=2)}"
        )
    return preview

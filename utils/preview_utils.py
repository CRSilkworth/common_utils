from utils.type_utils import describe_allowed
from utils.string_utils import data_to_readable_string
import json


def value_to_preview(value):
    preview = data_to_readable_string(value)

    if len(preview) > 1000:
        schema = describe_allowed(value)
        preview = (
            preview[:100]
            + "\n...\n"
            + preview[-100:]
            + "\n\nValue too large to show. This is it's basic"
            f" format:\n\n"
            f"{json.dumps(schema, indent=2)}"
        )
    return preview

from typing import Text, Any, List, Dict
from utils.type_utils import Allowed
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.gcp_utils import (
    upload_via_signed_post,
    read_from_gcs_signed_urls,
    request_policy,
)
from utils.type_utils import deserialize_typehint
from utils.misc_utils import failed_output


def generator_from_urls(signed_urls: List[Text], value_type: Any = Allowed):
    value = read_from_gcs_signed_urls(signed_urls) if signed_urls else None

    for file_content in value:
        file_content, output, _ = attempt_deserialize(file_content, value_type)
        if output:
            raise ValueError(
                "failed to deserialize file from generator: "
                f"{output['stderr_output']}"
            )
        for item in file_content:
            yield item


def upload_serialized_value(
    _value, doc_id, attribute_name, version, auth_data: Dict[Text, Text]
):
    policy_data = {}
    policy_data.update(auth_data)
    policy_data.update(
        {
            "doc_id": doc_id,
            "attribute_name": attribute_name,
            "version": version,
            "chunk_file_num": 0,
        }
    )
    policy = request_policy(
        auth_data["dash_app_url"],
        policy_data,
        token=auth_data["token"],
    )
    status = upload_via_signed_post(policy, _value)
    if status not in (200, 204):
        return failed_output(f"Failed to upload file to gcs. Got status code {status}")

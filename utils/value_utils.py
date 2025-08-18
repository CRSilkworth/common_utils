from typing import Text, Dict
from utils.gcp_utils import upload_via_signed_post, request_policy
from utils.misc_utils import failed_output


def upload_serialized_value(
    _value, doc_id, attribute_name, version, auth_data: Dict[Text, Text]
):
    policy_data = {"auth_data": auth_data}
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

    return policy

from typing import Text, Dict, Any
from datetime import timedelta
import uuid
import json
import requests
from io import BytesIO


def generate_signed_url(bucket_name, blob_name, expiration_minutes):
    """Generate a signed URL for accessing a file in GCS."""
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Generate signed URL
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
    )

    return signed_url


def upload_via_signed_post(policy: dict, json_str: str, filename: str = "value.json"):
    files = {"file": (filename, BytesIO(json_str.encode("utf-8")), "application/json")}

    response = requests.post(policy["url"], data=policy["fields"], files=files)

    return response.status_code


def upload_json_to_gcs(d: Dict[Text, Any], user_id: Text, bucket_name: Text) -> str:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    unique_id = str(uuid.uuid4())
    blob_path = f"{user_id}/{unique_id}.py"
    blob = bucket.blob(blob_path)

    blob.upload_from_string(json.dumps(d, indent=2), content_type="text/plain")
    return blob_path


def read_from_gcs(gcs_path: str) -> dict:
    from google.cloud import storage

    assert gcs_path.startswith("gs://")
    bucket_name, blob_name = gcs_path[5:].split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()  # or download_as_text() for str

    return content


def read_from_gcs_signed_url(gcs_url: str) -> str:
    """
    Fetch content from a GCS-signed or public URL using plain HTTP.

    Args:
        gcs_url (str): A signed GCS URL or a public object URL like
                       'https://storage.googleapis.com/my-bucket/my-file.json'

    Returns:
        str: Content of the blob as a string.
    """
    response = requests.get(gcs_url)
    if response.status_code != 200:
        return None
    return response.text

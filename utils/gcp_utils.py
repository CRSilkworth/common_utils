from typing import Text, Dict, Any
from datetime import timedelta
import uuid
import json


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


def upload_json_to_gcs(d: Dict[Text, Any], user_id: Text, bucket_name: Text) -> str:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    unique_id = str(uuid.uuid4())
    blob_path = f"{user_id}/{unique_id}.py"
    blob = bucket.blob(blob_path)

    blob.upload_from_string(json.dumps(d, indent=4), content_type="text/plain")
    return blob_path

from typing import Optional, Dict, Text, Any
import requests
import os


def serialize_qb_result(obj):
    """Convert SDK objects into JSON-safe dicts recursively."""
    if hasattr(obj, "serialize"):
        return obj.serialize()
    elif isinstance(obj, list):
        return [serialize_qb_result(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: serialize_qb_result(v) for k, v in obj.items()}
    else:
        return obj


class QuickBooksProxy:
    def __init__(self, auth_data: Optional[Dict[Text, Any]] = None):
        self.auth_data = auth_data

    def query(self, q):
        """Proxy the query method to the server."""
        resp = requests.post(
            os.path.join(self.auth_data["dash_app_url"], "qb-proxy"),
            json={
                "auth_data": self.auth_data,
                "method": "query",
                "args": [q],
                "kwargs": {},
            },
            headers={"Authorization": f"Bearer {self.auth_data['token']}"},
        )
        resp.raise_for_status()
        return resp.json()

    # Example for other SDK methods:
    def __getattr__(self, name):
        """Any other method call gets forwarded to the server."""

        def method(*args, **kwargs):
            resp = requests.post(
                os.path.join(self.auth_data["dash_app_url"], "qb-proxy"),
                json={
                    "auth_data": self.auth_data,
                    "method": name,
                    "args": args,
                    "kwargs": kwargs,
                },
                headers={"Authorization": f"Bearer {self.auth_data['token']}"},
            )
            resp.raise_for_status()
            return resp.json()

        return method

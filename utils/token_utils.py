from typing import Text
import jwt

# use the correct exception class
from jwt.exceptions import DecodeError
import flask
import os
from requests_oauthlib import OAuth2Session


def is_token_expired(token: str) -> bool:
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get("exp")

        if exp is None:
            return True  # no expiration = treat as expired

        import time

        return time.time() >= exp

    except DecodeError:
        return True  # if the token can't be decoded, treat as expired


def refresh_token() -> Text:
    session = OAuth2Session(
        os.environ.get("OAUTH_CLIENT_ID"),
    )

    # Use the refresh token to get a new access token
    new_token = session.refresh_token(
        "https://oauth2.googleapis.com/token",
        refresh_token=flask.session["google_user"]["refresh_token"],
        client_id=os.environ.get("OAUTH_CLIENT_ID"),
        client_secret=os.environ.get("OAUTH_CLIENT_SECRET"),
    )
    flask.session["google_user"]["token"] = new_token["id_token"]
    flask.session["google_user"]["refresh_token"] = new_token.get(
        "refresh_token",
        flask.session["google_user"]["refresh_token"],
    )
    return flask.session["google_user"]["token"]

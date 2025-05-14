from typing import Dict, Any, Union, Optional, Text

import sqlalchemy
from pymongo import MongoClient
from google.cloud import bigquery
from google.oauth2 import service_account
import sshtunnel
import os


def start_ssh_tunnel(
    remote_host: str,
    remote_port: int,
    ssh_host: Optional[Text] = None,
    ssh_port: Optional[Union[Text, int]] = None,
    ssh_user: Optional[Text] = None,
    ssh_private_key: Optional[Text] = None,
):
    if ssh_host and ssh_user and ssh_private_key:
        tunnel = sshtunnel.SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_pkey=ssh_private_key,
            remote_bind_address=(remote_host, remote_port),
        )
        tunnel.start()
        return tunnel
    return None


def connect_to_sql(
    uri: Optional[Text] = None,
    host: Optional[Text] = None,
    port: Optional[Union[Text, int]] = None,
    user: Optional[Text] = None,
    password: Optional[Text] = None,
    database: Optional[Text] = None,
    ssh_host: Optional[Text] = None,
    ssh_port: Optional[Union[Text, int]] = None,
    ssh_user: Optional[Text] = None,
    ssh_private_key: Optional[Text] = None,
    sql_type: Text = "mysql",
):
    ssh_tunnel = (
        start_ssh_tunnel(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            ssh_private_key=ssh_private_key,
            remote_host=host,
            remote_port=port,
        )
        if ssh_host
        else None
    )
    host = "127.0.0.1" if ssh_tunnel else host
    port = ssh_tunnel.local_bind_port if ssh_tunnel else port

    if sql_type == "mysql":
        prefix = "mysql+pymysql://"
    elif sql_type == "postgres":
        prefix = "postgresql+psycopg2://"

    if uri and user and user not in uri:
        uri = f'{uri.split("//")[0]}//{user}:{password}@{uri.split("//")[-1]}'

    if uri and database and database not in uri:
        uri = os.path.join(uri, database)

    if uri:
        db_url = uri
    elif user:
        db_url = f"{prefix}{user}:{password}@{host}:{port}" f"/{database}"
    else:
        db_url = f"{prefix}{host}:{port}/{database}"

    conn = sqlalchemy.create_engine(db_url)

    # Optional auto cleanup
    def cleanup():
        conn.dispose()
        if ssh_tunnel:
            ssh_tunnel.stop()

    return {"connection": conn, "ssh_tunnel": ssh_tunnel, "cleanup": cleanup}


def connect_to_mongo(
    uri: Optional[Text] = None,
    host: Optional[Text] = None,
    port: Optional[Union[Text, int]] = None,
    user: Optional[Text] = None,
    password: Optional[Text] = None,
    database: Optional[Text] = None,
    ssh_host: Optional[Text] = None,
    ssh_port: Optional[Union[Text, int]] = None,
    ssh_user: Optional[Text] = None,
    ssh_private_key: Optional[Text] = None,
):
    ssh_tunnel = (
        start_ssh_tunnel(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            ssh_private_key=ssh_private_key,
            remote_host=host,
            remote_port=port,
        )
        if ssh_host
        else None
    )
    host = "127.0.0.1" if ssh_tunnel else host
    port = ssh_tunnel.local_bind_port if ssh_tunnel else port

    if uri and user and user not in uri:
        uri = f'{uri.split("//")[0]}//{user}:{password}@{uri.split("//")[-1]}'

    if uri and database and database not in uri:
        uri = os.path.join(uri, database)

    if uri:
        uri = uri
    elif user:
        uri = f"mongodb://{user}:{password}@{host}:{port}/{database}"
    else:
        uri = f"mongodb://{host}:{port}/{database}"

    conn = MongoClient(uri)

    # Optional auto cleanup
    def cleanup():
        conn.close()
        if ssh_tunnel:
            ssh_tunnel.stop()

    return {"connection": conn, "ssh_tunnel": ssh_tunnel, "cleanup": cleanup}


def connect_to_biquery(gcp_credentials_json: Dict[Text, Any]):
    credentials = service_account.Credentials.from_service_account_info(
        gcp_credentials_json
    )
    conn = bigquery.Client(credentials=credentials, project=credentials.project_id)

    return {"connection": conn, "ssh_tunnel": None, "cleanup": None}

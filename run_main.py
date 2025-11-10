from typing import Optional, Dict, Text, Any
from utils.run_utils import run_sims
import json
import os
import requests
import traceback
import logging
from pathlib import Path

# blob_url = os.environ.get("BLOB_URL")


def main():
    input_data = {}
    run_output = None
    try:
        logging.info("downloading data")
        config_path = Path(os.environ.get("RUNNER_DATA_PATH"))

        # Load the data
        with config_path.open() as f:
            input_data = json.load(f)
        logging.info("running run_docs")
        # run_docs(**input_data)
        run_sims(**input_data)

    except Exception as e:
        logging.info(e.args[0])
        logging.info(traceback.format_exc())
        run_output = {
            "failed": True,
            "message": f"An unhandled exception has occured:\n{traceback.format_exc()}",
        }

    auth_data = input_data.get("auth_data", {})
    docs_to_run = input_data.get("docs_to_run", [])
    send_output(
        {},
        docs_to_run,
        auth_data,
        input_data.get("caller", None),
        run_output=run_output,
        run_completed=True,
    )


def send_output(
    outputs,
    docs_to_run,
    auth_data,
    caller,
    run_completed: bool = False,
    run_output: Optional[Dict[Text, Any]] = None,
):
    # Send the attribute result back to the backend
    if not run_output:
        run_output = {"failed": False, "message": ""}
    data = {
        "docs_to_run": docs_to_run,
        "outputs": outputs,
        "caller": caller,
        "auth_data": auth_data,
        "run_completed": run_completed,
        "run_output": run_output,
    }

    requests.post(
        os.path.join(auth_data["dash_app_url"], "job-result"),
        json=data,
        headers={"Authorization": f"Bearer {auth_data['token']}"},
    )


if __name__ == "__main__":
    main()

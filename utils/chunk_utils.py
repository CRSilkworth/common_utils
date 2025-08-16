from utils.misc_utils import failed_output
from utils.type_utils import (
    deserialize_typehint,
    describe_json_schema,
    chunked_type_map,
)
from utils.serialize_utils import attempt_deserialize, attempt_serialize
from utils.gcp_utils import (
    upload_via_signed_post,
    read_from_gcs_signed_urls,
    request_policy,
)
import json


def combine_chunk_outputs(chunk_output_files, att_dict, failed):
    output = {
        "failed": failed,
        "value": None,
        "combined_output": "",
        "stdout_output": "",
        "stderr_output": "",
    }
    chunk_schemas = None
    definitions = None
    size = 0
    num_chunks = 0
    signed_urls = []
    for chunk_file_output in chunk_output_files:
        if chunk_schemas is None:
            chunk_schemas = []
        chunk_schemas.extend(chunk_file_output.get("chunk_schemas", []))

        output["combined_output"] += "".join(chunk_file_output["combined_outputs"])
        output["stdout_output"] += "".join(chunk_file_output["stdout_outputs"])
        output["stderr_output"] += "".join(chunk_file_output["stderr_outputs"])

        signed_urls.append(chunk_file_output.get("signed_url", None))
        definitions = chunk_file_output.get("definitions", None)

        size += chunk_file_output.get("buffer_size", 0)
        num_chunks += chunk_file_output.get("num_chunks", 0)

    if failed:
        output["failed"] = True
        output["value"] = None
        return output

    value = read_from_gcs_signed_urls(signed_urls)

    def deserialized_gen(generator):
        for file_content in generator:
            file_content, output, _ = attempt_deserialize(
                file_content, att_dict["value_type"]
            )
            if output:
                raise ValueError(
                    "failed to deserialize file from generator: "
                    f"{output['stderr_output']}"
                )
            for item in file_content:
                yield item

    output["value"] = deserialized_gen(value)
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "x-type": "generator",
        "type": "array",
        "minItems": num_chunks,
        "maxItems": num_chunks,
        "items": chunk_schemas[0] if chunk_schemas else {},
        "definitions": definitions if definitions else [],
    }
    output["signed_urls"] = signed_urls
    output["_schema"] = json.dumps(schema, indent=2)
    output["preview"] = output["_schema"]

    local_rep = (att_dict["bucket"], att_dict["new_version"])
    local_type = deserialize_typehint(att_dict["_local_type"])
    _local_rep, _ = attempt_serialize(local_rep, local_type)

    output["_local_rep"] = _local_rep

    output["size_delta"] = size - att_dict["value_size"]

    return output


def upload_chunk_file(
    doc_id,
    attribute_name,
    att_dict,
    chunk_file_num,
    buffer,
    buffer_size,
    auth_data,
    definitions,
):
    # Combine all value_chunks into one
    combined_values = [chunk["value_chunk"] for chunk in buffer]
    combined_outputs = [chunk["combined_output"] for chunk in buffer]
    stdout_outputs = [chunk["stdout_output"] for chunk in buffer]
    stderr_outputs = [chunk["stderr_output"] for chunk in buffer]
    chunk_schemas = []
    for value_chunk in combined_values:
        chunk_schema, definitions = describe_json_schema(
            value_chunk, definitions=definitions
        )
        chunk_schemas.append(chunk_schema)

    _values, serialize_output = attempt_serialize(
        combined_values, chunked_type_map[att_dict["value_type"]]
    )
    if serialize_output:
        return serialize_output
    policy_data = {"auth_data": auth_data}
    policy_data.update(
        {
            "doc_id": doc_id,
            "attribute_name": attribute_name,
            "version": att_dict["new_version"],
            "chunk_file_num": chunk_file_num,
        },
    )
    policy = request_policy(
        auth_data["dash_app_url"], data=policy_data, token=auth_data["token"]
    )

    status = upload_via_signed_post(policy, _values)
    if status not in (200, 204):
        return failed_output(f"Failed to upload file to gcs. Got status code {status}")

    return {
        "chunk_file_num": chunk_file_num,
        "chunk_schemas": chunk_schemas,
        "definitions": definitions,
        "url": policy["url"],
        "signed_url": policy["signed_url"],
        "combined_outputs": combined_outputs,
        "stdout_outputs": stdout_outputs,
        "stderr_outputs": stderr_outputs,
        "num_chunks": len(buffer),
        "size": buffer_size,
    }

import base64
import io
import pandas as pd
import json
from typing import List, Dict, Optional, Union
from PIL import Image
import PyPDF2
import xml.etree.ElementTree as ET
from utils.type_utils import Files
from utils.json_schema_utils import describe_json_schema


def guess_file_extension(data: bytes) -> str:
    # Check for binary file signatures (magic numbers)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    elif data.startswith(b"GIF89a") or data.startswith(b"GIF87a"):
        return "gif"
    elif data.startswith(b"%PDF"):
        return "pdf"
    elif data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "xls"
    elif data[0:4] == b"PK\x03\x04":
        if b"[Content_Types].xml" in data:
            if b"xl/" in data:
                return "xlsx"
            elif b"word/" in data:
                return "docx"
            elif b"ppt/" in data:
                return "pptx"
            else:
                return "zip"  # fallback for unknown PK-based format

    # Try decoding as UTF-8 to handle text-based files
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "bin"  # unknown binary

    # Heuristics for text-based formats
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    elif "," in text:
        return "csv"
    elif "\t" in text:
        return "tsv"
    elif all(
        c
        in (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK"
            "LMNOPQRSTUVWXYZ0123456789 \n\r\t.,:;!?\"'-()"
        )
        for c in text[:100]
    ):
        return "txt"

    return "txt"  # fallback for undecorated readable text


def get_file_schema(
    files: Files,
) -> List[Dict[str, Optional[Union[str, Dict[str, Union[str, int, float]]]]]]:
    """
    Analyzes uploaded files and returns helpful metadata for each one.

    Supports CSV, Excel, JSON, text, image, PDF, XML.

    Args:
        files: A list of file dicts with keys 'file_content', 'file_name',
        and 'file_date'

    Returns:
        List[Dict[str, Union[str, Dict]]]: List of dictionaries containing metadata.
    """
    results = {
        "x-type": "Files",
        "type": "array",
        "minItems": len(files),
        "maxItems": len(files),
        "items": [],
        "additionalItems": False,
    }

    for file_num, file in enumerate(files):
        file_content = file["file_content"]
        file_name = file["file_name"]
        file_date = file["file_date"]
        info = {
            "x-type": "dict",
            "type": "object",
            "properties": {
                "file_name": file_name,
                "file_date": file_date,
                "file_content": {
                    "type": "str",
                    "description": "a string of the file type and base64 encoded string"
                    " of the entire file contents separated by a ','",
                },
            },
        }

        try:
            content_type, content_string = file_content.split(",")
            decoded = base64.b64decode(content_string)
            extension = file_name.split(".")[-1].lower()

            # CSV / TSV
            if extension in ["csv", "tsv"]:
                sep = "," if extension == "csv" else "\t"
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=sep)
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": extension,
                        "num_rows": df.shape[0],
                        "num_columns": df.shape[1],
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.astype(str).to_dict(),
                        "head": df.head(5).to_dict(orient="records"),
                    },
                }

            # Excel
            elif extension in ["xls", "xlsx"]:
                df = pd.read_excel(io.BytesIO(decoded))
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": extension,
                        "num_rows": df.shape[0],
                        "num_columns": df.shape[1],
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.astype(str).to_dict(),
                        "head": df.head(5).to_dict(orient="records"),
                    },
                }

            # JSON
            elif extension == "json":
                json_obj = json.loads(decoded.decode("utf-8"))
                if isinstance(json_obj, list):
                    count = len(json_obj)
                    sample = json_obj[:3]
                elif isinstance(json_obj, dict):
                    count = len(json_obj)
                    sample = {k: json_obj[k] for k in list(json_obj)[:3]}
                else:
                    count = "Unknown"
                    sample = str(json_obj)

                schema_hash, defs = describe_json_schema(json_obj)
                schema = {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$ref": f"#/$defs/{schema_hash}",
                    "$defs": defs,
                }
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": "json",
                        "item_count": count,
                        "sample": sample,
                        "schema": schema,
                    },
                }
            # Text files
            elif extension in ["txt", "md", "log"]:
                text = decoded.decode("utf-8")
                lines = text.splitlines()
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": "text",
                        "line_count": len(lines),
                        "preview": "\n".join(lines[:5]),
                    },
                }

            # Image files
            elif extension in ["png", "jpg", "jpeg", "gif", "bmp"]:
                image = Image.open(io.BytesIO(decoded))
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": "image",
                        "format": image.format,
                        "mode": image.mode,
                        "size": image.size,  # (width, height)
                    },
                }

            # PDF
            elif extension == "pdf":
                pdf = PyPDF2.PdfReader(io.BytesIO(decoded))
                num_pages = len(pdf.pages)
                text_sample = ""
                if num_pages > 0:
                    try:
                        text_sample = pdf.pages[0].extract_text()[:500]
                    except Exception:
                        text_sample = "[Unable to extract text from page 1]"
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": "pdf",
                        "num_pages": num_pages,
                        "sample_text": text_sample,
                    },
                }

            # XML
            elif extension == "xml":
                root = ET.fromstring(decoded.decode("utf-8"))
                elements = [elem.tag for elem in root.iter()]
                structure = list(dict.fromkeys(elements))[:10]
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": "xml",
                        "root_tag": root.tag,
                        "num_elements": len(elements),
                        "sample_tags": structure,
                    },
                }

            else:
                info["properties"]["summary"] = {
                    "type": "object",
                    "x-type": "file_summary",
                    "properties": {
                        "file_type": extension,
                        "note": "Unsupported file type. Data metadata shown.",
                    },
                }

        except Exception as e:
            info["properties"]["summary"] = {
                "type": "object",
                "x-type": "file_summary",
                "properties": {"error": f"Failed to process file: {str(e)}"},
            }

        results["items"].append(
            {"x-type": "File", "type": "object", "properties": info}
        )

    return results

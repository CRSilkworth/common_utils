import base64
import io
from PIL import Image
import PyPDF2
from utils.file_utils import guess_file_extension, get_file_schema
import json


def test_guess_file_extension_known_types():
    # PNG
    png_bytes = b"\x89PNG\r\n\x1a\nrest of data"
    assert guess_file_extension(png_bytes) == "png"

    # JPG
    jpg_bytes = b"\xff\xd8\xffrest of data"
    assert guess_file_extension(jpg_bytes) == "jpg"

    # GIF
    gif_bytes = b"GIF89arest of data"
    assert guess_file_extension(gif_bytes) == "gif"

    gif_bytes2 = b"GIF87arest of data"
    assert guess_file_extension(gif_bytes2) == "gif"

    # PDF
    pdf_bytes = b"%PDF-1.4 rest of data"
    assert guess_file_extension(pdf_bytes) == "pdf"

    # XLS
    xls_bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest of data"
    assert guess_file_extension(xls_bytes) == "xls"

    # PK based - XLSX
    xlsx_bytes = b"PK\x03\x04[Content_Types].xmlxl/"
    assert guess_file_extension(xlsx_bytes) == "xlsx"

    # PK based - DOCX
    docx_bytes = b"PK\x03\x04[Content_Types].xmlword/"
    assert guess_file_extension(docx_bytes) == "docx"

    # PK based - PPTX
    pptx_bytes = b"PK\x03\x04[Content_Types].xmlppt/"
    assert guess_file_extension(pptx_bytes) == "pptx"

    # PK based - ZIP fallback
    zip_bytes = b"PK\x03\x04[Content_Types].xmlother/"
    assert guess_file_extension(zip_bytes) == "zip"


def test_guess_file_extension_text_types():
    # JSON object
    json_bytes = b'{"key": "value"}'
    assert guess_file_extension(json_bytes) == "json"

    # JSON array
    json_array_bytes = b"[1, 2, 3]"
    assert guess_file_extension(json_array_bytes) == "json"

    # CSV
    csv_bytes = b"a,b,c\n1,2,3"
    assert guess_file_extension(csv_bytes) == "csv"

    # TSV
    tsv_bytes = b"a\tb\tc\n1\t2\t3"
    assert guess_file_extension(tsv_bytes) == "tsv"

    # TXT with only allowed chars
    txt_bytes = b"Hello world! This is a test."
    assert guess_file_extension(txt_bytes) == "txt"

    # TXT fallback
    weird_bytes = b"\x00\x01\x02\x03"
    assert guess_file_extension(weird_bytes) == "txt"


def test_get_file_schema_csv():
    data = "a,b\n1,2\n3,4"
    encoded = base64.b64encode(data.encode()).decode()
    files = [
        {
            "file_content": f"text/csv,{encoded}",
            "file_name": "test.csv",
            "file_date": "2023-01-01",
        }
    ]
    schema = get_file_schema(files)
    assert schema["minItems"] == 1
    assert schema["maxItems"] == 1
    items = schema["items"]
    assert len(items) == 1
    props = items[0]["properties"]
    assert props["file_name"] == "test.csv"
    assert props["file_date"] == "2023-01-01"
    assert props["summary"]["file_type"] == "csv"
    assert props["summary"]["num_rows"] == 2
    assert props["summary"]["num_columns"] == 2
    assert "a" in props["summary"]["columns"]


def test_get_file_schema_json():
    json_obj = [{"key": "value"}, {"key": "value2"}]
    encoded = base64.b64encode(json.dumps(json_obj).encode()).decode()
    files = [
        {
            "file_content": f"application/json,{encoded}",
            "file_name": "test.json",
            "file_date": "2023-01-01",
        }
    ]
    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["file_name"] == "test.json"
    assert props["summary"]["file_type"] == "json"
    assert props["summary"]["item_count"] == 2


def test_get_file_schema_text():
    text = "line1\nline2\nline3\nline4\nline5\nline6"
    encoded = base64.b64encode(text.encode()).decode()
    files = [
        {
            "file_content": f"text/plain,{encoded}",
            "file_name": "test.txt",
            "file_date": "2023-01-01",
        }
    ]
    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["file_name"] == "test.txt"
    assert props["summary"]["file_type"] == "text"
    assert props["summary"]["line_count"] == 6
    assert props["summary"]["preview"].count("\n") < 5


def test_get_file_schema_image(monkeypatch):
    # Create a simple 1x1 px PNG image in memory
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    encoded = base64.b64encode(img_bytes).decode()
    files = [
        {
            "file_content": f"image/png,{encoded}",
            "file_name": "test.png",
            "file_date": "2023-01-01",
        }
    ]

    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["file_name"] == "test.png"
    assert props["summary"]["file_type"] == "image"
    assert props["summary"]["format"] == "PNG"
    assert props["summary"]["size"] == (1, 1)


def test_get_file_schema_pdf(monkeypatch):
    # Create a minimal PDF file in memory
    pdf_writer = PyPDF2.PdfWriter()
    pdf_writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    pdf_writer.write(buf)
    pdf_bytes = buf.getvalue()

    encoded = base64.b64encode(pdf_bytes).decode()
    files = [
        {
            "file_content": f"application/pdf,{encoded}",
            "file_name": "test.pdf",
            "file_date": "2023-01-01",
        }
    ]

    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["file_name"] == "test.pdf"
    assert props["summary"]["file_type"] == "pdf"
    assert props["summary"]["num_pages"] == 1


def test_get_file_schema_xml():
    xml_str = "<root><child1/><child2/></root>"
    encoded = base64.b64encode(xml_str.encode()).decode()
    files = [
        {
            "file_content": f"application/xml,{encoded}",
            "file_name": "test.xml",
            "file_date": "2023-01-01",
        }
    ]

    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["file_name"] == "test.xml"
    assert props["summary"]["file_type"] == "xml"
    assert props["summary"]["root_tag"] == "root"
    assert "child1" in props["summary"]["sample_tags"]


def test_get_file_schema_unsupported_file():
    data = b"some random data"
    encoded = base64.b64encode(data).decode()
    files = [
        {
            "file_content": f"application/unknown,{encoded}",
            "file_name": "file.unknown",
            "file_date": "2023-01-01",
        }
    ]

    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert props["summary"]["file_type"] == "unknown"
    assert "Unsupported file type" in props["summary"]["note"]


def test_get_file_schema_error_handling():
    # Pass malformed base64 to trigger exception
    files = [
        {
            "file_content": "text/csv,not_base64",
            "file_name": "test.csv",
            "file_date": "2023-01-01",
        }
    ]

    schema = get_file_schema(files)
    props = schema["items"][0]["properties"]
    assert "error" in props["summary"]

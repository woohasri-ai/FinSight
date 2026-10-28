import json

from finsight.ingestion.save import save_pages_as_json


def test_save_pages_as_json_creates_file_and_preserves_data(tmp_path):
    output_path = tmp_path / "processed" / "pages.json"

    pages = [
        {"page_number": 1, "text": "TCS Annual Report"},
        {"page_number": 2, "text": "Revenue was ₹100 crore"},
    ]

    save_pages_as_json(pages, output_path)

    assert output_path.is_file()

    with output_path.open("r", encoding="utf-8") as file:
        saved_pages = json.load(file)

    assert saved_pages == pages

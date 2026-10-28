import pymupdf
import pytest

from finsight.ingestion.parse import extract_pdf_pages


def test_extract_pdf_pages_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "FinSight annual report")

        document.new_page()  # Intentionally leave page 2 empty.

        third_page = document.new_page()
        third_page.insert_text((72, 72), "Revenue was 100 crore")

        document.save(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[0]["text"] == "FinSight annual report"
    assert pages[1]["page_number"] == 3
    assert pages[1]["text"] == "Revenue was 100 crore"


def test_extract_pdf_pages_raises_for_missing_file(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_pages(missing_pdf)


def test_extract_pdf_pages_returns_empty_list_for_blank_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert pages == []

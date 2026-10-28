from pathlib import Path

import pymupdf


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, int | str]]:
    """Extract text from a PDF, preserving its one-based page numbers."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[dict[str, int | str]] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

    return pages

from pathlib import Path

from finsight.ingestion.parse import extract_pdf_pages
from finsight.ingestion.save import save_pages_as_json


def main() -> None:
    """Extract a PDF report and save its pages as JSON."""
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "data" / "raw" / "tcs_annual_report_2025_26.pdf"
    output_path = project_root / "data" / "processed" / "tcs_annual_report_2025_26.json"

    print("Extracting text from the TCS annual report...")
    pages = extract_pdf_pages(pdf_path)

    save_pages_as_json(pages, output_path)

    print(f"Pages extracted: {len(pages)}")
    print(f"JSON saved to: {output_path}")


if __name__ == "__main__":
    main()

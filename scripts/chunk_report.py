import json
from pathlib import Path

from tokenizers import Tokenizer

from finsight.ingestion.chunk import chunk_page_text


def main() -> None:
    """Chunk extracted annual-report pages and save the results as JSON."""
    project_root = Path(__file__).resolve().parent.parent

    input_path = project_root / "data" / "processed" / "tcs_annual_report_2025_26.json"
    output_path = project_root / "data" / "processed" / "tcs_annual_report_2025_26_chunks.json"

    if not input_path.is_file():
        raise FileNotFoundError(f"Extracted report JSON not found: {input_path}")

    pages: list[dict[str, int | str]] = json.loads(input_path.read_text(encoding="utf-8"))

    # This temporary tokenizer is suitable for testing the pipeline.
    # We'll replace it with the BGE model's tokenizer before building retrieval.
    tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    all_chunks: list[dict[str, int | str]] = []

    for page in pages:
        page_chunks = chunk_page_text(
            text=str(page["text"]),
            page_number=int(page["page_number"]),
            source_doc="tcs_annual_report_2025_26.pdf",
            tokenizer=tokenizer,
            chunk_size=500,
            overlap=80,
        )
        all_chunks.extend(page_chunks)

    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Pages processed: {len(pages)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

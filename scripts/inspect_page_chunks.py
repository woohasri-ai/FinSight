import json
from pathlib import Path
from typing import Any

CHUNKS_PATH = Path("data/processed/tcs_annual_report_2025_26_chunks.json")
TARGET_PAGE = 14


def main() -> None:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    chunks: list[dict[str, Any]] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    page_chunks = [chunk for chunk in chunks if int(chunk["page_number"]) == TARGET_PAGE]

    if not page_chunks:
        print(f"No chunks found for page {TARGET_PAGE}.")
        return

    print(f"Found {len(page_chunks)} chunks on page {TARGET_PAGE}.")

    for index, chunk in enumerate(page_chunks, start=1):
        print("\n" + "=" * 80)
        print(f"Chunk {index} of {len(page_chunks)}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print("-" * 80)
        print(chunk["text"])

    print("\nInspection complete.")


if __name__ == "__main__":
    main()

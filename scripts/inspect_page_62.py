import json
from pathlib import Path

chunks_path = Path("data/processed/tcs_annual_report_2025_26_chunks.json")
chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

page_chunks = [chunk for chunk in chunks if chunk["page_number"] == 62]

print(f"Found {len(page_chunks)} chunks on page 62.\n")

for chunk in page_chunks:
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print("-" * 80)
    print(chunk["text"])
    print("=" * 80)

import json
from pathlib import Path

chunks_path = Path("data/processed/tcs_annual_report_2025_26_chunks.json")
chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

for chunk in chunks:
    if chunk["page_number"] == 58:
        print("\n" + "=" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("-" * 60)
        print(chunk["text"][:1500])

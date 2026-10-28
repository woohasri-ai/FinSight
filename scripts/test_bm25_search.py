import json
from pathlib import Path

from finsight.retrieval.bm25 import build_bm25_index, search_bm25


def main() -> None:
    """Search TCS annual-report chunks using BM25."""
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "processed" / "tcs_annual_report_2025_26_chunks.json"

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    print(f"Loaded {len(chunks)} chunks.")
    index = build_bm25_index(chunks)

    query = input("\nEnter your question about the TCS annual report: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    results = search_bm25(query, chunks, index, top_k=5)

    if not results:
        print("\nNo matching passages found.")
        return

    print("\nTop BM25 matching passages:\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Page: {result['page_number']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source_doc']}")
        print(f"Text:\n{result['text']}\n")


if __name__ == "__main__":
    main()

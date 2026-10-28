import json
from pathlib import Path

from finsight.retrieval.dense import (
    load_embedding_model,
    load_saved_embeddings,
    search_chunks,
)


def main() -> None:
    """Search report chunks using precomputed embeddings."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"

    chunks_path = data_dir / "tcs_annual_report_2025_26_chunks.json"
    embeddings_path = data_dir / "tcs_annual_report_2025_26_embeddings.npy"

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = load_saved_embeddings(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Chunk count does not match saved embedding count.")

    print(f"Loaded {len(chunks)} chunks and their saved embeddings.")
    print("Loading BGE embedding model...")
    model = load_embedding_model()

    query = input("\nEnter your question about the TCS annual report: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    results = search_chunks(
        query,
        chunks,
        model,
        top_k=5,
        chunk_embeddings=embeddings,
    )

    print("\nTop matching passages:\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Page: {result['page_number']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source_doc']}")
        print(f"Text:\n{result['text']}\n")


if __name__ == "__main__":
    main()

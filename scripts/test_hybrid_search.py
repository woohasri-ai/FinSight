import json
from pathlib import Path

import numpy as np

from finsight.retrieval.bm25 import build_bm25_index
from finsight.retrieval.dense import load_embedding_model, load_saved_embeddings
from finsight.retrieval.hybrid import hybrid_search
from finsight.retrieval.rerank import load_reranker, rerank_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "tcs_annual_report_2025_26_chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "tcs_annual_report_2025_26_embeddings.npy"

# Retrieve a wider candidate pool before reranking.
CANDIDATE_K = 50
HYBRID_TOP_K = 50
RERANK_TOP_K = 5

TARGET_QUERY = "What AI capabilities and partnerships is TCS developing to support clients?"
EXPECTED_PAGES = [14, 62]


def load_chunks() -> list[dict[str, int | float | str]]:
    """Load processed chunks and validate the input file."""
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}\nRun scripts/chunk_report.py first."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks: list[dict[str, int | float | str]] = json.load(file)

    if not chunks:
        raise ValueError("The chunk file contains no chunks.")

    return chunks


def print_candidate_summary(
    candidates: list[dict[str, int | float | str]],
    expected_pages: list[int] | None = None,
) -> None:
    """Display the hybrid candidate pages before reranking."""
    print(f"\nHybrid retrieval returned {len(candidates)} candidates.")

    candidate_pages = [int(candidate["page_number"]) for candidate in candidates]

    print("Candidate pages in rank order:")
    print(candidate_pages)

    if expected_pages:
        found_pages = sorted(set(candidate_pages) & set(expected_pages))
        missing_pages = sorted(set(expected_pages) - set(candidate_pages))

        print(f"Expected pages: {expected_pages}")
        print(f"Expected pages found before reranking: {found_pages}")
        print(f"Expected pages missing before reranking: {missing_pages}")


def main() -> None:
    chunks = load_chunks()

    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_PATH}\nRun scripts/build_embeddings.py first."
        )

    embeddings: np.ndarray = load_saved_embeddings(EMBEDDINGS_PATH)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Chunk/embedding mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loaded embeddings with shape {embeddings.shape}.")

    print("\nLoading embedding model and BM25 index...")
    embedding_model = load_embedding_model()
    bm25_index = build_bm25_index(chunks)

    query = input("\nEnter your question about the TCS annual report: ").strip()

    if not query:
        print("Question cannot be empty.")
        return

    is_target_query = query.casefold() == TARGET_QUERY.casefold()
    expected_pages = EXPECTED_PAGES if is_target_query else None

    candidates = hybrid_search(
        query=query,
        chunks=chunks,
        model=embedding_model,
        bm25_index=bm25_index,
        chunk_embeddings=embeddings,
        top_k=HYBRID_TOP_K,
        candidate_k=CANDIDATE_K,
    )

    if not candidates:
        print("\nNo relevant passages found.")
        return

    print_candidate_summary(candidates, expected_pages)

    print("\nLoading cross-encoder reranker...")
    reranker = load_reranker()

    results = rerank_chunks(
        query=query,
        chunks=candidates,
        model=reranker,
        top_k=RERANK_TOP_K,
    )

    print(f"\nTop {len(results)} reranked passages:")

    reranked_pages: list[int] = []

    for rank, result in enumerate(results, start=1):
        page_number = int(result["page_number"])
        reranked_pages.append(page_number)

        print(f"\n--- Result {rank} ---")
        print(f"Page: {page_number}")
        print(f"Reranker score: {float(result['score']):.4f}")
        print(f"Source: {result['source_doc']}")
        print(f"Chunk ID: {result.get('chunk_id', 'not available')}")
        print(f"Text:\n{result['text']}")

    if expected_pages:
        candidate_pages = {int(candidate["page_number"]) for candidate in candidates}
        found_before_reranking = sorted(candidate_pages & set(expected_pages))
        found_after_reranking = sorted(set(reranked_pages) & set(expected_pages))
        missing_after_reranking = sorted(set(expected_pages) - set(reranked_pages))

        print("\n=== Targeted retrieval diagnostics ===")
        print(f"Expected pages: {expected_pages}")
        print(f"Found in hybrid candidates: {found_before_reranking}")
        print(f"Found in reranked top {RERANK_TOP_K}: {found_after_reranking}")
        print(f"Still missing after reranking: {missing_after_reranking}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from finsight.retrieval.bm25 import build_bm25_index
from finsight.retrieval.dense import load_embedding_model, load_saved_embeddings
from finsight.retrieval.hybrid import hybrid_search
from finsight.retrieval.rerank import load_reranker, rerank_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "tcs_annual_report_2025_26_chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "tcs_annual_report_2025_26_embeddings.npy"
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_golden_set.json"

CANDIDATE_K = 50
HYBRID_CUTOFFS = (5, 10, 50)
RERANK_TOP_K = 10
DIAGNOSTIC_QUESTION_ID = "TCS-RET-009"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate FinSight's hybrid retrieval and optional reranking."
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranker loading, inference, and reranker diagnostics.",
    )
    return parser.parse_args()


def recall_at_k(
    results: list[dict[str, Any]],
    expected_pages: list[int],
    k: int,
) -> float:
    """Calculate fractional page-level recall among the first k results."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    unique_expected_pages = set(expected_pages)

    if not unique_expected_pages:
        return 0.0

    retrieved_pages = {int(result["page_number"]) for result in results[:k]}
    found_pages = retrieved_pages & unique_expected_pages

    return len(found_pages) / len(unique_expected_pages)


def chunk_recall_at_k(
    results: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """Calculate fractional chunk-level recall among the first k results."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    expected_ids = set(relevant_chunk_ids)

    if not expected_ids:
        return 0.0

    retrieved_ids = {str(result["chunk_id"]) for result in results[:k] if "chunk_id" in result}

    found_ids = retrieved_ids & expected_ids

    return len(found_ids) / len(expected_ids)


def load_json(path: Path) -> Any:
    """Load JSON data with a clear error if the file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def diagnose_relevant_chunks(
    query: str,
    hybrid_results: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    reranker: Any,
) -> None:
    """Print hybrid and full-pool reranker ranks for labeled relevant chunks."""
    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        print("No labeled relevant chunk IDs are available for diagnosis.")
        return

    print("\nDiagnostic: relevant chunks in hybrid candidates")

    found_candidate_ids: set[str] = set()

    for rank, result in enumerate(hybrid_results, start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in relevant_ids:
            found_candidate_ids.add(chunk_id)
            print(
                f"Hybrid rank: {rank} | "
                f"Page: {result['page_number']} | "
                f"Chunk: {chunk_id} | "
                f"Hybrid score: {result['score']:.4f}"
            )

    missing_candidates = relevant_ids - found_candidate_ids

    if missing_candidates:
        print(f"Relevant chunks missing from hybrid candidates: {sorted(missing_candidates)}")
        return

    print("All labeled relevant chunks are present in hybrid candidates.")

    # Diagnostic only: inspect the complete candidate pool.
    diagnostic_results = rerank_chunks(
        query=query,
        chunks=hybrid_results,
        model=reranker,
        top_k=len(hybrid_results),
    )

    print("\nDiagnostic: relevant chunks after reranking")

    found_reranked_ids: set[str] = set()

    for rank, result in enumerate(diagnostic_results, start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in relevant_ids:
            found_reranked_ids.add(chunk_id)
            print(
                f"Reranker rank: {rank} | "
                f"Page: {result['page_number']} | "
                f"Chunk: {chunk_id} | "
                f"Reranker score: {result['score']:.4f}"
            )

    missing_after_reranking = relevant_ids - found_reranked_ids

    if missing_after_reranking:
        print(f"Relevant chunks missing from reranker output: {sorted(missing_after_reranking)}")


def main() -> None:
    args = parse_args()
    run_reranker = not args.no_rerank

    chunks: list[dict[str, Any]] = load_json(CHUNKS_PATH)
    golden_set: list[dict[str, Any]] = load_json(GOLDEN_SET_PATH)

    if not chunks:
        raise ValueError("The chunk file contains no chunks.")

    if not golden_set:
        raise ValueError("The retrieval golden set contains no questions.")

    embeddings: np.ndarray = load_saved_embeddings(EMBEDDINGS_PATH)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Chunk/embedding mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    print(f"Loaded {len(chunks)} chunks and {len(golden_set)} evaluation questions.")
    print(f"Hybrid candidate pool: {CANDIDATE_K}")
    print(f"Hybrid evaluation cutoffs: {HYBRID_CUTOFFS}")

    if run_reranker:
        print(f"Reranked cutoff: {RERANK_TOP_K}")
    else:
        print("Reranking disabled: hybrid-only evaluation.")

    print("\nLoading embedding model and BM25 index...")
    embedding_model = load_embedding_model()
    bm25_index = build_bm25_index(chunks)

    reranker = None

    if run_reranker:
        print("\nLoading reranker...")
        reranker = load_reranker()

    metric_totals = {
        **{f"hybrid_recall@{k}": 0.0 for k in HYBRID_CUTOFFS},
    }

    chunk_metric_totals = {
        **{f"hybrid_chunk_recall@{k}": 0.0 for k in HYBRID_CUTOFFS},
    }

    if run_reranker:
        metric_totals[f"reranked_recall@{RERANK_TOP_K}"] = 0.0
        chunk_metric_totals[f"reranked_chunk_recall@{RERANK_TOP_K}"] = 0.0

    chunk_evaluated_questions = 0

    for question in golden_set:
        question_id = question["id"]
        query = question["question"]
        expected_pages = [int(page) for page in question["expected_pages"]]
        relevant_chunk_ids = question.get("relevant_chunk_ids", [])

        print(f"\n{question_id}: {query}")
        print(f"  Expected pages: {expected_pages}")

        if relevant_chunk_ids:
            print(f"  Relevant chunk IDs: {relevant_chunk_ids}")

        hybrid_results = hybrid_search(
            query=query,
            chunks=chunks,
            model=embedding_model,
            bm25_index=bm25_index,
            chunk_embeddings=embeddings,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        if not hybrid_results:
            print("  No hybrid results returned.")

            for k in HYBRID_CUTOFFS:
                metric_totals[f"hybrid_recall@{k}"] += 0.0

            if run_reranker:
                metric_totals[f"reranked_recall@{RERANK_TOP_K}"] += 0.0

            if relevant_chunk_ids:
                chunk_evaluated_questions += 1

                for k in HYBRID_CUTOFFS:
                    chunk_metric_totals[f"hybrid_chunk_recall@{k}"] += 0.0

                if run_reranker:
                    chunk_metric_totals[f"reranked_chunk_recall@{RERANK_TOP_K}"] += 0.0

            continue

        for k in HYBRID_CUTOFFS:
            pages = [int(result["page_number"]) for result in hybrid_results[:k]]
            score = recall_at_k(hybrid_results, expected_pages, k)
            metric_totals[f"hybrid_recall@{k}"] += score

            print(f"  Hybrid top {k} pages: {pages}")
            print(f"  Hybrid Recall@{k}: {score:.2f}")

            if relevant_chunk_ids:
                chunk_score = chunk_recall_at_k(
                    hybrid_results,
                    relevant_chunk_ids,
                    k,
                )
                chunk_metric_totals[f"hybrid_chunk_recall@{k}"] += chunk_score
                print(f"  Hybrid Chunk Recall@{k}: {chunk_score:.2f}")

        if not run_reranker:
            if relevant_chunk_ids:
                chunk_evaluated_questions += 1
            continue

        # The reranker is guaranteed to be loaded when run_reranker is True.
        assert reranker is not None

        reranked_results = rerank_chunks(
            query=query,
            chunks=hybrid_results,
            model=reranker,
            top_k=RERANK_TOP_K,
        )

        if question_id == DIAGNOSTIC_QUESTION_ID:
            diagnose_relevant_chunks(
                query=query,
                hybrid_results=hybrid_results,
                relevant_chunk_ids=relevant_chunk_ids,
                reranker=reranker,
            )

        reranked_pages = [int(result["page_number"]) for result in reranked_results]
        reranked_score = recall_at_k(
            reranked_results,
            expected_pages,
            RERANK_TOP_K,
        )
        metric_totals[f"reranked_recall@{RERANK_TOP_K}"] += reranked_score

        print(f"  Reranked top {RERANK_TOP_K} pages: {reranked_pages}")
        print(f"  Reranked Recall@{RERANK_TOP_K}: {reranked_score:.2f}")

        if relevant_chunk_ids:
            reranked_chunk_score = chunk_recall_at_k(
                reranked_results,
                relevant_chunk_ids,
                RERANK_TOP_K,
            )
            chunk_metric_totals[f"reranked_chunk_recall@{RERANK_TOP_K}"] += reranked_chunk_score

            print(f"  Reranked Chunk Recall@{RERANK_TOP_K}: {reranked_chunk_score:.2f}")

            chunk_evaluated_questions += 1

    question_count = len(golden_set)

    print("\n=== Aggregate page-level results ===")

    for metric_name, total in metric_totals.items():
        average = total / question_count
        print(f"{metric_name}: {average:.3f} ({total:.2f}/{question_count})")

    print("\n=== Aggregate chunk-level results ===")

    if chunk_evaluated_questions == 0:
        print("No questions have relevant_chunk_ids; chunk-level metrics unavailable.")
    else:
        print(f"Questions evaluated for chunk relevance: {chunk_evaluated_questions}")

        for metric_name, total in chunk_metric_totals.items():
            average = total / chunk_evaluated_questions
            print(f"{metric_name}: {average:.3f} ({total:.2f}/{chunk_evaluated_questions})")


if __name__ == "__main__":
    main()

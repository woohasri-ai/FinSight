
import argparse
import json
from collections.abc import Sequence
from math import log2
from pathlib import Path
from typing import Any

import numpy as np

from finsight.evaluation.retrieval_report import (
    build_retrieval_report,
    save_retrieval_report,
)
from finsight.retrieval.bm25 import build_bm25_index, search_bm25
from finsight.retrieval.dense import (
    load_embedding_model,
    load_saved_embeddings,
    search_chunks,
)
from finsight.retrieval.hybrid import hybrid_search
from finsight.retrieval.rerank import load_reranker, rerank_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tcs_annual_report_2025_26_chunks.json"
)
EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tcs_annual_report_2025_26_embeddings.npy"
)
GOLDEN_SET_PATH = (
    PROJECT_ROOT / "data" / "eval" / "retrieval_golden_set.json"
)
REPORTS_DIR = PROJECT_ROOT / "data" / "eval" / "runs"

CANDIDATE_K = 50
HYBRID_CUTOFFS = (5, 10, 50)
RERANK_TOP_K = 10
RANKING_METRIC_K = 10
DIAGNOSTIC_QUESTION_ID = "TCS-RET-003"


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

    retrieved_pages = {
        int(result["page_number"]) for result in results[:k]
    }
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

    retrieved_ids = {
        str(result["chunk_id"])
        for result in results[:k]
        if "chunk_id" in result
    }
    found_ids = retrieved_ids & expected_ids

    return len(found_ids) / len(expected_ids)


def reciprocal_rank(
    results: Sequence[dict[str, Any]],
    relevant_chunk_ids: set[str],
    k: int = RANKING_METRIC_K,
) -> float:
    """Calculate reciprocal rank of the first relevant chunk within top k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_chunk_ids:
        return 0.0

    for rank, result in enumerate(results[:k], start=1):
        if str(result["chunk_id"]) in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    results: Sequence[dict[str, Any]],
    relevant_chunk_ids: set[str],
    k: int = RANKING_METRIC_K,
) -> float:
    """Calculate NDCG@k using binary chunk relevance."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_chunk_ids:
        return 0.0

    dcg = 0.0
    seen_chunk_ids: set[str] = set()

    for rank, result in enumerate(results[:k], start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)

        if chunk_id in relevant_chunk_ids:
            dcg += 1.0 / log2(rank + 1)

    ideal_relevant_count = min(len(relevant_chunk_ids), k)
    ideal_dcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


def load_json(path: Path) -> Any:
    """Load JSON data with a clear error if the file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_relevant_chunk_ranks(
    relevant_chunk_ids: list[str],
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> None:
    """Compare the ranks of labeled relevant chunks across retrievers."""
    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        print("No labeled relevant chunk IDs are available for diagnosis.")
        return

    rankings = {
        "Dense": dense_results,
        "BM25": bm25_results,
        "Hybrid": hybrid_results,
    }

    rank_maps: dict[str, dict[str, int]] = {
        name: {
            str(result["chunk_id"]): rank
            for rank, result in enumerate(results, start=1)
        }
        for name, results in rankings.items()
    }

    print("\nDiagnostic: relevant chunk ranks by retriever")
    print("-" * 110)
    print(
        f"{'Page':<7} {'Chunk ID':<48} "
        f"{'Dense rank':<13} {'BM25 rank':<13} {'Hybrid rank':<13}"
    )
    print("-" * 110)

    for chunk_id in sorted(relevant_ids):
        page_number = next(
            (
                int(result["page_number"])
                for results in rankings.values()
                for result in results
                if str(result["chunk_id"]) == chunk_id
            ),
            None,
        )

        page_display = (
            str(page_number) if page_number is not None else "Unknown"
        )

        rank_displays = []

        for name in ("Dense", "BM25", "Hybrid"):
            rank = rank_maps[name].get(chunk_id)
            rank_displays.append(
                str(rank) if rank is not None else "Not in top 50"
            )

        print(
            f"{page_display:<7} {chunk_id:<48} "
            f"{rank_displays[0]:<13} "
            f"{rank_displays[1]:<13} "
            f"{rank_displays[2]:<13}"
        )

    print("-" * 110)
    print("Ranks are measured within each retriever's returned candidate list.")


def diagnose_relevant_chunks(
    query: str,
    hybrid_results: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    reranker: Any,
) -> None:
    """Print top reranked candidates and identify labeled relevant chunks."""
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
        print(
            "Relevant chunks missing from hybrid candidates: "
            f"{sorted(missing_candidates)}"
        )
    else:
        print("All labeled relevant chunks are present in hybrid candidates.")

    diagnostic_results = rerank_chunks(
        query=query,
        chunks=hybrid_results,
        model=reranker,
        top_k=len(hybrid_results),
    )

    print("\nDiagnostic: top 15 reranked candidates")

    for rank, result in enumerate(diagnostic_results[:15], start=1):
        chunk_id = str(result["chunk_id"])
        relevance = (
            "RELEVANT"
            if chunk_id in relevant_ids
            else "not labeled relevant"
        )

        text_preview = " ".join(str(result["text"]).split())

        if len(text_preview) > 220:
            text_preview = text_preview[:220] + "..."

        print(
            f"\nRank: {rank} | "
            f"Page: {result['page_number']} | "
            f"Chunk: {chunk_id} | "
            f"Reranker score: {result['score']:.4f} | "
            f"{relevance}"
        )
        print(f"Preview: {text_preview}")


def initialize_question_metrics(
    question_id: str,
    relevant_chunk_ids: list[str],
    run_reranker: bool,
) -> dict[str, Any]:
    """Create the per-question report record."""
    metrics: dict[str, Any] = {
        "question_id": question_id,
        "has_relevant_chunk_labels": bool(relevant_chunk_ids),
    }

    for k in HYBRID_CUTOFFS:
        metrics[f"hybrid_recall@{k}"] = 0.0

    if run_reranker:
        metrics[f"reranked_recall@{RERANK_TOP_K}"] = 0.0

    if relevant_chunk_ids:
        for k in HYBRID_CUTOFFS:
            metrics[f"hybrid_chunk_recall@{k}"] = 0.0

        metrics[f"hybrid_mrr@{RANKING_METRIC_K}"] = 0.0
        metrics[f"hybrid_ndcg@{RANKING_METRIC_K}"] = 0.0

        if run_reranker:
            metrics[f"reranked_chunk_recall@{RERANK_TOP_K}"] = 0.0
            metrics[f"reranked_mrr@{RANKING_METRIC_K}"] = 0.0
            metrics[f"reranked_ndcg@{RANKING_METRIC_K}"] = 0.0

    return metrics


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
            f"Chunk/embedding mismatch: {len(chunks)} chunks "
            f"but {len(embeddings)} embeddings."
        )

    print(
        f"Loaded {len(chunks)} chunks and "
        f"{len(golden_set)} evaluation questions."
    )
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
        f"hybrid_recall@{k}": 0.0
        for k in HYBRID_CUTOFFS
    }

    chunk_metric_totals = {
        f"hybrid_chunk_recall@{k}": 0.0
        for k in HYBRID_CUTOFFS
    }

    ranking_metric_totals = {
        f"hybrid_mrr@{RANKING_METRIC_K}": 0.0,
        f"hybrid_ndcg@{RANKING_METRIC_K}": 0.0,
    }

    if run_reranker:
        metric_totals[f"reranked_recall@{RERANK_TOP_K}"] = 0.0
        chunk_metric_totals[
            f"reranked_chunk_recall@{RERANK_TOP_K}"
        ] = 0.0
        ranking_metric_totals[
            f"reranked_mrr@{RANKING_METRIC_K}"
        ] = 0.0
        ranking_metric_totals[
            f"reranked_ndcg@{RANKING_METRIC_K}"
        ] = 0.0

    chunk_evaluated_questions = 0
    per_question_metrics: list[dict[str, Any]] = []

    for question in golden_set:
        question_id = question["id"]
        query = question["question"]
        expected_pages = [
            int(page) for page in question["expected_pages"]
        ]
        relevant_chunk_ids = question.get("relevant_chunk_ids", [])
        relevant_ids = set(relevant_chunk_ids)

        question_metrics = initialize_question_metrics(
            question_id=question_id,
            relevant_chunk_ids=relevant_chunk_ids,
            run_reranker=run_reranker,
        )

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

        if question_id == DIAGNOSTIC_QUESTION_ID:
            dense_results = search_chunks(
                query=query,
                chunks=chunks,
                model=embedding_model,
                top_k=CANDIDATE_K,
                chunk_embeddings=embeddings,
            )
            bm25_results = search_bm25(
                query=query,
                chunks=chunks,
                index=bm25_index,
                top_k=CANDIDATE_K,
            )

            print_relevant_chunk_ranks(
                relevant_chunk_ids=relevant_chunk_ids,
                dense_results=dense_results,
                bm25_results=bm25_results,
                hybrid_results=hybrid_results,
            )

        if not hybrid_results:
            print("  No hybrid results returned.")

            if relevant_chunk_ids:
                chunk_evaluated_questions += 1

            per_question_metrics.append(question_metrics)
            continue

        for k in HYBRID_CUTOFFS:
            pages = [
                int(result["page_number"])
                for result in hybrid_results[:k]
            ]
            score = recall_at_k(
                hybrid_results,
                expected_pages,
                k,
            )

            metric_name = f"hybrid_recall@{k}"
            metric_totals[metric_name] += score
            question_metrics[metric_name] = score

            print(f"  Hybrid top {k} pages: {pages}")
            print(f"  Hybrid Recall@{k}: {score:.2f}")

            if relevant_chunk_ids:
                chunk_score = chunk_recall_at_k(
                    hybrid_results,
                    relevant_chunk_ids,
                    k,
                )
                chunk_metric_name = f"hybrid_chunk_recall@{k}"
                chunk_metric_totals[chunk_metric_name] += chunk_score
                question_metrics[chunk_metric_name] = chunk_score

                print(f"  Hybrid Chunk Recall@{k}: {chunk_score:.2f}")

        if relevant_chunk_ids:
            hybrid_mrr = reciprocal_rank(
                hybrid_results,
                relevant_ids,
                RANKING_METRIC_K,
            )
            hybrid_ndcg = ndcg_at_k(
                hybrid_results,
                relevant_ids,
                RANKING_METRIC_K,
            )

            mrr_name = f"hybrid_mrr@{RANKING_METRIC_K}"
            ndcg_name = f"hybrid_ndcg@{RANKING_METRIC_K}"

            ranking_metric_totals[mrr_name] += hybrid_mrr
            ranking_metric_totals[ndcg_name] += hybrid_ndcg

            question_metrics[mrr_name] = hybrid_mrr
            question_metrics[ndcg_name] = hybrid_ndcg

            print(f"  Hybrid MRR@{RANKING_METRIC_K}: {hybrid_mrr:.3f}")
            print(f"  Hybrid NDCG@{RANKING_METRIC_K}: {hybrid_ndcg:.3f}")

        if run_reranker:
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

            reranked_pages = [
                int(result["page_number"])
                for result in reranked_results
            ]
            reranked_score = recall_at_k(
                reranked_results,
                expected_pages,
                RERANK_TOP_K,
            )

            reranked_recall_name = (
                f"reranked_recall@{RERANK_TOP_K}"
            )
            metric_totals[reranked_recall_name] += reranked_score
            question_metrics[reranked_recall_name] = reranked_score

            print(f"  Reranked top {RERANK_TOP_K} pages: {reranked_pages}")
            print(
                f"  Reranked Recall@{RERANK_TOP_K}: "
                f"{reranked_score:.2f}"
            )

            if relevant_chunk_ids:
                reranked_chunk_score = chunk_recall_at_k(
                    reranked_results,
                    relevant_chunk_ids,
                    RERANK_TOP_K,
                )

                reranked_chunk_name = (
                    f"reranked_chunk_recall@{RERANK_TOP_K}"
                )
                chunk_metric_totals[
                    reranked_chunk_name
                ] += reranked_chunk_score
                question_metrics[
                    reranked_chunk_name
                ] = reranked_chunk_score

                reranked_mrr = reciprocal_rank(
                    reranked_results,
                    relevant_ids,
                    RANKING_METRIC_K,
                )
                reranked_ndcg = ndcg_at_k(
                    reranked_results,
                    relevant_ids,
                    RANKING_METRIC_K,
                )

                reranked_mrr_name = (
                    f"reranked_mrr@{RANKING_METRIC_K}"
                )
                reranked_ndcg_name = (
                    f"reranked_ndcg@{RANKING_METRIC_K}"
                )

                ranking_metric_totals[
                    reranked_mrr_name
                ] += reranked_mrr
                ranking_metric_totals[
                    reranked_ndcg_name
                ] += reranked_ndcg

                question_metrics[reranked_mrr_name] = reranked_mrr
                question_metrics[reranked_ndcg_name] = reranked_ndcg

                print(
                    f"  Reranked Chunk Recall@{RERANK_TOP_K}: "
                    f"{reranked_chunk_score:.2f}"
                )
                print(
                    f"  Reranked MRR@{RANKING_METRIC_K}: "
                    f"{reranked_mrr:.3f}"
                )
                print(
                    f"  Reranked NDCG@{RANKING_METRIC_K}: "
                    f"{reranked_ndcg:.3f}"
                )

        if relevant_chunk_ids:
            chunk_evaluated_questions += 1

        per_question_metrics.append(question_metrics)

    question_count = len(golden_set)

    print("\n=== Aggregate page-level results ===")

    aggregate_metrics: dict[str, Any] = {
        "page_metric_question_count": question_count,
        "chunk_metric_question_count": chunk_evaluated_questions,
    }

    for metric_name, total in metric_totals.items():
        average = total / question_count
        aggregate_metrics[metric_name] = average
        print(f"{metric_name}: {average:.3f} ({total:.2f}/{question_count})")

    print("\n=== Aggregate chunk-level results ===")

    if chunk_evaluated_questions == 0:
        print(
            "No questions have relevant_chunk_ids; "
            "chunk-level metrics unavailable."
        )
    else:
        print(
            "Questions evaluated for chunk relevance: "
            f"{chunk_evaluated_questions}"
        )

        for metric_name, total in chunk_metric_totals.items():
            average = total / chunk_evaluated_questions
            aggregate_metrics[metric_name] = average
            print(
                f"{metric_name}: {average:.3f} "
                f"({total:.2f}/{chunk_evaluated_questions})"
            )

        print("\n=== Aggregate ranking results ===")

        for metric_name, total in ranking_metric_totals.items():
            average = total / chunk_evaluated_questions
            aggregate_metrics[metric_name] = average
            print(
                f"{metric_name}: {average:.3f} "
                f"({total:.2f}/{chunk_evaluated_questions})"
            )

    configuration = {
        "candidate_k": CANDIDATE_K,
        "hybrid_cutoffs": list(HYBRID_CUTOFFS),
        "reranker_enabled": run_reranker,
        "rerank_top_k": RERANK_TOP_K if run_reranker else None,
        "ranking_metric_k": RANKING_METRIC_K,
        "chunk_file": str(CHUNKS_PATH.relative_to(PROJECT_ROOT)),
        "embedding_file": str(EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
        "golden_set_file": str(GOLDEN_SET_PATH.relative_to(PROJECT_ROOT)),
    }

    report = build_retrieval_report(
        benchmark_name="TCS retrieval benchmark",
        chunk_count=len(chunks),
        configuration=configuration,
        aggregate_metrics=aggregate_metrics,
        per_question_metrics=per_question_metrics,
    )

    report_path = save_retrieval_report(report, REPORTS_DIR)

    print("\n=== Evaluation report saved ===")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Run ID: {report['run_id']}")


if __name__ == "__main__":
    main()

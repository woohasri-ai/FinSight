import argparse
import json
from collections import defaultdict
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
from finsight.retrieval.company_scope import resolve_company_symbols
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
    / "multi_company"
    / "tier1_combined_chunks.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_company"
    / "tier1_combined_embeddings.npy"
)

GOLDEN_SET_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "tier1_retrieval_golden_set_40.json"
)

REPORTS_DIR = PROJECT_ROOT / "data" / "eval" / "runs"

CANDIDATE_K = 50
HYBRID_CUTOFFS = (5, 10, 50)
RERANK_TOP_K = 10
RANKING_METRIC_K = 10

DEFAULT_DIAGNOSTIC_QUESTION_ID = "HCLTECH-RET-001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinSight Tier 1 hybrid retrieval and reranking."
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranker loading and inference.",
    )

    parser.add_argument(
        "--diagnostic-question",
        type=str,
        default=None,
        help=(
            "Evaluate only this benchmark question and generate its retrieval "
            "diagnostic. Example: LTM-RET-004"
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def company_from_question_id(question_id: str) -> str:
    return question_id.split("-RET-")[0]


def benchmark_company_scope(
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> set[str] | None:
    """Resolve explicit company mentions, then fall back to the benchmark ID."""
    resolved_symbols = resolve_company_symbols(
        str(question["question"]),
        chunks,
    )

    if resolved_symbols:
        return resolved_symbols

    benchmark_company = company_from_question_id(
        str(question["id"])
    ).upper()

    available_symbols = {
        str(chunk.get("symbol", "")).strip().upper()
        for chunk in chunks
        if str(chunk.get("symbol", "")).strip()
    }

    if benchmark_company in available_symbols:
        return {benchmark_company}

    return None


def validate_benchmark(
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> None:
    chunk_map = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    question_ids = [
        question["id"]
        for question in questions
    ]

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs found in the benchmark.")

    errors = []

    for question in questions:
        for chunk_id in question.get("relevant_chunk_ids", []):
            chunk = chunk_map.get(chunk_id)

            if chunk is None:
                errors.append(
                    f"{question['id']}: missing relevant chunk {chunk_id}"
                )
                continue

            if chunk["page_number"] not in question["expected_pages"]:
                errors.append(
                    f"{question['id']}: page mismatch for {chunk_id}"
                )

            if chunk.get("source_doc") != question["source_doc"]:
                errors.append(
                    f"{question['id']}: source document mismatch for {chunk_id}"
                )

    if errors:
        details = "\n".join(
            f"- {error}"
            for error in errors
        )
        raise ValueError(f"Benchmark validation failed:\n{details}")


def recall_at_k(
    results: list[dict[str, Any]],
    expected_pages: list[int],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    expected = set(expected_pages)

    if not expected:
        return 0.0

    retrieved = {
        int(result["page_number"])
        for result in results[:k]
    }

    return len(retrieved & expected) / len(expected)


def chunk_recall_at_k(
    results: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    expected = set(relevant_chunk_ids)

    if not expected:
        return 0.0

    retrieved = {
        str(result["chunk_id"])
        for result in results[:k]
        if "chunk_id" in result
    }

    return len(retrieved & expected) / len(expected)


def reciprocal_rank(
    results: Sequence[dict[str, Any]],
    relevant_chunk_ids: set[str],
    k: int = RANKING_METRIC_K,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    for rank, result in enumerate(results[:k], start=1):
        if str(result["chunk_id"]) in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    results: Sequence[dict[str, Any]],
    relevant_chunk_ids: set[str],
    k: int = RANKING_METRIC_K,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_chunk_ids:
        return 0.0

    dcg = 0.0
    seen_chunk_ids = set()

    for rank, result in enumerate(results[:k], start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)

        if chunk_id in relevant_chunk_ids:
            dcg += 1.0 / log2(rank + 1)

    ideal_count = min(len(relevant_chunk_ids), k)

    ideal_dcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


def print_relevant_chunk_ranks(
    relevant_chunk_ids: list[str],
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> None:
    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        print("No labeled relevant chunks available for diagnosis.")
        return

    rankings = {
        "Dense": dense_results,
        "BM25": bm25_results,
        "Hybrid": hybrid_results,
    }

    rank_maps = {
        name: {
            str(result["chunk_id"]): rank
            for rank, result in enumerate(results, start=1)
        }
        for name, results in rankings.items()
    }

    print("\nDiagnostic: relevant chunk ranks by retriever")
    print("-" * 110)
    print(
        f"{'Chunk ID':<55}"
        f"{'Dense rank':<15}"
        f"{'BM25 rank':<15}"
        f"{'Hybrid rank':<15}"
    )
    print("-" * 110)

    for chunk_id in sorted(relevant_ids):
        ranks = [
            str(rank_maps[name].get(chunk_id, "Not in top 50"))
            for name in ("Dense", "BM25", "Hybrid")
        ]

        print(
            f"{chunk_id:<55}"
            f"{ranks[0]:<15}"
            f"{ranks[1]:<15}"
            f"{ranks[2]:<15}"
        )

    print("-" * 110)


def diagnose_relevant_chunks(
    question_id: str,
    query: str,
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
) -> None:
    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        return

    lines = [
        "RETRIEVAL DIAGNOSTIC",
        f"Question ID: {question_id}",
        f"Query: {query}",
        "",
        "RELEVANT CHUNKS IN HYBRID CANDIDATES",
    ]

    found_ids = set()

    for rank, result in enumerate(hybrid_results, start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in relevant_ids:
            found_ids.add(chunk_id)

            lines.append(
                f"Hybrid rank: {rank} | "
                f"Page: {result['page_number']} | "
                f"Chunk: {chunk_id} | "
                f"Score: {result['score']:.4f}"
            )

    missing_ids = relevant_ids - found_ids

    if missing_ids:
        lines.append(
            f"Missing from candidates: {sorted(missing_ids)}"
        )
    else:
        lines.append("All labeled relevant chunks are present.")

    # Use the exact same ranking that was used for evaluation metrics.
    # Do not call rerank_chunks() again here.
    lines.extend(
        [
            "",
            "RERANKED POSITIONS OF EXPECTED CHUNKS",
        ]
    )

    relevant_ranks = {}

    for rank, result in enumerate(reranked_results, start=1):
        chunk_id = str(result["chunk_id"])

        if chunk_id in relevant_ids:
            relevant_ranks[chunk_id] = rank

            lines.append(
                f"Reranked position: {rank} | "
                f"Page: {result['page_number']} | "
                f"Chunk: {chunk_id} | "
                f"Score: {result['score']:.4f}"
            )

    for chunk_id in sorted(relevant_ids):
        if chunk_id not in relevant_ranks:
            lines.append(
                f"Not found after reranking: {chunk_id}"
            )

    lines.extend(
        [
            "",
            "TOP 15 RERANKED CANDIDATES",
        ]
    )

    for rank, result in enumerate(reranked_results[:15], start=1):
        chunk_id = str(result["chunk_id"])

        relevance = (
            "RELEVANT"
            if chunk_id in relevant_ids
            else "not labeled relevant"
        )

        preview = " ".join(
            str(result["text"]).split()
        )[:300]

        lines.append(
            f"\nRank: {rank} | "
            f"Page: {result['page_number']} | "
            f"Chunk: {chunk_id} | "
            f"Score: {result['score']:.4f} | "
            f"{relevance}"
        )

        lines.append(f"Preview: {preview}")

    output_path = (
        REPORTS_DIR
        / f"{question_id.lower().replace('-', '_')}_diagnostic.txt"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nDiagnostic saved to: {output_path}")


def initialize_question_metrics(
    question_id: str,
    relevant_chunk_ids: list[str],
    run_reranker: bool,
) -> dict[str, Any]:
    metrics = {
        "question_id": question_id,
        "company": company_from_question_id(question_id),
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


def print_company_breakdown(
    per_question_metrics: list[dict[str, Any]],
    metric_names: list[str],
) -> None:
    grouped = defaultdict(list)

    for metrics in per_question_metrics:
        grouped[metrics["company"]].append(metrics)

    print("\n=== Per-company results ===")
    print("Scores are macro-averages across each company's questions.")

    for company in sorted(grouped):
        company_questions = grouped[company]

        print(
            f"\n{company}: {len(company_questions)} questions"
        )

        for metric_name in metric_names:
            values = [
                question[metric_name]
                for question in company_questions
                if metric_name in question
            ]

            if values:
                average = sum(values) / len(values)
                print(f"  {metric_name}: {average:.3f}")


def main() -> None:
    args = parse_args()
    run_reranker = not args.no_rerank

    diagnostic_question_id = (
        args.diagnostic_question
        or DEFAULT_DIAGNOSTIC_QUESTION_ID
    )

    chunks: list[dict[str, Any]] = load_json(CHUNKS_PATH)
    all_questions: list[dict[str, Any]] = load_json(GOLDEN_SET_PATH)

    if not chunks:
        raise ValueError(
            "The combined chunk file contains no chunks."
        )

    if not all_questions:
        raise ValueError(
            "The retrieval benchmark contains no questions."
        )

    validate_benchmark(chunks, all_questions)

    if args.diagnostic_question:
        matching_questions = [
            question
            for question in all_questions
            if question["id"] == diagnostic_question_id
        ]

        if not matching_questions:
            available_ids = ", ".join(
                question["id"]
                for question in all_questions
            )

            raise ValueError(
                f"Question ID '{diagnostic_question_id}' was not found. "
                f"Available IDs: {available_ids}"
            )

        golden_set = matching_questions

        print(
            f"Single-question diagnostic mode: {diagnostic_question_id}"
        )
    else:
        golden_set = all_questions

    embeddings: np.ndarray = load_saved_embeddings(
        EMBEDDINGS_PATH
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Chunk/embedding mismatch: {len(chunks)} chunks "
            f"but {len(embeddings)} embeddings."
        )

    print(f"Loaded {len(chunks)} combined chunks.")
    print(f"Loaded {len(golden_set)} benchmark questions.")
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
        metric_totals[
            f"reranked_recall@{RERANK_TOP_K}"
        ] = 0.0

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
    per_question_metrics = []

    for question in golden_set:
        question_id = question["id"]
        query = question["question"]

        expected_pages = [
            int(page)
            for page in question["expected_pages"]
        ]

        relevant_chunk_ids = question.get(
            "relevant_chunk_ids",
            [],
        )

        relevant_ids = set(relevant_chunk_ids)

        allowed_symbols = benchmark_company_scope(
            question,
            chunks,
        )

        question_metrics = initialize_question_metrics(
            question_id,
            relevant_chunk_ids,
            run_reranker,
        )

        question_metrics["retrieval_scope_symbols"] = (
            sorted(allowed_symbols)
            if allowed_symbols is not None
            else None
        )

        print(f"\n{question_id}: {query}")
        print(f"  Expected pages: {expected_pages}")

        print(
            "  Company retrieval scope: "
            f"{sorted(allowed_symbols) if allowed_symbols else 'All available companies'}"
        )

        if relevant_chunk_ids:
            print(
                f"  Relevant chunk IDs: {relevant_chunk_ids}"
            )

        hybrid_results = hybrid_search(
            query=query,
            chunks=chunks,
            model=embedding_model,
            bm25_index=bm25_index,
            chunk_embeddings=embeddings,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
            allowed_symbols=allowed_symbols,
        )

        if question_id == diagnostic_question_id:
            dense_results = search_chunks(
                query=query,
                chunks=chunks,
                model=embedding_model,
                top_k=CANDIDATE_K,
                chunk_embeddings=embeddings,
                allowed_symbols=allowed_symbols,
            )

            bm25_results = search_bm25(
                query=query,
                chunks=chunks,
                index=bm25_index,
                top_k=CANDIDATE_K,
                allowed_symbols=allowed_symbols,
            )

            print_relevant_chunk_ranks(
                relevant_chunk_ids,
                dense_results,
                bm25_results,
                hybrid_results,
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

                chunk_name = f"hybrid_chunk_recall@{k}"

                chunk_metric_totals[chunk_name] += chunk_score
                question_metrics[chunk_name] = chunk_score

                print(
                    f"  Hybrid Chunk Recall@{k}: {chunk_score:.2f}"
                )

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

            print(
                f"  Hybrid MRR@{RANKING_METRIC_K}: {hybrid_mrr:.3f}"
            )

            print(
                f"  Hybrid NDCG@{RANKING_METRIC_K}: {hybrid_ndcg:.3f}"
            )

        if run_reranker:
            assert reranker is not None

            # Rerank the complete candidate pool exactly once.
            # Evaluation metrics use the top 10; diagnostics use the
            # same full ranking, preventing duplicate inference.
            reranked_results_full = rerank_chunks(
                query=query,
                chunks=hybrid_results,
                model=reranker,
                top_k=len(hybrid_results),
            )

            reranked_results = reranked_results_full[
                :RERANK_TOP_K
            ]

            if question_id == diagnostic_question_id:
                diagnose_relevant_chunks(
                    question_id,
                    query,
                    hybrid_results,
                    reranked_results_full,
                    relevant_chunk_ids,
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

            reranked_name = (
                f"reranked_recall@{RERANK_TOP_K}"
            )

            metric_totals[reranked_name] += reranked_score
            question_metrics[reranked_name] = reranked_score

            print(
                f"  Reranked top {RERANK_TOP_K} pages: {reranked_pages}"
            )

            print(
                f"  Reranked Recall@{RERANK_TOP_K}: {reranked_score:.2f}"
            )

            if relevant_chunk_ids:
                reranked_chunk_score = chunk_recall_at_k(
                    reranked_results,
                    relevant_chunk_ids,
                    RERANK_TOP_K,
                )

                chunk_name = (
                    f"reranked_chunk_recall@{RERANK_TOP_K}"
                )

                chunk_metric_totals[chunk_name] += (
                    reranked_chunk_score
                )

                question_metrics[chunk_name] = (
                    reranked_chunk_score
                )

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

                mrr_name = (
                    f"reranked_mrr@{RANKING_METRIC_K}"
                )

                ndcg_name = (
                    f"reranked_ndcg@{RANKING_METRIC_K}"
                )

                ranking_metric_totals[mrr_name] += reranked_mrr
                ranking_metric_totals[ndcg_name] += reranked_ndcg

                question_metrics[mrr_name] = reranked_mrr
                question_metrics[ndcg_name] = reranked_ndcg

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

        print(
            f"{metric_name}: {average:.3f} "
            f"({total:.2f}/{question_count})"
        )

    print("\n=== Aggregate chunk-level results ===")

    if chunk_evaluated_questions == 0:
        print("No questions have relevant chunk labels.")

    else:
        print(
            "Questions evaluated for chunk relevance: "
            f"{chunk_evaluated_questions}"
        )

        for metric_name, total in chunk_metric_totals.items():
            average = total / chunk_evaluated_questions

            aggregate_metrics[metric_name] = average

            print(
                f"{metric_name}: "
                f"{average:.3f} "
                f"({total:.2f}/{chunk_evaluated_questions})"
            )

        print("\n=== Aggregate ranking results ===")

        for metric_name, total in ranking_metric_totals.items():
            average = total / chunk_evaluated_questions

            aggregate_metrics[metric_name] = average

            print(
                f"{metric_name}: "
                f"{average:.3f} "
                f"({total:.2f}/{chunk_evaluated_questions})"
            )

    company_metric_names = list(metric_totals)

    if chunk_evaluated_questions:
        company_metric_names.extend(chunk_metric_totals)
        company_metric_names.extend(ranking_metric_totals)

    print_company_breakdown(
        per_question_metrics,
        company_metric_names,
    )

    configuration = {
        "candidate_k": CANDIDATE_K,
        "hybrid_cutoffs": list(HYBRID_CUTOFFS),
        "reranker_enabled": run_reranker,
        "rerank_top_k": (
            RERANK_TOP_K
            if run_reranker
            else None
        ),
        "ranking_metric_k": RANKING_METRIC_K,
        "company_scoped_retrieval": True,
        "company_scope_resolution": (
            "Explicit company mentions, then benchmark question ID fallback"
        ),
        "chunk_file": str(
            CHUNKS_PATH.relative_to(PROJECT_ROOT)
        ),
        "embedding_file": str(
            EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)
        ),
        "golden_set_file": str(
            GOLDEN_SET_PATH.relative_to(PROJECT_ROOT)
        ),
        "company_question_counts": {
            company: sum(
                1
                for question in golden_set
                if company_from_question_id(
                    question["id"]
                ) == company
            )
            for company in sorted(
                {
                    company_from_question_id(question["id"])
                    for question in golden_set
                }
            )
        },
        "diagnostic_question_id": (
            diagnostic_question_id
            if args.diagnostic_question
            else None
        ),
    }

    benchmark_name = (
        f"Single-question diagnostic: {diagnostic_question_id}"
        if args.diagnostic_question
        else "Tier 1 multi-company retrieval benchmark (40 questions)"
    )

    report = build_retrieval_report(
        benchmark_name=benchmark_name,
        chunk_count=len(chunks),
        configuration=configuration,
        aggregate_metrics=aggregate_metrics,
        per_question_metrics=per_question_metrics,
    )

    report_path = save_retrieval_report(
        report,
        REPORTS_DIR,
    )

    print("\n=== Evaluation report saved ===")
    print(
        f"Report: {report_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Run ID: {report['run_id']}")


if __name__ == "__main__":
    main()
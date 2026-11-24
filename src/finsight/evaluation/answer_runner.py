import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from finsight.evaluation.deterministic import (
    evaluate_abstention,
    evaluate_citations,
    evaluate_fact_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "eval" / "answer_golden_set.json"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "runs"


class AnsweringAgent(Protocol):
    """Minimal interface required by the answer benchmark."""

    async def answer(self, question: str) -> Any:
        """Return a generated answer for one question."""


def load_answer_benchmark(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the answer benchmark JSON file."""

    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Answer benchmark file not found: {benchmark_path}")

    with benchmark_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("The answer benchmark must be a non-empty JSON list.")

    required_fields = {
        "id",
        "question",
        "source_doc",
        "expected_answer",
        "required_facts",
        "expected_citations",
        "answerable",
        "topic",
    }

    seen_ids: set[str] = set()

    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark question at index {index} must be an object.")

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Benchmark question at index {index} is missing fields: {sorted(missing_fields)}"
            )

        for field in ("id", "question", "source_doc", "expected_answer", "topic"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(
                    f"Benchmark field '{field}' at index {index} must be a non-empty string."
                )

        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate benchmark question ID: {item['id']}")

        seen_ids.add(item["id"])

        if not isinstance(item["required_facts"], list) or not all(
            isinstance(fact, str) and fact.strip() for fact in item["required_facts"]
        ):
            raise ValueError(f"required_facts must contain non-empty strings: {item['id']}")

        if not isinstance(item["expected_citations"], list):
            raise ValueError(f"expected_citations must be a list: {item['id']}")

        for citation in item["expected_citations"]:
            if not isinstance(citation, dict):
                raise ValueError(f"Each expected citation must be an object: {item['id']}")

            page_number = citation.get("page_number")

            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number <= 0
            ):
                raise ValueError(
                    f"Expected citation page numbers must be positive integers: {item['id']}"
                )

        if not isinstance(item["answerable"], bool):
            raise ValueError(f"answerable must be a boolean: {item['id']}")

        if not item["answerable"] and item["required_facts"]:
            raise ValueError(f"Unanswerable questions must not have required facts: {item['id']}")

    return questions


def _get_aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate answer benchmark metrics."""

    successful_runs = [result for result in results if result["status"] == "completed"]

    answerable_results = [result for result in successful_runs if result["expected_answerable"]]

    unanswerable_results = [
        result for result in successful_runs if not result["expected_answerable"]
    ]

    def average(values: list[float]) -> float | None:
        if not values:
            return None

        return sum(values) / len(values)

    fact_coverages = [result["fact_coverage"]["coverage"] for result in answerable_results]

    citation_precisions = [
        result["citations"]["citation_precision"] for result in answerable_results
    ]

    citation_recalls = [result["citations"]["citation_recall"] for result in answerable_results]

    correctly_handled = sum(1 for result in successful_runs if result["passed"])

    abstention_correct = sum(1 for result in successful_runs if result["abstention"]["correct"])

    correct_abstentions = sum(1 for result in unanswerable_results if result["passed"])

    return {
        "total_questions": len(results),
        "completed_questions": len(successful_runs),
        "failed_questions": len(results) - len(successful_runs),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
        "passed_questions": correctly_handled,
        "overall_pass_rate": (correctly_handled / len(results) if results else 0.0),
        "fact_coverage": average(fact_coverages),
        "citation_precision": average(citation_precisions),
        "citation_recall": average(citation_recalls),
        "abstention_accuracy": (
            abstention_correct / len(successful_runs) if successful_runs else None
        ),
        "correct_abstentions": correct_abstentions,
        "answerable_question_pass_rate": (
            sum(1 for result in answerable_results if result["passed"]) / len(answerable_results)
            if answerable_results
            else None
        ),
        "unanswerable_question_pass_rate": (
            correct_abstentions / len(unanswerable_results) if unanswerable_results else None
        ),
    }


async def run_answer_benchmark(
    agent: AnsweringAgent,
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    benchmark_name: str = "TCS answer benchmark",
) -> tuple[dict[str, Any], Path]:
    """Run the answer benchmark and save its timestamped report."""

    if not isinstance(benchmark_name, str) or not benchmark_name.strip():
        raise ValueError("benchmark_name must not be empty.")

    questions = load_answer_benchmark(benchmark_path)
    per_question_results: list[dict[str, Any]] = []

    for question_data in questions:
        question_id = question_data["id"]

        try:
            generated = await agent.answer(question_data["question"])

            answer = generated.answer
            abstained = generated.abstained
            citations = generated.citations

            if not isinstance(answer, str):
                raise ValueError("Agent returned a non-string answer.")

            if not isinstance(abstained, bool):
                raise ValueError("Agent returned an invalid abstention status.")

            generated_pages = [citation.page_number for citation in citations]

            expected_pages = [
                citation["page_number"] for citation in question_data["expected_citations"]
            ]

            abstention_results = evaluate_abstention(
                expected_answerable=question_data["answerable"],
                actually_answered=not abstained,
            )

            if question_data["answerable"]:
                fact_results = evaluate_fact_coverage(
                    answer=answer,
                    required_facts=question_data["required_facts"],
                )

                citation_results = evaluate_citations(
                    expected_pages=expected_pages,
                    generated_pages=generated_pages,
                )

                document_match = all(
                    citation.document_name == question_data["source_doc"] for citation in citations
                )

                passed = (
                    abstention_results["correct"]
                    and fact_results["coverage"] == 1.0
                    and citation_results["citation_precision"] == 1.0
                    and citation_results["citation_recall"] == 1.0
                    and document_match
                )

            else:
                fact_results = None

                citation_results = evaluate_citations(
                    expected_pages=[],
                    generated_pages=generated_pages,
                )

                document_match = not citations

                passed = abstention_results["correct"] and not citations

            per_question_results.append(
                {
                    "id": question_id,
                    "question": question_data["question"],
                    "topic": question_data["topic"],
                    "source_doc": question_data["source_doc"],
                    "expected_answer": question_data["expected_answer"],
                    "generated_answer": answer,
                    "expected_answerable": question_data["answerable"],
                    "actually_answered": not abstained,
                    "status": "completed",
                    "passed": passed,
                    "fact_coverage": fact_results,
                    "citations": {
                        **citation_results,
                        "expected_document": question_data["source_doc"],
                        "generated_documents": [citation.document_name for citation in citations],
                        "document_match": document_match,
                    },
                    "abstention": abstention_results,
                }
            )

        except Exception as exc:
            per_question_results.append(
                {
                    "id": question_id,
                    "question": question_data["question"],
                    "topic": question_data["topic"],
                    "source_doc": question_data["source_doc"],
                    "expected_answerable": question_data["answerable"],
                    "status": "error",
                    "passed": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    configuration = {
        "agent_class": type(agent).__name__,
        "candidate_k": getattr(agent, "candidate_k", None),
        "hybrid_top_k": getattr(agent, "hybrid_top_k", None),
        "rerank_top_k": getattr(agent, "rerank_top_k", None),
        "llm_model": getattr(getattr(agent, "llm", None), "model", None),
        "benchmark_file": str(benchmark_path),
    }

    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_timestamp": timestamp.isoformat(),
        "benchmark": {
            "name": benchmark_name,
            "question_count": len(questions),
        },
        "configuration": configuration,
        "aggregate_metrics": _get_aggregate_metrics(per_question_results),
        "per_question_metrics": per_question_results,
    }

    json.dumps(report, ensure_ascii=False)

    report_path = save_answer_report(report, output_dir)

    return report, report_path


def save_answer_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save an answer evaluation report without overwriting existing reports."""

    run_id = report.get("run_id")

    if not isinstance(run_id, str) or not re.fullmatch(
        r"\d{8}T\d{12}Z",
        run_id,
    ):
        raise ValueError("Report contains an invalid run_id.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"answer_eval_{run_id}.json"
    temporary_path = output_dir / f".answer_eval_{run_id}.tmp"

    if output_path.exists():
        raise FileExistsError(f"Evaluation report already exists: {output_path}")

    serialized_report = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )

    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            file.write(serialized_report)
            file.write("\n")

        if output_path.exists():
            raise FileExistsError(f"Evaluation report already exists: {output_path}")

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path

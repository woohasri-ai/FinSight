import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finsight.evaluation.answer_runner import (
    load_answer_benchmark,
    run_answer_benchmark,
    save_answer_report,
)


def make_benchmark_question(
    *,
    question_id: str = "TCS-ANS-TEST-001",
    answerable: bool = True,
) -> dict:
    return {
        "id": question_id,
        "question": "What was TCS's total employee strength in FY2026?",
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "expected_answer": "617,437 employees",
        "required_facts": ["Total employees: 617,437"] if answerable else [],
        "expected_citations": [{"page_number": 108}] if answerable else [],
        "answerable": answerable,
        "topic": "employee_strength",
    }


def make_generated_answer(
    *,
    answer: str = "TCS had 617,437 total employees in FY2026.",
    abstained: bool = False,
    page_number: int = 108,
    document_name: str = "tcs_annual_report_2025_26.pdf",
) -> SimpleNamespace:
    citation = SimpleNamespace(
        document_name=document_name,
        page_number=page_number,
        chunk_id="tcs-fy26-page-108",
        evidence_text="Total employees: 617,437",
    )

    return SimpleNamespace(
        answer=answer,
        abstained=abstained,
        citations=[] if abstained else [citation],
    )


class MockAgent:
    candidate_k = 50
    hybrid_top_k = 50
    rerank_top_k = 5

    def __init__(self, generated_answer: SimpleNamespace):
        self.generated_answer = generated_answer
        self.questions: list[str] = []

    async def answer(self, question: str) -> SimpleNamespace:
        self.questions.append(question)
        return self.generated_answer


class FailingAgent:
    async def answer(self, question: str) -> SimpleNamespace:
        raise RuntimeError("Mock agent failure")


def write_benchmark(
    tmp_path: Path,
    questions: list[dict],
) -> Path:
    benchmark_path = tmp_path / "answer_golden_set.json"
    benchmark_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return benchmark_path


def test_load_answer_benchmark_success(tmp_path: Path) -> None:
    benchmark_path = write_benchmark(
        tmp_path,
        [make_benchmark_question()],
    )

    questions = load_answer_benchmark(benchmark_path)

    assert len(questions) == 1
    assert questions[0]["id"] == "TCS-ANS-TEST-001"


def test_load_answer_benchmark_rejects_duplicate_ids(tmp_path: Path) -> None:
    question = make_benchmark_question()
    duplicate = make_benchmark_question()

    benchmark_path = write_benchmark(
        tmp_path,
        [question, duplicate],
    )

    with pytest.raises(ValueError, match="Duplicate benchmark question ID"):
        load_answer_benchmark(benchmark_path)


def test_load_answer_benchmark_rejects_unanswerable_question_with_facts(
    tmp_path: Path,
) -> None:
    question = make_benchmark_question(answerable=False)
    question["required_facts"] = ["Total employees: 617,437"]

    benchmark_path = write_benchmark(
        tmp_path,
        [question],
    )

    with pytest.raises(ValueError, match="Unanswerable questions must not have required facts"):
        load_answer_benchmark(benchmark_path)


@pytest.mark.asyncio
async def test_answer_benchmark_passes_correct_answerable_response(
    tmp_path: Path,
) -> None:
    benchmark_path = write_benchmark(
        tmp_path,
        [make_benchmark_question()],
    )
    output_dir = tmp_path / "reports"

    agent = MockAgent(make_generated_answer())

    report, report_path = await run_answer_benchmark(
        agent,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
    )

    result = report["per_question_metrics"][0]

    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["fact_coverage"]["coverage"] == 1.0
    assert result["citations"]["citation_recall"] == 1.0
    assert result["citations"]["document_match"] is True
    assert report["aggregate_metrics"]["overall_pass_rate"] == 1.0
    assert report_path.is_file()
    assert len(agent.questions) == 1


@pytest.mark.asyncio
async def test_answer_benchmark_passes_correct_abstention(
    tmp_path: Path,
) -> None:
    benchmark_path = write_benchmark(
        tmp_path,
        [make_benchmark_question(answerable=False)],
    )
    output_dir = tmp_path / "reports"

    generated = SimpleNamespace(
        answer="The annual report does not provide this information.",
        abstained=True,
        citations=[],
    )
    agent = MockAgent(generated)

    report, _ = await run_answer_benchmark(
        agent,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
    )

    result = report["per_question_metrics"][0]

    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["fact_coverage"] is None
    assert result["citations"]["document_match"] is True
    assert report["aggregate_metrics"]["correct_abstentions"] == 1


@pytest.mark.asyncio
async def test_answer_benchmark_fails_when_expected_citation_is_missing(
    tmp_path: Path,
) -> None:
    benchmark_path = write_benchmark(
        tmp_path,
        [make_benchmark_question()],
    )
    output_dir = tmp_path / "reports"

    generated = SimpleNamespace(
        answer="TCS had 617,437 total employees in FY2026.",
        abstained=False,
        citations=[],
    )
    agent = MockAgent(generated)

    report, _ = await run_answer_benchmark(
        agent,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
    )

    result = report["per_question_metrics"][0]

    assert result["passed"] is False
    assert result["citations"]["citation_recall"] == 0.0


@pytest.mark.asyncio
async def test_answer_benchmark_records_agent_errors(
    tmp_path: Path,
) -> None:
    benchmark_path = write_benchmark(
        tmp_path,
        [make_benchmark_question()],
    )
    output_dir = tmp_path / "reports"

    report, report_path = await run_answer_benchmark(
        FailingAgent(),
        benchmark_path=benchmark_path,
        output_dir=output_dir,
    )

    result = report["per_question_metrics"][0]

    assert result["status"] == "error"
    assert result["passed"] is False
    assert result["error"]["type"] == "RuntimeError"
    assert result["error"]["message"] == "Mock agent failure"
    assert report["aggregate_metrics"]["failed_questions"] == 1
    assert report_path.is_file()


def test_save_answer_report_preserves_unicode_and_prevents_overwrite(
    tmp_path: Path,
) -> None:
    report = {
        "run_id": "20261102T123456123456Z",
        "answer": "Revenue was ₹267,021 crore.",
    }

    report_path = save_answer_report(report, tmp_path)

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "₹267,021" in saved_report["answer"]

    with pytest.raises(FileExistsError):
        save_answer_report(report, tmp_path)

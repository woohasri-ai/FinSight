import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from finsight.evaluation.retrieval_report import (
    build_retrieval_report,
    save_retrieval_report,
)


@pytest.fixture
def sample_report():
    return build_retrieval_report(
        benchmark_name="TCS retrieval benchmark",
        chunk_count=642,
        configuration={
            "candidate_k": 50,
            "hybrid_cutoffs": [5, 10, 50],
            "reranker_enabled": True,
            "rerank_top_k": 10,
        },
        aggregate_metrics={
            "hybrid_recall@5": 0.75,
            "hybrid_mrr@10": 0.70,
        },
        per_question_metrics=[
            {
                "question_id": "TCS-RET-001",
                "hybrid_recall@5": 1.0,
            },
            {
                "question_id": "TCS-RET-002",
                "hybrid_recall@5": 0.5,
            },
        ],
        run_timestamp=datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC),
    )


def test_report_contains_run_metadata(sample_report):
    assert sample_report["schema_version"] == "1.0"
    assert sample_report["run_id"] == "20261101T103000000000Z"
    assert sample_report["run_timestamp"] == "2026-11-01T10:30:00+00:00"


def test_report_derives_question_count_from_question_results(sample_report):
    assert sample_report["benchmark"]["question_count"] == 2
    assert sample_report["benchmark"]["chunk_count"] == 642


def test_report_preserves_configuration_and_metrics(sample_report):
    assert sample_report["configuration"]["candidate_k"] == 50
    assert sample_report["aggregate_metrics"]["hybrid_recall@5"] == 0.75
    assert len(sample_report["per_question_metrics"]) == 2


def test_report_rejects_empty_benchmark_name():
    with pytest.raises(ValueError, match="benchmark_name"):
        build_retrieval_report(
            benchmark_name=" ",
            chunk_count=10,
            configuration={},
            aggregate_metrics={},
            per_question_metrics=[],
        )


def test_report_rejects_negative_chunk_count():
    with pytest.raises(ValueError, match="chunk_count"):
        build_retrieval_report(
            benchmark_name="TCS benchmark",
            chunk_count=-1,
            configuration={},
            aggregate_metrics={},
            per_question_metrics=[],
        )


def test_report_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_retrieval_report(
            benchmark_name="TCS benchmark",
            chunk_count=10,
            configuration={},
            aggregate_metrics={},
            per_question_metrics=[],
            run_timestamp=datetime(2026, 11, 1, 10, 30),
        )


def test_report_rejects_non_json_serializable_values():
    with pytest.raises(TypeError):
        build_retrieval_report(
            benchmark_name="TCS benchmark",
            chunk_count=10,
            configuration={"unsupported": object()},
            aggregate_metrics={},
            per_question_metrics=[],
        )


def test_save_report_creates_timestamped_json_file(
    sample_report,
    tmp_path: Path,
):
    output_path = save_retrieval_report(sample_report, tmp_path)

    assert output_path.exists()
    assert output_path.name == "retrieval_eval_20261101T103000000000Z.json"

    saved_report = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_report == sample_report


def test_save_report_creates_missing_output_directory(
    sample_report,
    tmp_path: Path,
):
    output_dir = tmp_path / "nested" / "reports"

    output_path = save_retrieval_report(sample_report, output_dir)

    assert output_path.exists()
    assert output_path.parent == output_dir


def test_save_report_does_not_overwrite_existing_report(
    sample_report,
    tmp_path: Path,
):
    output_path = save_retrieval_report(sample_report, tmp_path)
    original_contents = output_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_retrieval_report(sample_report, tmp_path)

    assert output_path.read_text(encoding="utf-8") == original_contents


def test_save_report_rejects_invalid_run_id(tmp_path: Path):
    with pytest.raises(ValueError, match="run_id"):
        save_retrieval_report(
            {"run_id": "../invalid"},
            tmp_path,
        )

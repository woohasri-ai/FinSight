import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_retrieval_report(
    *,
    benchmark_name: str,
    chunk_count: int,
    configuration: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    per_question_metrics: list[dict[str, Any]],
    run_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report for one retrieval evaluation run."""

    if not benchmark_name.strip():
        raise ValueError("benchmark_name must not be empty.")

    if chunk_count < 0:
        raise ValueError("chunk_count must not be negative.")

    timestamp = run_timestamp or datetime.now(UTC)

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware.")

    timestamp = timestamp.astimezone(UTC)

    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_timestamp": timestamp.isoformat(),
        "benchmark": {
            "name": benchmark_name,
            "question_count": len(per_question_metrics),
            "chunk_count": chunk_count,
        },
        "configuration": configuration,
        "aggregate_metrics": aggregate_metrics,
        "per_question_metrics": per_question_metrics,
    }

    # Fail early if a caller supplies values that JSON cannot serialize.
    json.dumps(report, ensure_ascii=False)

    return report


def save_retrieval_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save one report using its unique run ID in the filename."""

    run_id = report.get("run_id")

    if not isinstance(run_id, str) or not re.fullmatch(
        r"\d{8}T\d{12}Z",
        run_id,
    ):
        raise ValueError("Report contains an invalid run_id.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"retrieval_eval_{run_id}.json"
    temporary_path = output_dir / f".retrieval_eval_{run_id}.tmp"

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

        # Avoid replacing an existing report.
        if output_path.exists():
            raise FileExistsError(f"Evaluation report already exists: {output_path}")

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path

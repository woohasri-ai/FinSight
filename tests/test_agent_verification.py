from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from finsight.agent.pipeline import FinSightAgent
from finsight.generation.schemas import GeneratedAnswer
from finsight.sql.database import get_connection, initialize_database
from finsight.sql.repository import insert_verified_fact


def create_agent(db_path, monkeypatch):
    """Create an agent with mocked retrieval and generation dependencies."""

    hybrid_mock = MagicMock()
    rerank_mock = MagicMock()

    generation_mock = AsyncMock(
        return_value=GeneratedAnswer(
            answer="TCS revenue was 275,000 INR crore.",
            citations=[],
            abstained=False,
        )
    )

    monkeypatch.setattr(
        "finsight.agent.pipeline.hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        "finsight.agent.pipeline.rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        "finsight.agent.pipeline.generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=[{"text": "Annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    return agent, hybrid_mock, rerank_mock, generation_mock


@pytest.mark.parametrize(
    (
        "company",
        "fiscal_year",
        "metric",
        "value",
        "question",
        "display_value",
    ),
    [
        (
            "TCS",
            "FY2026",
            "revenue",
            275000,
            "What was TCS revenue in FY2026?",
            "275,000",
        ),
        (
            "Infosys",
            "FY2025",
            "operating_profit",
            12345,
            "What was Infosys operating profit in FY2025?",
            "12,345",
        ),
    ],
)
@pytest.mark.asyncio
async def test_agent_abstains_when_only_unverified_financial_fact_exists(
    tmp_path,
    monkeypatch,
    company,
    fiscal_year,
    metric,
    value,
    question,
    display_value,
):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path)

    connection = get_connection(db_path)

    try:
        insert_verified_fact(
            connection,
            company=company,
            fiscal_year=fiscal_year,
            metric=metric,
            category=None,
            value=value,
            unit="INR crore",
            source_doc=f"{company.lower()}_annual_report.pdf",
            source_page=50,
            source_excerpt="Illustrative unverified financial record.",
            verification_status="unverified",
        )
        connection.commit()
    finally:
        connection.close()

    agent, hybrid_mock, rerank_mock, generation_mock = create_agent(
        db_path,
        monkeypatch,
    )

    result = await agent.answer(question)

    assert result.abstained is True
    assert company in result.answer
    assert fiscal_year in result.answer
    assert metric.replace("_", " ") in result.answer.lower()
    assert "verified" in result.answer.lower()

    assert display_value not in result.answer
    assert str(value) not in result.answer
    assert result.citations == []

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_selects_latest_unverified_year_when_question_omits_year(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path)

    connection = get_connection(db_path)

    try:
        for fiscal_year, value in (
            ("FY2025", 250000),
            ("FY2026", 275000),
        ):
            insert_verified_fact(
                connection,
                company="TCS",
                fiscal_year=fiscal_year,
                metric="revenue",
                category=None,
                value=value,
                unit="INR crore",
                source_doc=f"tcs_annual_report_{fiscal_year}.pdf",
                source_page=50,
                source_excerpt="Illustrative unverified revenue record.",
                verification_status="unverified",
            )

        connection.commit()
    finally:
        connection.close()

    agent, hybrid_mock, rerank_mock, generation_mock = create_agent(
        db_path,
        monkeypatch,
    )

    result = await agent.answer("What was TCS revenue?")

    assert result.abstained is True
    assert "TCS" in result.answer
    assert "revenue" in result.answer.lower()
    assert "FY2026" in result.answer
    assert "FY2025" not in result.answer

    assert "250,000" not in result.answer
    assert "275,000" not in result.answer
    assert "250000" not in result.answer
    assert "275000" not in result.answer

    assert result.citations == []

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.parametrize(
    "unverified_category",
    [
        "permanent",
        "other_than_permanent",
        "total",
    ],
)
@pytest.mark.asyncio
async def test_agent_abstains_when_any_employee_count_category_is_unverified(
    tmp_path,
    monkeypatch,
    unverified_category,
):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path)

    employee_facts = [
        {"category": "permanent", "value": 584519},
        {"category": "other_than_permanent", "value": 32918},
        {"category": "total", "value": 617437},
    ]

    connection = get_connection(db_path)

    try:
        for fact in employee_facts:
            verification_status = (
                "unverified" if fact["category"] == unverified_category else "verified"
            )

            verification_metadata = {}

            if verification_status == "verified":
                verification_metadata = {
                    "verified_at": datetime.now(UTC).isoformat(),
                    "verified_by": "test_reviewer",
                }

            insert_verified_fact(
                connection,
                company="TCS",
                fiscal_year="FY2026",
                metric="employee_count",
                category=fact["category"],
                value=fact["value"],
                unit="employees",
                source_doc="tcs_annual_report_fy2026.pdf",
                source_page=108,
                source_excerpt="Illustrative employee-count record.",
                verification_status=verification_status,
                **verification_metadata,
            )

        connection.commit()
    finally:
        connection.close()

    agent, hybrid_mock, rerank_mock, generation_mock = create_agent(
        db_path,
        monkeypatch,
    )

    result = await agent.answer("What was the total employee count of TCS in FY2026?")

    assert result.abstained is True
    assert "TCS" in result.answer
    assert "FY2026" in result.answer
    assert "employee" in result.answer.lower()
    assert "verified" in result.answer.lower()

    for fact in employee_facts:
        assert str(fact["value"]) not in result.answer
        assert f"{fact['value']:,}" not in result.answer

    assert result.citations == []

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.parametrize(
    "missing_category",
    [
        "permanent",
        "other_than_permanent",
        "total",
    ],
)
@pytest.mark.asyncio
async def test_agent_abstains_when_employee_count_category_is_missing(
    tmp_path,
    monkeypatch,
    missing_category,
):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path)

    employee_facts = [
        {"category": "permanent", "value": 584519},
        {"category": "other_than_permanent", "value": 32918},
        {"category": "total", "value": 617437},
    ]

    connection = get_connection(db_path)

    try:
        for fact in employee_facts:
            if fact["category"] == missing_category:
                continue

            insert_verified_fact(
                connection,
                company="TCS",
                fiscal_year="FY2026",
                metric="employee_count",
                category=fact["category"],
                value=fact["value"],
                unit="employees",
                source_doc="tcs_annual_report_fy2026.pdf",
                source_page=108,
                source_excerpt="Illustrative employee-count record.",
                verification_status="verified",
                verified_at=datetime.now(UTC).isoformat(),
                verified_by="test_reviewer",
            )

        connection.commit()
    finally:
        connection.close()

    agent, hybrid_mock, rerank_mock, generation_mock = create_agent(
        db_path,
        monkeypatch,
    )

    result = await agent.answer("What was the total employee count of TCS in FY2026?")

    assert result.abstained is True
    assert "TCS" in result.answer
    assert "FY2026" in result.answer
    assert "employee" in result.answer.lower()
    assert "verified" in result.answer.lower()

    for fact in employee_facts:
        assert str(fact["value"]) not in result.answer
        assert f"{fact['value']:,}" not in result.answer

    assert result.citations == []

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()

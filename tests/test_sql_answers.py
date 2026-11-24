import pytest

from finsight.sql.answers import (
    build_employee_count_answer,
    build_employee_count_answer_for_question,
)
from finsight.sql.database import get_connection
from finsight.sql.seeding import seed_verified_facts


def test_build_employee_count_answer_returns_verified_answer(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        result = build_employee_count_answer(
            connection,
            company="TCS",
            fiscal_year="FY2026",
        )
    finally:
        connection.close()

    assert result is not None
    assert result.abstained is False

    assert "617,437" in result.answer
    assert "584,519" in result.answer
    assert "32,918" in result.answer

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.document_name == "tcs_annual_report_2025_26.pdf"
    assert citation.page_number == 108
    assert citation.chunk_id == "tcs_annual_report_2025_26.pdf:page-108:chunk-1"
    assert "6,17,437" in citation.evidence_text


def test_build_employee_count_answer_returns_none_when_facts_missing(
    tmp_path,
):
    from finsight.sql.database import initialize_database

    db_path = tmp_path / "finsight.db"

    initialize_database(db_path)

    connection = get_connection(db_path)

    try:
        result = build_employee_count_answer(
            connection,
            company="INFOSYS",
            fiscal_year="FY2026",
        )
    finally:
        connection.close()

    assert result is None


def test_build_employee_count_answer_rejects_inconsistent_totals(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        connection.execute(
            """
            UPDATE verified_facts
            SET value = ?
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            AND category = ?
            """,
            (600000, "TCS", "FY2026", "employee_count", "total"),
        )

        connection.commit()

        with pytest.raises(
            ValueError,
            match="employee counts are inconsistent",
        ):
            build_employee_count_answer(
                connection,
                company="TCS",
                fiscal_year="FY2026",
            )

    finally:
        connection.close()


def test_build_employee_count_answer_rejects_mismatched_citations(
    tmp_path,
):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        connection.execute(
            """
            UPDATE verified_facts
            SET chunk_id = ?
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            AND category = ?
            """,
            (
                "incorrect_chunk_id",
                "TCS",
                "FY2026",
                "employee_count",
                "other_than_permanent",
            ),
        )

        connection.commit()

        with pytest.raises(
            ValueError,
            match="do not share the same source citation",
        ):
            build_employee_count_answer(
                connection,
                company="TCS",
                fiscal_year="FY2026",
            )

    finally:
        connection.close()


def test_employee_count_question_router_answers_tcs_question(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        result = build_employee_count_answer_for_question(
            connection,
            "What was TCS's total employee strength in FY2026?",
        )
    finally:
        connection.close()

    assert result is not None
    assert result.abstained is False
    assert "617,437" in result.answer
    assert "584,519" in result.answer
    assert "32,918" in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 108


def test_employee_count_router_answers_permanent_category(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        result = build_employee_count_answer_for_question(
            connection,
            "How many permanent employees did TCS have in FY2026?",
        )
    finally:
        connection.close()

    assert result is not None
    assert result.abstained is False
    assert "584,519" in result.answer
    assert "617,437" not in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 108


def test_employee_count_router_answers_other_than_permanent_category(
    tmp_path,
):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        result = build_employee_count_answer_for_question(
            connection,
            "How many other than permanent employees did TCS have in FY2026?",
        )
    finally:
        connection.close()

    assert result is not None
    assert result.abstained is False
    assert "32,918" in result.answer
    assert "584,519" not in result.answer
    assert "617,437" not in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 108

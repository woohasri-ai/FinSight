import pytest

from finsight.sql.database import get_connection, initialize_database
from finsight.sql.repository import insert_verified_fact
from finsight.sql.seeding import seed_verified_facts


def test_seed_verified_facts_inserts_three_records(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        results = connection.execute(
            """
            SELECT category, value
            FROM verified_facts
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            """,
            ("TCS", "FY2026", "employee_count"),
        ).fetchall()
    finally:
        connection.close()

    actual_facts = {row["category"]: row["value"] for row in results}

    assert len(actual_facts) == 3
    assert actual_facts["permanent"] == 584519
    assert actual_facts["other_than_permanent"] == 32918
    assert actual_facts["total"] == 617437


def test_seed_verified_facts_prevents_duplicates(tmp_path):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)
    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM verified_facts
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            """,
            ("TCS", "FY2026", "employee_count"),
        ).fetchone()[0]
    finally:
        connection.close()

    assert count == 3


def test_seed_verified_facts_rejects_conflicting_records(tmp_path):
    db_path = tmp_path / "finsight.db"

    initialize_database(db_path)

    connection = get_connection(db_path)

    try:
        insert_verified_fact(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            category="other_than_permanent",
            value=30000,
            unit="employees",
            source_doc="tcs_annual_report_2025_26.pdf",
            source_page=129,
            source_excerpt="Other than permanent: 30,000",
        )

        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        ValueError,
        match="Conflicting existing fact",
    ):
        seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        results = connection.execute(
            """
            SELECT category, value
            FROM verified_facts
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            """,
            ("TCS", "FY2026", "employee_count"),
        ).fetchall()
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["category"] == "other_than_permanent"
    assert results[0]["value"] == 30000

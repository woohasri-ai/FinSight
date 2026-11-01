import sqlite3

import pytest

from finsight.sql.database import get_connection, initialize_database


def test_initialize_database_creates_verified_facts_table(tmp_path):
    db_path = tmp_path / "finsight.db"

    initialize_database(db_path=db_path)

    assert db_path.is_file()

    connection = sqlite3.connect(db_path)

    try:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'verified_facts'
            """
        ).fetchone()
    finally:
        connection.close()

    assert result is not None


def test_initialize_database_creates_indexes(tmp_path):
    db_path = tmp_path / "finsight.db"

    initialize_database(db_path=db_path)

    connection = sqlite3.connect(db_path)

    try:
        indexes = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
            AND name LIKE 'idx_verified_facts_%'
            """
        ).fetchall()
    finally:
        connection.close()

    index_names = {row[0] for row in indexes}

    assert "idx_verified_facts_lookup" in index_names
    assert "idx_verified_facts_source" in index_names


def test_get_connection_creates_directory_and_supports_named_rows(tmp_path):
    db_path = tmp_path / "nested" / "data" / "finsight.db"

    connection = get_connection(db_path)

    try:
        assert db_path.is_file()

        result = connection.execute("SELECT 617437 AS employee_count").fetchone()

        assert result["employee_count"] == 617437

    finally:
        connection.close()


def test_database_stores_and_retrieves_verified_fact(tmp_path):
    db_path = tmp_path / "finsight.db"

    initialize_database(db_path=db_path)

    connection = get_connection(db_path)

    try:
        connection.execute(
            """
            INSERT INTO verified_facts (
                company,
                fiscal_year,
                metric,
                category,
                value,
                unit,
                source_doc,
                source_page,
                source_excerpt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TCS",
                "FY2026",
                "employee_count",
                "permanent",
                584519,
                "employees",
                "tcs_annual_report_2025_26.pdf",
                129,
                "Permanent employees: 5,84,519",
            ),
        )

        connection.commit()

        result = connection.execute(
            """
            SELECT *
            FROM verified_facts
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            AND category = ?
            """,
            ("TCS", "FY2026", "employee_count", "permanent"),
        ).fetchone()

    finally:
        connection.close()

    assert result is not None
    assert result["company"] == "TCS"
    assert result["fiscal_year"] == "FY2026"
    assert result["value"] == 584519
    assert result["category"] == "permanent"
    assert result["source_page"] == 129


def test_database_rejects_invalid_source_page(tmp_path):
    db_path = tmp_path / "finsight.db"

    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO verified_facts (
                    company,
                    fiscal_year,
                    metric,
                    category,
                    value,
                    unit,
                    source_doc,
                    source_page,
                    source_excerpt
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "TCS",
                    "FY2026",
                    "employee_count",
                    "permanent",
                    584519,
                    "employees",
                    "tcs_annual_report_2025_26.pdf",
                    0,
                    "Permanent employees: 5,84,519",
                ),
            )
    finally:
        connection.close()


def test_initialize_database_migrates_existing_table_without_losing_data(
    tmp_path,
):
    db_path = tmp_path / "legacy_finsight.db"

    # Create an older database schema without chunk_id.
    connection = get_connection(db_path)

    connection.execute(
        """
        CREATE TABLE verified_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            fiscal_year TEXT NOT NULL,
            metric TEXT NOT NULL,
            category TEXT,
            value NUMERIC NOT NULL,
            unit TEXT NOT NULL,
            source_doc TEXT NOT NULL,
            source_page INTEGER NOT NULL,
            source_excerpt TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        INSERT INTO verified_facts (
            company,
            fiscal_year,
            metric,
            category,
            value,
            unit,
            source_doc,
            source_page,
            source_excerpt
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "TCS",
            "FY2026",
            "employee_count",
            "permanent",
            584519,
            "employees",
            "tcs_annual_report_2025_26.pdf",
            129,
            "Permanent 5,84,519",
        ),
    )

    connection.commit()
    connection.close()

    # Initialize should migrate the existing database.
    initialize_database(db_path=db_path)

    connection = get_connection(db_path)

    try:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(verified_facts)").fetchall()
        ]

        result = connection.execute(
            "SELECT * FROM verified_facts WHERE company = ?",
            ("TCS",),
        ).fetchone()

    finally:
        connection.close()

    assert columns.count("chunk_id") == 1

    assert result is not None
    assert result["company"] == "TCS"
    assert result["value"] == 584519
    assert result["source_page"] == 129
    assert result["chunk_id"] == ""

    # Running initialization again must remain safe.
    initialize_database(db_path=db_path)

    connection = get_connection(db_path)

    try:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(verified_facts)").fetchall()
        ]
    finally:
        connection.close()

    assert columns.count("chunk_id") == 1

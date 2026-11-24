import pytest

from finsight.sql.database import get_connection, initialize_database
from finsight.sql.repository import get_verified_facts, insert_verified_fact


def insert_test_fact(connection, **overrides):
    fact = {
        "company": "TCS",
        "fiscal_year": "FY2026",
        "metric": "employee_count",
        "category": "permanent",
        "value": 584519,
        "unit": "employees",
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "source_page": 108,
        "source_excerpt": "Permanent (D): 5,84,519",
    }
    fact.update(overrides)
    return insert_verified_fact(connection, **fact)


def test_insert_verified_fact_returns_record_id(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        fact_id = insert_test_fact(connection)
        connection.commit()

        result = connection.execute(
            "SELECT * FROM verified_facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
    finally:
        connection.close()

    assert fact_id == 1
    assert result is not None
    assert result["company"] == "TCS"
    assert result["value"] == 584519
    assert result["category"] == "permanent"
    assert result["source_page"] == 108
    assert result["verification_status"] == "unverified"
    assert result["verified_at"] is None
    assert result["verified_by"] is None


def test_insert_verified_fact_does_not_commit_automatically(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        insert_test_fact(connection)
        connection.rollback()

        result = connection.execute("SELECT COUNT(*) FROM verified_facts").fetchone()
    finally:
        connection.close()

    assert result[0] == 0


def test_insert_verified_fact_accepts_verification_metadata(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        fact_id = insert_test_fact(
            connection,
            verification_status="verified",
            verified_at="2026-11-01T10:00:00+00:00",
            verified_by="manual_report_review",
        )
        connection.commit()

        result = connection.execute(
            "SELECT * FROM verified_facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
    finally:
        connection.close()

    assert result["verification_status"] == "verified"
    assert result["verified_at"] == "2026-11-01T10:00:00+00:00"
    assert result["verified_by"] == "manual_report_review"


def test_insert_verified_fact_rejects_invalid_verification_status(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        with pytest.raises(ValueError, match="verification_status"):
            insert_test_fact(
                connection,
                verification_status="pending",
            )
    finally:
        connection.close()


def test_insert_verified_fact_requires_verification_metadata(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        with pytest.raises(ValueError, match="verified_at and verified_by"):
            insert_test_fact(
                connection,
                verification_status="verified",
                verified_at="2026-11-01T10:00:00+00:00",
            )
    finally:
        connection.close()


def test_insert_unverified_fact_rejects_verification_metadata(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        with pytest.raises(ValueError, match="cannot have verified_at"):
            insert_test_fact(
                connection,
                verified_by="manual_report_review",
            )
    finally:
        connection.close()


def test_get_verified_facts_returns_verified_facts_only_by_default(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        insert_test_fact(connection, category="permanent")
        insert_test_fact(
            connection,
            category="other_than_permanent",
            value=32918,
            verification_status="verified",
            verified_at="2026-11-01T10:00:00+00:00",
            verified_by="manual_report_review",
        )
        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["category"] == "other_than_permanent"
    assert results[0]["verification_status"] == "verified"


def test_get_verified_facts_can_include_unverified_facts(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        insert_test_fact(connection, category="permanent")
        insert_test_fact(
            connection,
            category="other_than_permanent",
            value=32918,
            verification_status="verified",
            verified_at="2026-11-01T10:00:00+00:00",
            verified_by="manual_report_review",
        )
        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            include_unverified=True,
        )
    finally:
        connection.close()

    assert len(results) == 2
    statuses = {fact["verification_status"] for fact in results}
    assert statuses == {"verified", "unverified"}


def test_get_verified_facts_returns_all_verified_employee_categories(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    facts = [
        ("permanent", 584519),
        ("other_than_permanent", 32918),
        ("total", 617437),
    ]

    try:
        for category, value in facts:
            insert_test_fact(
                connection,
                category=category,
                value=value,
                verification_status="verified",
                verified_at="2026-11-01T10:00:00+00:00",
                verified_by="manual_report_review",
            )

        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
        )
    finally:
        connection.close()

    assert len(results) == 3
    retrieved_facts = {fact["category"]: fact["value"] for fact in results}

    assert retrieved_facts["permanent"] == 584519
    assert retrieved_facts["other_than_permanent"] == 32918
    assert retrieved_facts["total"] == 617437


def test_get_verified_facts_filters_by_category(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        for category, value in [
            ("permanent", 584519),
            ("other_than_permanent", 32918),
            ("total", 617437),
        ]:
            insert_test_fact(
                connection,
                category=category,
                value=value,
                verification_status="verified",
                verified_at="2026-11-01T10:00:00+00:00",
                verified_by="manual_report_review",
            )

        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            category="permanent",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["category"] == "permanent"
    assert results[0]["value"] == 584519


def test_get_verified_facts_returns_empty_list_when_not_found(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="revenue",
        )
    finally:
        connection.close()

    assert results == []


def test_get_verified_facts_filters_by_fiscal_year(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        for fiscal_year, value in [
            ("FY2025", 607979),
            ("FY2026", 584519),
        ]:
            insert_test_fact(
                connection,
                fiscal_year=fiscal_year,
                value=value,
                verification_status="verified",
                verified_at="2026-11-01T10:00:00+00:00",
                verified_by="manual_report_review",
            )

        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            category="permanent",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["fiscal_year"] == "FY2026"
    assert results[0]["value"] == 584519


def test_get_verified_facts_filters_by_company(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        for company, value in [
            ("TCS", 584519),
            ("INFOSYS", 337000),
        ]:
            insert_test_fact(
                connection,
                company=company,
                value=value,
                verification_status="verified",
                verified_at="2026-11-01T10:00:00+00:00",
                verified_by="manual_report_review",
            )

        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            category="permanent",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["company"] == "TCS"
    assert results[0]["value"] == 584519


def test_get_verified_facts_filters_by_metric(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    try:
        insert_test_fact(
            connection,
            metric="employee_count",
            value=584519,
            verification_status="verified",
            verified_at="2026-11-01T10:00:00+00:00",
            verified_by="manual_report_review",
        )
        insert_test_fact(
            connection,
            metric="revenue",
            value=267021,
            unit="INR crore",
            verification_status="verified",
            verified_at="2026-11-01T10:00:00+00:00",
            verified_by="manual_report_review",
        )

        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["metric"] == "employee_count"
    assert results[0]["value"] == 584519


def test_get_verified_facts_preserves_source_and_verification_metadata(tmp_path):
    db_path = tmp_path / "finsight.db"
    initialize_database(db_path=db_path)
    connection = get_connection(db_path)

    source_doc = "tcs_annual_report_2025_26.pdf"
    source_excerpt = "Permanent (D): 5,84,519"
    verified_at = "2026-11-01T10:00:00+00:00"

    try:
        insert_test_fact(
            connection,
            source_doc=source_doc,
            source_excerpt=source_excerpt,
            verification_status="verified",
            verified_at=verified_at,
            verified_by="manual_report_review",
        )
        connection.commit()

        results = get_verified_facts(
            connection,
            company="TCS",
            fiscal_year="FY2026",
            metric="employee_count",
            category="permanent",
        )
    finally:
        connection.close()

    assert len(results) == 1
    assert results[0]["source_doc"] == source_doc
    assert results[0]["source_page"] == 108
    assert results[0]["source_excerpt"] == source_excerpt
    assert results[0]["unit"] == "employees"
    assert results[0]["verification_status"] == "verified"
    assert results[0]["verified_at"] == verified_at
    assert results[0]["verified_by"] == "manual_report_review"


import sqlite3


VALID_VERIFICATION_STATUSES = {"verified", "unverified"}


def insert_verified_fact(
    connection: sqlite3.Connection,
    *,
    company: str,
    fiscal_year: str,
    metric: str,
    category: str | None,
    value: int | float,
    unit: str,
    source_doc: str,
    source_page: int,
    source_excerpt: str,
    chunk_id: str = "",
    verification_status: str = "unverified",
    verified_at: str | None = None,
    verified_by: str | None = None,
) -> int:
    """Insert a financial fact and return its database ID.

    Facts default to unverified.
    Verified facts require a timestamp and reviewer identifier.
    This function does not commit the transaction.
    """

    if verification_status not in VALID_VERIFICATION_STATUSES:
        raise ValueError(
            "verification_status must be 'verified' or 'unverified'."
        )

    if verification_status == "verified":
        if not verified_at or not verified_by:
            raise ValueError(
                "Verified facts require both verified_at and verified_by."
            )
    elif verified_at is not None or verified_by is not None:
        raise ValueError(
            "Unverified facts cannot have verified_at or verified_by."
        )

    cursor = connection.execute(
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
            source_excerpt,
            chunk_id,
            verification_status,
            verified_at,
            verified_by
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company,
            fiscal_year,
            metric,
            category,
            value,
            unit,
            source_doc,
            source_page,
            source_excerpt,
            chunk_id,
            verification_status,
            verified_at,
            verified_by,
        ),
    )

    return int(cursor.lastrowid)


def get_verified_facts(
    connection: sqlite3.Connection,
    *,
    company: str,
    fiscal_year: str,
    metric: str,
    category: str | None = None,
    include_unverified: bool = False,
) -> list[dict[str, int | float | str | None]]:
    """Retrieve financial facts.

    Returns verified facts only by default.
    Set include_unverified=True to retrieve both statuses.
    """

    query = """
        SELECT
            id,
            company,
            fiscal_year,
            metric,
            category,
            value,
            unit,
            source_doc,
            source_page,
            source_excerpt,
            chunk_id,
            verification_status,
            verified_at,
            verified_by
        FROM verified_facts
        WHERE company = ?
        AND fiscal_year = ?
        AND metric = ?
    """

    parameters: list[str] = [company, fiscal_year, metric]

    if not include_unverified:
        query += " AND verification_status = ?"
        parameters.append("verified")

    if category is not None:
        query += " AND category = ?"
        parameters.append(category)

    query += " ORDER BY category, id"

    rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]

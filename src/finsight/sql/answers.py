import re
import sqlite3

from finsight.generation.schemas import Citation, GeneratedAnswer
from finsight.sql.repository import get_verified_facts

EMPLOYEE_COUNT_PATTERNS = (
    r"\bemployee strength\b",
    r"\btotal employees\b",
    r"\btotal employee count\b",
    r"\boverall employee count\b",
    r"\bemployee count\b",
    r"\bheadcount\b",
    r"\bworkforce size\b",
    r"\btotal workforce\b",
)

OTHER_THAN_PERMANENT_PATTERN = r"\bother\s*(?:-| )?\s*than\s*(?:-| )?\s*permanent\b"

PERMANENT_PATTERN = r"\bpermanent\s+employees?\b"

REQUIRED_EMPLOYEE_CATEGORIES = {
    "permanent",
    "other_than_permanent",
    "total",
}


def _normalize_search_text(value: str) -> str:
    """Normalize text for conservative phrase matching."""

    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase occurs as a complete phrase."""

    normalized_text = f" {_normalize_search_text(text)} "
    normalized_phrase = f" {_normalize_search_text(phrase)} "

    if normalized_phrase == "  ":
        return False

    return normalized_phrase in normalized_text


def _get_validated_employee_facts(
    connection: sqlite3.Connection,
    *,
    company: str,
    fiscal_year: str,
) -> dict[str, dict[str, int | float | str | None]] | None:
    """Retrieve and validate all employee-count categories."""

    facts = get_verified_facts(
        connection,
        company=company,
        fiscal_year=fiscal_year,
        metric="employee_count",
    )

    if not facts:
        return None

    facts_by_category = {}

    for fact in facts:
        category = fact["category"]

        if category in facts_by_category:
            raise ValueError(f"Duplicate employee-count category: {category}")

        facts_by_category[category] = fact

    if not REQUIRED_EMPLOYEE_CATEGORIES.issubset(facts_by_category):
        return None

    permanent = facts_by_category["permanent"]
    other_than_permanent = facts_by_category["other_than_permanent"]
    total = facts_by_category["total"]

    values = (
        permanent["value"],
        other_than_permanent["value"],
        total["value"],
    )

    if any(
        not isinstance(value, (int, float)) or not float(value).is_integer() or value < 0
        for value in values
    ):
        raise ValueError("Employee counts must be non-negative integers.")

    permanent_value = int(permanent["value"])
    other_value = int(other_than_permanent["value"])
    total_value = int(total["value"])

    if permanent_value + other_value != total_value:
        raise ValueError(
            "Verified employee counts are inconsistent: category values do not equal the total."
        )

    source_fields = (
        "source_doc",
        "source_page",
        "chunk_id",
        "source_excerpt",
    )

    for fact in (permanent, other_than_permanent):
        if any(fact[field] != total[field] for field in source_fields):
            raise ValueError("Employee-count facts do not share the same source citation.")

    return facts_by_category


def _build_employee_citation(
    fact: dict[str, int | float | str | None],
) -> Citation:
    """Build a citation from a verified employee fact."""

    return Citation(
        document_name=fact["source_doc"],
        page_number=fact["source_page"],
        chunk_id=fact["chunk_id"],
        evidence_text=fact["source_excerpt"],
    )


def build_employee_count_answer(
    connection: sqlite3.Connection,
    *,
    company: str,
    fiscal_year: str,
) -> GeneratedAnswer | None:
    """Build an employee-count answer from verified SQL facts."""

    facts = _get_validated_employee_facts(
        connection,
        company=company,
        fiscal_year=fiscal_year,
    )

    if facts is None:
        return None

    permanent_value = int(facts["permanent"]["value"])
    other_value = int(facts["other_than_permanent"]["value"])
    total_value = int(facts["total"]["value"])

    citation = _build_employee_citation(facts["total"])

    answer = (
        f"{company} reported {total_value:,} total employees "
        f"in {fiscal_year}, comprising "
        f"{permanent_value:,} permanent employees and "
        f"{other_value:,} other-than-permanent employees."
    )

    return GeneratedAnswer(
        answer=answer,
        citations=[citation],
        abstained=False,
    )


def build_employee_category_answer(
    connection: sqlite3.Connection,
    *,
    company: str,
    fiscal_year: str,
    category: str,
) -> GeneratedAnswer | None:
    """Build an answer for one verified employee category."""

    if category not in ("permanent", "other_than_permanent"):
        raise ValueError(f"Unsupported employee category: {category}")

    facts = _get_validated_employee_facts(
        connection,
        company=company,
        fiscal_year=fiscal_year,
    )

    if facts is None:
        return None

    fact = facts[category]
    value = int(fact["value"])

    category_label = "permanent" if category == "permanent" else "other-than-permanent"

    answer = f"{company} reported {value:,} {category_label} employees in {fiscal_year}."

    return GeneratedAnswer(
        answer=answer,
        citations=[_build_employee_citation(fact)],
        abstained=False,
    )


def _is_employee_count_question(question: str) -> bool:
    """Return whether the question concerns employee counts or categories."""

    normalized_question = question.casefold()

    return (
        any(re.search(pattern, normalized_question) for pattern in EMPLOYEE_COUNT_PATTERNS)
        or re.search(
            OTHER_THAN_PERMANENT_PATTERN,
            normalized_question,
        )
        is not None
        or re.search(
            PERMANENT_PATTERN,
            normalized_question,
        )
        is not None
    )


def _get_employee_company_and_year(
    connection: sqlite3.Connection,
    question: str,
) -> tuple[str, str] | None:
    """Resolve one employee-count company and fiscal year."""

    company_rows = connection.execute(
        """
        SELECT DISTINCT company
        FROM verified_facts
        WHERE metric = ?
        """,
        ("employee_count",),
    ).fetchall()

    matched_companies = [
        row["company"] for row in company_rows if _contains_phrase(question, row["company"])
    ]

    if len(matched_companies) != 1:
        return None

    company = matched_companies[0]

    year_match = re.search(
        r"\bfy\s*[- ]?\s*(20\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )

    if year_match is not None:
        fiscal_year = f"FY{year_match.group(1)}"

        year_exists = connection.execute(
            """
            SELECT 1
            FROM verified_facts
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            LIMIT 1
            """,
            (company, fiscal_year, "employee_count"),
        ).fetchone()

        if year_exists is None:
            return None

        return company, fiscal_year

    year_rows = connection.execute(
        """
        SELECT DISTINCT fiscal_year
        FROM verified_facts
        WHERE company = ?
        AND metric = ?
        """,
        (company, "employee_count"),
    ).fetchall()

    available_years = []

    for row in year_rows:
        year = re.search(r"(20\d{2})$", row["fiscal_year"])

        if year:
            available_years.append((int(year.group(1)), row["fiscal_year"]))

    if not available_years:
        return None

    return company, max(available_years)[1]


def build_employee_count_answer_for_question(
    connection: sqlite3.Connection,
    question: str,
) -> GeneratedAnswer | None:
    """Recognize employee-count questions and answer using verified SQL."""

    if not _is_employee_count_question(question):
        return None

    resolved = _get_employee_company_and_year(
        connection,
        question,
    )

    if resolved is None:
        return None

    company, fiscal_year = resolved
    normalized_question = question.casefold()

    other_than_permanent_match = re.search(
        OTHER_THAN_PERMANENT_PATTERN,
        normalized_question,
    )

    permanent_match = re.search(
        PERMANENT_PATTERN,
        normalized_question,
    )

    if other_than_permanent_match:
        return build_employee_category_answer(
            connection,
            company=company,
            fiscal_year=fiscal_year,
            category="other_than_permanent",
        )

    if permanent_match:
        return build_employee_category_answer(
            connection,
            company=company,
            fiscal_year=fiscal_year,
            category="permanent",
        )

    return build_employee_count_answer(
        connection,
        company=company,
        fiscal_year=fiscal_year,
    )


def find_unverified_employee_fact_match(
    connection: sqlite3.Connection,
    question: str,
) -> dict[str, str] | None:
    """Find incomplete employee-count data without reading its values."""

    if not _is_employee_count_question(question):
        return None

    resolved = _get_employee_company_and_year(
        connection,
        question,
    )

    if resolved is None:
        return None

    company, fiscal_year = resolved

    rows = connection.execute(
        """
        SELECT category, verification_status
        FROM verified_facts
        WHERE company = ?
        AND fiscal_year = ?
        AND metric = ?
        """,
        (company, fiscal_year, "employee_count"),
    ).fetchall()

    facts_by_category = {row["category"]: row["verification_status"] for row in rows}

    is_incomplete = any(
        facts_by_category.get(category) != "verified" for category in REQUIRED_EMPLOYEE_CATEGORIES
    )

    if not is_incomplete:
        return None

    return {
        "company": company,
        "fiscal_year": fiscal_year,
        "metric": "employee_count",
    }


def find_unverified_financial_fact_match(
    connection: sqlite3.Connection,
    question: str,
) -> dict[str, str] | None:
    """Find a matching unverified financial fact without reading its value."""

    normalized_question = _normalize_search_text(question)

    year_match = re.search(
        r"\bfy\s*[- ]?\s*(20\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )

    requested_year = f"FY{year_match.group(1)}" if year_match is not None else None

    query = """
        SELECT DISTINCT company, fiscal_year, metric
        FROM verified_facts
        WHERE verification_status = ?
        AND metric != ?
    """

    parameters: list[str] = ["unverified", "employee_count"]

    if requested_year is not None:
        query += " AND fiscal_year = ?"
        parameters.append(requested_year)

    rows = connection.execute(query, parameters).fetchall()

    matching_companies = {
        row["company"] for row in rows if _contains_phrase(normalized_question, row["company"])
    }

    if len(matching_companies) != 1:
        return None

    company = next(iter(matching_companies))
    matching_rows = []

    for row in rows:
        if row["company"] != company:
            continue

        normalized_metric = _normalize_search_text(row["metric"])

        if not normalized_metric or not _contains_phrase(
            normalized_question,
            normalized_metric,
        ):
            continue

        matching_rows.append(
            {
                "company": company,
                "fiscal_year": row["fiscal_year"],
                "metric": normalized_metric,
            }
        )

    if not matching_rows:
        return None

    longest_metric_length = max(len(match["metric"]) for match in matching_rows)

    matching_rows = [
        match for match in matching_rows if len(match["metric"]) == longest_metric_length
    ]

    distinct_metrics = {match["metric"] for match in matching_rows}

    if len(distinct_metrics) != 1:
        return None

    def fiscal_year_number(match: dict[str, str]) -> int:
        year = re.search(r"(20\d{2})$", match["fiscal_year"])
        return int(year.group(1)) if year else 0

    return max(matching_rows, key=fiscal_year_number)

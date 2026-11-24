from datetime import UTC, datetime
from pathlib import Path

from finsight.sql.database import get_connection, initialize_database
from finsight.sql.repository import insert_verified_fact

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.db"

REVIEWER_ID = "manual_report_review"


VERIFIED_FACTS = [
    {
        "company": "TCS",
        "fiscal_year": "FY2026",
        "metric": "employee_count",
        "category": "permanent",
        "value": 584519,
        "unit": "employees",
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "source_page": 108,
        "chunk_id": "tcs_annual_report_2025_26.pdf:page-108:chunk-1",
        "source_excerpt": (
            "Permanent (D): 5,84,519\n"
            "Other than Permanent (E):32,918\n"
            "Total employees (D + E): 6,17,437"
        ),
        "verification_status": "verified",
        "verified_by": REVIEWER_ID,
    },
    {
        "company": "TCS",
        "fiscal_year": "FY2026",
        "metric": "employee_count",
        "category": "other_than_permanent",
        "value": 32918,
        "unit": "employees",
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "source_page": 108,
        "chunk_id": "tcs_annual_report_2025_26.pdf:page-108:chunk-1",
        "source_excerpt": (
            "Permanent (D): 5,84,519\n"
            "Other than Permanent (E):32,918\n"
            "Total employees (D + E): 6,17,437"
        ),
        "verification_status": "verified",
        "verified_by": REVIEWER_ID,
    },
    {
        "company": "TCS",
        "fiscal_year": "FY2026",
        "metric": "employee_count",
        "category": "total",
        "value": 617437,
        "unit": "employees",
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "source_page": 108,
        "chunk_id": "tcs_annual_report_2025_26.pdf:page-108:chunk-1",
        "source_excerpt": (
            "Permanent (D): 5,84,519\n"
            "Other than Permanent (E):32,918\n"
            "Total employees (D + E): 6,17,437"
        ),
        "verification_status": "verified",
        "verified_by": REVIEWER_ID,
    },
]


def seed_verified_facts(
    db_path: Path = DATABASE_PATH,
) -> None:
    """Seed verified facts while preserving provenance and verification history."""

    initialize_database(db_path)
    connection = get_connection(db_path)

    inserted = 0
    skipped = 0
    updated = 0
    promoted = 0

    identity_fields = (
        "company",
        "fiscal_year",
        "metric",
        "category",
        "value",
        "unit",
    )

    provenance_fields = (
        "source_doc",
        "source_page",
        "chunk_id",
        "source_excerpt",
    )

    try:
        for fact in VERIFIED_FACTS:
            existing = connection.execute(
                """
                SELECT *
                FROM verified_facts
                WHERE company = ?
                  AND fiscal_year = ?
                  AND metric = ?
                  AND category = ?
                """,
                (
                    fact["company"],
                    fact["fiscal_year"],
                    fact["metric"],
                    fact["category"],
                ),
            ).fetchall()

            if len(existing) > 1:
                raise ValueError(f"Duplicate database records found for {fact['category']}.")

            if existing:
                record = existing[0]

                for field in identity_fields:
                    if record[field] != fact[field]:
                        raise ValueError(
                            f"Conflicting existing fact for "
                            f"{fact['category']}: {field} does not match."
                        )

                metadata_changed = any(record[field] != fact[field] for field in provenance_fields)

                verification_changed = (
                    record["verification_status"] != "verified"
                    or record["verified_by"] != fact["verified_by"]
                )

                if metadata_changed or verification_changed:
                    verified_at = record["verified_at"]

                    if not verified_at:
                        verified_at = datetime.now(UTC).isoformat()

                    connection.execute(
                        """
                        UPDATE verified_facts
                        SET source_doc = ?,
                            source_page = ?,
                            chunk_id = ?,
                            source_excerpt = ?,
                            verification_status = ?,
                            verified_at = ?,
                            verified_by = ?
                        WHERE id = ?
                        """,
                        (
                            fact["source_doc"],
                            fact["source_page"],
                            fact["chunk_id"],
                            fact["source_excerpt"],
                            "verified",
                            verified_at,
                            fact["verified_by"],
                            record["id"],
                        ),
                    )

                    if metadata_changed:
                        print(f"Updated provenance: {fact['category']}")
                        updated += 1

                    if verification_changed:
                        print(f"Verified: {fact['category']}")
                        promoted += 1

                else:
                    print(f"Already exists and verified: {fact['category']}")
                    skipped += 1

                continue

            verified_at = datetime.now(UTC).isoformat()

            insert_verified_fact(
                connection,
                **fact,
                verified_at=verified_at,
            )

            print(f"Inserted and verified: {fact['category']} = {fact['value']:,}")
            inserted += 1

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    print("\n=== SEEDING SUMMARY ===")
    print(f"Inserted: {inserted}")
    print(f"Updated provenance: {updated}")
    print(f"Promoted to verified: {promoted}")
    print(f"Skipped: {skipped}")
    print(f"Database: {db_path}")


if __name__ == "__main__":
    seed_verified_facts()

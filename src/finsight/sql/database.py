
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "finsight.db"
)

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(
    db_path: Path = DEFAULT_DATABASE_PATH,
) -> sqlite3.Connection:
    """Create and return a connection to the FinSight SQLite database."""

    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database(
    db_path: Path = DEFAULT_DATABASE_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    """Initialize or safely migrate the FinSight database."""

    schema_sql = schema_path.read_text(encoding="utf-8-sig")

    connection = get_connection(db_path)

    try:
        connection.executescript(schema_sql)

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(verified_facts)"
            ).fetchall()
        }

        migrations = {
            "chunk_id": """
                ALTER TABLE verified_facts
                ADD COLUMN chunk_id TEXT NOT NULL DEFAULT ''
            """,
            "verification_status": """
                ALTER TABLE verified_facts
                ADD COLUMN verification_status TEXT NOT NULL
                DEFAULT 'unverified'
                CHECK (
                    verification_status IN ('verified', 'unverified')
                )
            """,
            "verified_at": """
                ALTER TABLE verified_facts
                ADD COLUMN verified_at TEXT
            """,
            "verified_by": """
                ALTER TABLE verified_facts
                ADD COLUMN verified_by TEXT
            """,
        }

        for column_name, migration_sql in migrations.items():
            if column_name not in columns:
                connection.execute(migration_sql)

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verified_facts_verification
            ON verified_facts (verification_status)
            """
        )

        connection.commit()

    finally:
        connection.close()

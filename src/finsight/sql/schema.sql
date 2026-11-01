
CREATE TABLE IF NOT EXISTS verified_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    company TEXT NOT NULL
        CHECK (length(trim(company)) > 0),

    fiscal_year TEXT NOT NULL
        CHECK (length(trim(fiscal_year)) > 0),

    metric TEXT NOT NULL
        CHECK (length(trim(metric)) > 0),

    category TEXT
        CHECK (category IS NULL OR length(trim(category)) > 0),

    value NUMERIC NOT NULL,

    unit TEXT NOT NULL
        CHECK (length(trim(unit)) > 0),

    source_doc TEXT NOT NULL
        CHECK (length(trim(source_doc)) > 0),

    chunk_id TEXT NOT NULL DEFAULT '',

    source_page INTEGER NOT NULL
        CHECK (source_page > 0),

    source_excerpt TEXT NOT NULL
        CHECK (length(trim(source_excerpt)) > 0),

    verification_status TEXT NOT NULL DEFAULT 'unverified'
        CHECK (verification_status IN ('verified', 'unverified')),

    verified_at TEXT,

    verified_by TEXT
);


CREATE INDEX IF NOT EXISTS idx_verified_facts_lookup
ON verified_facts (company, fiscal_year, metric, category);


CREATE INDEX IF NOT EXISTS idx_verified_facts_source
ON verified_facts (source_doc, source_page);

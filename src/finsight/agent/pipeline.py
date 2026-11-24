import asyncio
import logging
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from finsight.generation.financial_metrics import extract_financial_metric
from finsight.generation.generator import generate_answer
from finsight.generation.schemas import Citation, GeneratedAnswer
from finsight.llm.client import LLMClient
from finsight.retrieval.company_scope import resolve_company_symbols
from finsight.retrieval.hybrid import hybrid_search
from finsight.retrieval.rerank import rerank_chunks
from finsight.retrieval.year_scope import resolve_fiscal_years
from finsight.sql.answers import (
    build_employee_count_answer_for_question,
    find_unverified_employee_fact_match,
    find_unverified_financial_fact_match,
)
from finsight.sql.database import DEFAULT_DATABASE_PATH, get_connection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

logger.propagate = False


RERANK_CANDIDATE_LIMIT = 20
DIAGNOSTIC_TEXT_LENGTH = 240


# Only a consolidated statement heading qualifies as revenue evidence.
_CONSOLIDATED_INCOME_STATEMENT_PATTERN = re.compile(
    r"\bconsolidated\s+statements?\s+of\s+"
    r"(?:profit\s+(?:and|&)\s+loss|profit\s+or\s+loss|"
    r"income|comprehensive\s+income)\b",
    re.IGNORECASE,
)

# Used to reject a revenue row belonging to a standalone statement.
_STANDALONE_INCOME_STATEMENT_PATTERN = re.compile(
    r"\bstandalone\s+statements?\s+of\s+"
    r"(?:profit\s+(?:and|&)\s+loss|profit\s+or\s+loss|"
    r"income|comprehensive\s+income)\b",
    re.IGNORECASE,
)

_REVENUE_LABEL_PATTERN = re.compile(
    r"^\s*(?:revenue\s+from\s+operations|revenues)\s*$",
    re.IGNORECASE,
)

_FINANCIAL_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"(?<!\d)"
    r"\(?-?\d[\d,]*(?:\.\d+)?\)?"
)

_UNIT_WORD_PATTERN = re.compile(
    r"\b(crores?|millions?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Revenue question detection
# ---------------------------------------------------------------------

def _is_cross_company_revenue_question(question: str) -> bool:
    """Identify cross-company revenue comparisons and rankings."""

    normalized = " ".join(question.casefold().split())

    patterns = (
        r"\bhighest\s+revenue\b",
        r"\bwhich\s+company\b.{0,100}\brevenue\b",
        r"\btop\s+companies?\b.{0,100}\brevenue\b",
        r"\brank\s+companies\b.{0,100}\brevenue\b",
        r"\bcompanies?\s+ranked\b.{0,100}\brevenue\b",
        r"\bcompany\s+with\s+the\s+most\s+revenue\b",
        r"\bcompare\b.{0,200}\brevenues?\b",
        r"\brevenues?\b.{0,200}\bcompare\b",
        r"\brevenue\s+comparison\b",
        r"\brevenues?\s+comparison\b",
        r"\bcomparison\b.{0,150}\brevenues?\b",
        r"\b(?:vs\.?|versus)\b.{0,150}\brevenues?\b",
        r"\brevenues?\b.{0,150}\b(?:vs\.?|versus)\b",
        r"\bdifference\b.{0,150}\brevenues?\b",
        r"\brevenues?\b.{0,150}\bdifference\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


# ---------------------------------------------------------------------
# Fiscal year and number formatting
# ---------------------------------------------------------------------

def _year_sort_key(year: str) -> tuple[int, int]:
    """Convert a fiscal-year string into a sortable key."""

    match = re.search(
        r"(\d{4})\s*[-/]\s*(\d{2,4})",
        year,
    )

    if not match:
        return (0, 0)

    start_year = int(match.group(1))
    end_text = match.group(2)

    end_year = (
        int(end_text)
        if len(end_text) == 4
        else (start_year // 100) * 100 + int(end_text)
    )

    if end_year < start_year:
        end_year += 100

    return (start_year, end_year)


def _fiscal_year_end_number(year: str) -> int | None:
    """Return the ending calendar year of a fiscal-year label."""

    key = _year_sort_key(year)

    if key == (0, 0):
        return None

    return key[1]


def _format_indian_number(value: float) -> str:
    """Format a number using Indian digit grouping."""

    rounded = round(value, 1)

    if rounded.is_integer():
        number = str(int(rounded))
    else:
        number = f"{rounded:.1f}"

    integer_part, separator, decimal_part = number.partition(".")

    if len(integer_part) <= 3:
        grouped = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []

        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]

        grouped = ",".join(groups + [last_three])

    return grouped + (
        separator + decimal_part
        if separator
        else ""
    )


# ---------------------------------------------------------------------
# Financial statement parsing
# ---------------------------------------------------------------------

def _line_offsets(lines: list[str]) -> list[int]:
    """Return the starting character offset of each line."""

    offsets = []
    position = 0

    for line in lines:
        offsets.append(position)
        position += len(line) + 1

    return offsets


def _detect_revenue_unit(
    lines: list[str],
    offsets: list[int],
    revenue_position: int,
) -> str | None:
    """Detect the financial statement's reporting unit."""

    candidates: list[tuple[int, str]] = []

    for line_index, line in enumerate(lines):
        unit_matches = list(
            _UNIT_WORD_PATTERN.finditer(line)
        )

        if not unit_matches:
            continue

        lowered = line.casefold()

        if re.search(
            r"\b(?:u\.?\s*s\.?\s*dollars?|dollars?|usd)\b",
            lowered,
        ):
            continue

        stripped = line.strip()

        explicit_declaration = (
            stripped.startswith(("(", "["))
            or bool(re.search(r"\bin\b", line, re.IGNORECASE))
            or bool(re.search(r"\binr\b", line, re.IGNORECASE))
            or "₹" in line
            or bool(re.search(r"\brs\.?\b", line, re.IGNORECASE))
        )

        if not explicit_declaration:
            continue

        line_position = offsets[line_index]
        distance = abs(line_position - revenue_position)

        for match in unit_matches:
            unit_word = match.group(1).casefold()

            unit_kind = (
                "millions"
                if unit_word.startswith("million")
                else "crores"
            )

            candidates.append((distance, unit_kind))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    nearest_distance = candidates[0][0]

    if nearest_distance > 15000:
        return None

    nearest_kinds = {
        unit_kind
        for distance, unit_kind in candidates
        if distance == nearest_distance
    }

    if len(nearest_kinds) != 1:
        return None

    return next(iter(nearest_kinds))


def _extract_revenue_header_years(
    text: str,
    lines: list[str],
    line_index: int,
    revenue_position: int,
    statement_matches: list[re.Match],
) -> list[int]:
    """Extract financial-year labels preceding a revenue row."""

    preceding_statement_matches = [
        match
        for match in statement_matches
        if match.start() < revenue_position
    ]

    if preceding_statement_matches:
        header_region = text[
            preceding_statement_matches[-1].end():
            revenue_position
        ]
    else:
        start_line = max(0, line_index - 60)
        header_region = "\n".join(
            lines[start_line:line_index]
        )

    year_markers = list(
        re.finditer(
            r"\byear\s+ended\b",
            header_region,
            re.IGNORECASE,
        )
    )

    if not year_markers:
        return []

    complete_header = header_region[
        year_markers[0].start():
    ]

    return [
        int(value)
        for value in re.findall(
            r"\b(?:19|20)\d{2}\b",
            complete_header,
        )
    ]


def _is_numeric_table_line(line: str) -> bool:
    """Check whether a line contains only numeric table data and symbols."""

    without_numbers = _FINANCIAL_NUMBER_PATTERN.sub(
        "",
        line,
    )

    without_symbols = re.sub(
        r"[\s₹$€£¥`'(),./:\\\-+%]",
        "",
        without_numbers,
    )

    return not without_symbols


def _parse_table_numbers(line: str) -> list[float]:
    """Parse numeric values from one financial table line."""

    raw_values = _FINANCIAL_NUMBER_PATTERN.findall(line)
    parsed_values: list[float] = []

    for raw_value in raw_values:
        cleaned = (
            raw_value.replace(",", "")
            .replace("(", "-")
            .replace(")", "")
        )

        try:
            parsed_values.append(float(cleaned))
        except ValueError:
            continue

    return parsed_values


def _extract_revenue_data_values(
    lines: list[str],
    line_index: int,
    expected_columns: int,
) -> list[float]:
    """Extract revenue values while skipping isolated note references."""

    data_values: list[float] = []
    started = False

    for following_line in lines[line_index + 1:line_index + 13]:
        raw_values = _parse_table_numbers(following_line)

        if not raw_values:
            if started and following_line.strip():
                break
            continue

        if (
            len(raw_values) == 1
            and abs(raw_values[0]) <= 100
        ):
            continue

        if (
            not started
            and expected_columns > 0
            and len(raw_values) > expected_columns
        ):
            raw_values = list(raw_values)

            while (
                len(raw_values) > expected_columns
                and raw_values
                and abs(raw_values[0]) <= 100
            ):
                raw_values.pop(0)

        if not _is_numeric_table_line(following_line):
            if started:
                break
            continue

        started = True
        data_values.extend(raw_values)

        if expected_columns <= 0:
            break

        if len(data_values) >= expected_columns:
            break

    return data_values


def _extract_consolidated_revenue(
    chunk: dict[str, int | float | str],
    target_fiscal_year: str | None = None,
) -> float | None:
    """Extract verified consolidated revenue normalized to ₹ crore."""

    text = str(chunk.get("text", ""))

    if not text.strip():
        return None

    consolidated_matches = list(
        _CONSOLIDATED_INCOME_STATEMENT_PATTERN.finditer(text)
    )

    standalone_matches = list(
        _STANDALONE_INCOME_STATEMENT_PATTERN.finditer(text)
    )

    if not consolidated_matches:
        return None

    lines = text.splitlines()
    offsets = _line_offsets(lines)

    target_end_year = (
        _fiscal_year_end_number(target_fiscal_year)
        if target_fiscal_year
        else None
    )

    for line_index, line in enumerate(lines):
        if not _REVENUE_LABEL_PATTERN.match(line):
            continue

        revenue_position = offsets[line_index]

        heading_candidates: list[tuple[int, str]] = []

        for match in consolidated_matches:
            heading_candidates.append(
                (
                    abs(match.start() - revenue_position),
                    "consolidated",
                )
            )

        for match in standalone_matches:
            heading_candidates.append(
                (
                    abs(match.start() - revenue_position),
                    "standalone",
                )
            )

        if not heading_candidates:
            continue

        heading_candidates.sort(key=lambda item: item[0])
        nearest_distance = heading_candidates[0][0]

        nearest_heading_types = {
            heading_type
            for distance, heading_type in heading_candidates
            if distance == nearest_distance
        }

        if nearest_heading_types != {"consolidated"}:
            continue

        if nearest_distance > 20000:
            continue

        header_years = _extract_revenue_header_years(
            text=text,
            lines=lines,
            line_index=line_index,
            revenue_position=revenue_position,
            statement_matches=consolidated_matches,
        )

        if not header_years:
            continue

        if target_end_year is not None:
            matching_year_indices = [
                index
                for index, year in enumerate(header_years)
                if year == target_end_year
            ]

            if not matching_year_indices:
                continue

            selected_index = matching_year_indices[0]

        else:
            latest_year = max(header_years)

            matching_year_indices = [
                index
                for index, year in enumerate(header_years)
                if year == latest_year
            ]

            if not matching_year_indices:
                continue

            selected_index = matching_year_indices[0]

        data_values = _extract_revenue_data_values(
            lines=lines,
            line_index=line_index,
            expected_columns=len(header_years),
        )

        if selected_index >= len(data_values):
            continue

        unit_kind = _detect_revenue_unit(
            lines=lines,
            offsets=offsets,
            revenue_position=revenue_position,
        )

        if unit_kind is None:
            continue

        revenue = data_values[selected_index]

        if revenue <= 0:
            continue

        if unit_kind == "millions":
            return revenue / 10.0

        return revenue

    return None


# ---------------------------------------------------------------------
# Chunk ordering and revenue evidence selection
# ---------------------------------------------------------------------

def _chunk_order_key(
    chunk: dict[str, int | float | str],
) -> tuple[int, str]:
    """Sort page chunks numerically rather than lexicographically."""

    chunk_id = str(chunk.get("chunk_id", ""))

    match = re.search(
        r"chunk[-_](\d+)",
        chunk_id,
        re.IGNORECASE,
    )

    if match:
        return (int(match.group(1)), chunk_id)

    return (10**9, chunk_id)


def _find_revenue_row_chunk(
    chunks: list[dict[str, int | float | str]],
) -> dict[str, int | float | str] | None:
    """Return the original chunk containing the financial revenue row."""

    ordered_chunks = sorted(
        chunks,
        key=_chunk_order_key,
    )

    for label_pattern in (
        re.compile(
            r"(?im)^\s*revenue\s+from\s+operations\s*$"
        ),
        re.compile(
            r"(?im)^\s*revenues\s*$"
        ),
    ):
        for chunk in ordered_chunks:
            text = str(chunk.get("text", ""))

            if label_pattern.search(text):
                return chunk

    return None


# ---------------------------------------------------------------------
# Deterministic company revenue comparison
# ---------------------------------------------------------------------

def _find_highest_revenue_answer(
    question: str,
    chunks: list[dict[str, int | float | str]],
) -> GeneratedAnswer | None:
    """Compare verified consolidated revenue across the relevant companies."""

    if not _is_cross_company_revenue_question(question):
        return None

    resolved_symbols = resolve_company_symbols(
        question,
        chunks,
    )

    normalized_resolved_symbols = {
        str(symbol).strip().upper()
        for symbol in (resolved_symbols or set())
        if str(symbol).strip()
    }

    if (
        normalized_resolved_symbols
        and len(normalized_resolved_symbols) < 2
    ):
        return None

    company_chunks: dict[
        str,
        list[dict[str, int | float | str]],
    ] = {}

    for chunk in chunks:
        symbol = str(chunk.get("symbol", "")).strip().upper()

        if not symbol:
            continue

        if (
            normalized_resolved_symbols
            and symbol not in normalized_resolved_symbols
        ):
            continue

        company_chunks.setdefault(symbol, []).append(chunk)

    if not company_chunks:
        return GeneratedAnswer(
            answer=(
                "I couldn't compare company revenues because no "
                "matching company symbols were found in the index."
            ),
            citations=[],
            abstained=True,
        )

    if normalized_resolved_symbols:
        missing_symbols = (
            normalized_resolved_symbols - set(company_chunks)
        )

        if missing_symbols:
            return GeneratedAnswer(
                answer=(
                    "I couldn't complete the requested revenue "
                    "comparison because these companies are not "
                    "available in the indexed data: "
                    f"{', '.join(sorted(missing_symbols))}."
                ),
                citations=[],
                abstained=True,
            )

    is_named_company_comparison = bool(
        normalized_resolved_symbols
    )

    requested_years = resolve_fiscal_years(question)

    years_by_symbol: dict[str, set[str]] = {}

    for symbol, symbol_chunks in company_chunks.items():
        years = {
            str(chunk.get("fiscal_year", "")).strip()
            for chunk in symbol_chunks
            if chunk.get("fiscal_year")
        }

        years.discard("")
        years_by_symbol[symbol] = years

    if requested_years:
        target_years = set(requested_years)

    else:
        common_years = set.intersection(
            *years_by_symbol.values()
        )

        if not common_years:
            return GeneratedAnswer(
                answer=(
                    "I couldn't establish a common fiscal year across "
                    "the companies being compared. A reliable revenue "
                    "comparison cannot be provided."
                ),
                citations=[],
                abstained=True,
            )

        latest_year = max(
            common_years,
            key=_year_sort_key,
        )

        target_years = {latest_year}

    if len(target_years) != 1:
        return GeneratedAnswer(
            answer=(
                "Please specify one fiscal year for the revenue "
                "comparison. I cannot combine figures from different "
                "reporting years."
            ),
            citations=[],
            abstained=True,
        )

    target_year = next(iter(target_years))

    ranked: list[dict[str, object]] = []
    excluded: list[str] = []

    for symbol, symbol_chunks in company_chunks.items():
        matching_chunks = [
            chunk
            for chunk in symbol_chunks
            if str(chunk.get("fiscal_year", "")).strip()
            == target_year
        ]

        company_name = str(
            symbol_chunks[0].get("company_name", symbol)
        )

        if not matching_chunks:
            excluded.append(
                f"{company_name} ({symbol})"
            )
            continue

        page_groups: dict[
            str,
            list[dict[str, int | float | str]],
        ] = {}

        for chunk in matching_chunks:
            page_key = str(
                chunk.get(
                    "page_number",
                    chunk.get("chunk_id", "unknown"),
                )
            )

            page_groups.setdefault(page_key, []).append(chunk)

        verified_candidates: list[
            tuple[
                float,
                dict[str, int | float | str],
                str,
            ]
        ] = []

        for page_chunks in page_groups.values():
            ordered_page_chunks = sorted(
                page_chunks,
                key=_chunk_order_key,
            )

            combined_text = "\n".join(
                str(item.get("text", ""))
                for item in ordered_page_chunks
            )

            combined_chunk = dict(
                ordered_page_chunks[0]
            )
            combined_chunk["text"] = combined_text

            revenue = _extract_consolidated_revenue(
                combined_chunk,
                target_fiscal_year=target_year,
            )

            if revenue is None:
                continue

            row_chunk = _find_revenue_row_chunk(
                ordered_page_chunks
            )

            if row_chunk is None:
                continue

            verified_candidates.append(
                (
                    revenue,
                    row_chunk,
                    combined_text,
                )
            )

        if not verified_candidates:
            excluded.append(
                f"{company_name} ({symbol})"
            )
            continue

        verified_candidates.sort(
            key=lambda item: (
                not bool(
                    _CONSOLIDATED_INCOME_STATEMENT_PATTERN.search(
                        item[2]
                    )
                ),
                len(str(item[1].get("text", ""))),
            )
        )

        revenue, evidence_chunk, evidence_text = (
            verified_candidates[0]
        )

        ranked.append(
            {
                "symbol": symbol,
                "company_name": str(
                    evidence_chunk.get(
                        "company_name",
                        company_name,
                    )
                ),
                "revenue_crore": revenue,
                "chunk": evidence_chunk,
                "evidence_text": evidence_text,
            }
        )

    if not ranked:
        return GeneratedAnswer(
            answer=(
                "I couldn't find verified consolidated revenue "
                f"figures for the requested companies in FY {target_year}."
            ),
            citations=[],
            abstained=True,
        )

    if is_named_company_comparison and excluded:
        return GeneratedAnswer(
            answer=(
                "I couldn't complete the requested comparison for "
                f"FY {target_year}. Verified consolidated revenue "
                "was unavailable for: "
                f"{', '.join(sorted(excluded))}. "
                "I have not estimated or substituted any figures."
            ),
            citations=[],
            abstained=True,
        )

    ranked.sort(
        key=lambda item: float(item["revenue_crore"]),
        reverse=True,
    )

    citations: list[Citation] = []

    lines = [
        f"**Consolidated revenue comparison — FY {target_year}**",
        "",
        "| Company | Symbol | Revenue (₹ crore) |",
        "|---|---|---:|",
    ]

    for item in ranked:
        company_name = str(item["company_name"])
        symbol = str(item["symbol"])
        revenue = float(item["revenue_crore"])
        evidence_chunk = item["chunk"]

        lines.append(
            f"| {company_name} | {symbol} | "
            f"{_format_indian_number(revenue)} |"
        )

        citations.append(
            Citation(
                document_name=str(
                    evidence_chunk.get(
                        "source_doc",
                        "Unknown document",
                    )
                ),
                page_number=int(
                    evidence_chunk.get("page_number", 0)
                ),
                chunk_id=str(
                    evidence_chunk.get("chunk_id", "")
                ),
                evidence_text=str(item["evidence_text"]),
            )
        )

    if len(ranked) == 2:
        higher = ranked[0]
        lower = ranked[1]

        difference = (
            float(higher["revenue_crore"])
            - float(lower["revenue_crore"])
        )

        lower_name = str(
            lower["company_name"]
        ).rstrip(".")

        lines.extend(
            [
                "",
                f"**Difference:** {higher['company_name']} "
                f"reported ₹{_format_indian_number(difference)} crore "
                f"more revenue than {lower_name}.",
                "",
                f"**Conclusion:** {higher['company_name']} "
                f"({higher['symbol']}) reported the higher "
                f"consolidated revenue in FY {target_year}.",
            ]
        )

        if is_named_company_comparison:
            lines.extend(
                [
                    "",
                    "Comparison covers "
                    f"{len(ranked)} of {len(company_chunks)} "
                    "requested companies.",
                ]
            )

    else:
        highest = ranked[0]

        lines.extend(
            [
                "",
                f"**Highest revenue:** {highest['company_name']} "
                f"({highest['symbol']}) with "
                f"₹{_format_indian_number(float(highest['revenue_crore']))} "
                "crore.",
                "",
                f"Comparison covers {len(ranked)} of "
                f"{len(company_chunks)} "
                + (
                    "requested companies."
                    if is_named_company_comparison
                    else "indexed companies."
                ),
            ]
        )

    if excluded:
        lines.extend(
            [
                "",
                "Revenue could not be verified for: "
                + ", ".join(sorted(excluded))
                + ". These companies were excluded rather than estimated.",
            ]
        )

    logger.info(
        "FinSight deterministic revenue comparison | "
        "year: %s | companies in scope: %d | "
        "verified: %d | excluded: %d | named scope: %s",
        target_year,
        len(company_chunks),
        len(ranked),
        len(excluded),
        is_named_company_comparison,
    )

    return GeneratedAnswer(
        answer="\n".join(lines),
        citations=citations,
        abstained=False,
    )


# ---------------------------------------------------------------------
# Retrieval diagnostics
# ---------------------------------------------------------------------

def _log_ranked_chunks(
    stage: str,
    chunks: list[dict[str, int | float | str]],
) -> None:
    """Log chunk identifiers, ranking scores, and short text previews."""

    logger.info(
        "FinSight retrieval diagnostics | %s | chunks: %d",
        stage,
        len(chunks),
    )

    for rank, chunk in enumerate(chunks, start=1):
        text = " ".join(
            str(chunk.get("text", "")).split()
        )

        logger.info(
            "FinSight retrieval diagnostics | "
            "%s | rank: %02d | "
            "chunk_id: %s | "
            "symbol: %s | "
            "fiscal_year: %s | "
            "page: %s | "
            "score: %.6f | "
            "text: %s",
            stage,
            rank,
            chunk.get("chunk_id", "missing"),
            chunk.get("symbol", "unknown"),
            chunk.get("fiscal_year", "unknown"),
            chunk.get("page_number", "unknown"),
            float(chunk.get("score", 0.0)),
            text[:DIAGNOSTIC_TEXT_LENGTH],
        )


# ---------------------------------------------------------------------
# Exact total-assets evidence
# ---------------------------------------------------------------------

def _find_exact_total_assets_evidence(
    question: str,
    chunks: list[dict[str, int | float | str]],
    allowed_symbols: set[str] | None,
    allowed_fiscal_years: set[str] | None,
) -> list[dict[str, int | float | str]]:
    """Search the full index for the exact total-assets row."""

    normalized_question = " ".join(
        question.casefold().split()
    )

    if not re.search(
        r"\btotal\s+assets?\b",
        normalized_question,
    ):
        return []

    if re.search(
        r"\b(?:percentage|percent|ratio|proportion)\b|%",
        normalized_question,
    ):
        return []

    if not allowed_symbols or not allowed_fiscal_years:
        logger.warning(
            "Exact total-assets lookup skipped: company or fiscal-year "
            "scope could not be resolved."
        )
        return []

    expected_symbols = {
        str(symbol).strip().upper()
        for symbol in allowed_symbols
    }

    expected_years = {
        str(year).strip()
        for year in allowed_fiscal_years
    }

    basis = (
        "standalone"
        if re.search(
            r"\bstandalone\b",
            normalized_question,
        )
        else "consolidated"
    )

    balance_sheet_pattern = re.compile(
        rf"\b{basis}\s+balance\s+sheet\b",
        re.IGNORECASE,
    )

    total_assets_pattern = re.compile(
        r"(?im)^[ \t]*total[ \t]+assets[ \t]*$"
    )

    matches: list[dict[str, int | float | str]] = []

    for chunk in chunks:
        symbol = str(
            chunk.get("symbol", "")
        ).strip().upper()

        fiscal_year = str(
            chunk.get("fiscal_year", "")
        ).strip()

        if symbol not in expected_symbols:
            continue

        if fiscal_year not in expected_years:
            continue

        text = str(chunk.get("text", ""))

        if not text.strip():
            continue

        heading_matches = list(
            balance_sheet_pattern.finditer(text)
        )

        if not heading_matches:
            continue

        label_matches = list(
            total_assets_pattern.finditer(text)
        )

        if not label_matches:
            continue

        valid_table_match = False

        for label_match in label_matches:
            heading_nearby = any(
                abs(heading.start() - label_match.start()) <= 5000
                for heading in heading_matches
            )

            if not heading_nearby:
                continue

            following_text = text[
                label_match.end():
                label_match.end() + 250
            ]

            if not re.search(r"\d", following_text):
                continue

            valid_table_match = True
            break

        if not valid_table_match:
            continue

        if re.search(
            r"statement\s+of\s+segment\s+assets\s+and\s+liabilities",
            text,
            re.IGNORECASE,
        ):
            continue

        if re.search(
            r"segment\s+assets\s+and\s+liabilities",
            text,
            re.IGNORECASE,
        ):
            continue

        matches.append(chunk)

    unique_matches: list[
        dict[str, int | float | str]
    ] = []

    seen_ids: set[str] = set()

    for chunk in matches:
        chunk_id = str(
            chunk.get("chunk_id", "")
        ).strip()

        if not chunk_id or chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        unique_matches.append(chunk)

    logger.info(
        "FinSight exact total-assets lookup | "
        "company symbols: %s | years: %s | basis: %s | matches: %d",
        sorted(expected_symbols),
        sorted(expected_years),
        basis,
        len(unique_matches),
    )

    return unique_matches[:3]


# ---------------------------------------------------------------------
# Exact operating-margin evidence across reconstructed pages
# ---------------------------------------------------------------------

def _find_exact_operating_margin_evidence(
    question: str,
    chunks: list[dict[str, int | float | str]],
    allowed_symbols: set[str] | None,
    allowed_fiscal_years: set[str] | None,
) -> list[dict[str, int | float | str]]:
    """Find operating-margin evidence across all chunks of a report page.

    Annual-report chunking can split a section heading and its metric value
    into adjacent chunks on the same page. Search the reconstructed page,
    not each chunk independently, and accept a fiscal-year qualifier between
    the metric label and the reported percentage.
    """

    normalized_question = " ".join(
        question.casefold().split()
    )

    if not re.search(
        r"\b(?:operating|ebit)\s+(?:profit\s+)?margins?\b",
        normalized_question,
    ):
        return []

    if not allowed_symbols or not allowed_fiscal_years:
        logger.warning(
            "Exact operating-margin lookup skipped: company or fiscal-year "
            "scope could not be resolved."
        )
        return []

    expected_symbols = {
        str(symbol).strip().upper()
        for symbol in allowed_symbols
    }
    expected_years = {
        str(year).strip()
        for year in allowed_fiscal_years
    }

    requested_basis = (
        "standalone"
        if re.search(r"\bstandalone\b", normalized_question)
        else "consolidated"
    )

    heading_pattern = re.compile(
        rf"\b{requested_basis}\s+financial\s+performance\b",
        re.IGNORECASE,
    )

    other_basis = (
        "standalone"
        if requested_basis == "consolidated"
        else "consolidated"
    )

    other_heading_pattern = re.compile(
        rf"\b{other_basis}\s+financial\s+performance\b",
        re.IGNORECASE,
    )

    metric_label_pattern = re.compile(
        r"\b(?:EBIT|operating(?:\s+profit)?)\s+margins?\b",
        re.IGNORECASE,
    )

    reported_value_pattern = re.compile(
        r"\b(?:stood\s+at|was|is|of)\s*"
        r"(\d{1,2}(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )

    # Group chunks by document, page, company and fiscal year.
    # Different reports, companies, pages and years are never mixed.
    page_groups: dict[
        tuple[str, str, str, str],
        list[dict[str, int | float | str]],
    ] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        symbol = str(
            chunk.get("symbol", "")
        ).strip().upper()

        fiscal_year = str(
            chunk.get("fiscal_year", "")
        ).strip()

        if symbol not in expected_symbols:
            continue

        if fiscal_year not in expected_years:
            continue

        source_doc = str(
            chunk.get(
                "source_doc",
                chunk.get("document_name", ""),
            )
        ).strip()

        page_number = str(
            chunk.get("page_number", "")
        ).strip()

        if not source_doc or not page_number:
            continue

        key = (
            source_doc,
            page_number,
            symbol,
            fiscal_year,
        )

        page_groups.setdefault(key, []).append(chunk)

    matches: list[
        dict[str, int | float | str]
    ] = []

    seen_ids: set[str] = set()

    for page_key, page_chunks in page_groups.items():
        fiscal_year = page_key[3]

        # Always join chunks in numeric chunk order, even when the
        # source index or retrieval loader returns them out of order.
        ordered_page_chunks = sorted(
            page_chunks,
            key=_chunk_order_key,
        )

        page_text = "\n".join(
            str(chunk.get("text", ""))
            for chunk in ordered_page_chunks
            if str(chunk.get("text", "")).strip()
        )

        if not page_text:
            continue

        heading_match = heading_pattern.search(page_text)

        if heading_match is None:
            continue

        # Restrict extraction to the requested section. This prevents
        # a heading elsewhere on the same page from qualifying a metric
        # from the opposite reporting basis.
        section_end_match = other_heading_pattern.search(
            page_text,
            heading_match.end(),
        )

        section_text = page_text[
            heading_match.end():
            section_end_match.start()
            if section_end_match
            else len(page_text)
        ]

        # Build a year expression such as 2025-26 or 2025-2026.
        year_match = re.search(
            r"(\d{4})\s*[-/–—]\s*(\d{2,4})",
            fiscal_year,
        )

        if not year_match:
            continue

        year_start = year_match.group(1)
        year_end = year_match.group(2)
        year_end_short = year_end[-2:]

        requested_year_pattern = re.compile(
            rf"\b{year_start}\s*[-/–—]\s*"
            rf"(?:{year_end_short}|{year_end})\b",
            re.IGNORECASE,
        )

        selected_value_match = None

        for label_match in metric_label_pattern.finditer(
            section_text
        ):
            # The value may follow wording such as:
            # "for the financial year 2025-26 stood at 12.6%".
            following_text = section_text[
                label_match.end():
                label_match.end() + 260
            ]

            value_match = reported_value_pattern.search(
                following_text
            )

            if value_match is None:
                continue

            between_label_and_value = following_text[
                :value_match.start()
            ]

            if not requested_year_pattern.search(
                between_label_and_value
            ):
                continue

            selected_value_match = value_match
            break

        if selected_value_match is None:
            continue

        # Locate the original chunk containing the percentage.
        # The complete page text is carried into the extractor so it can
        # see both the section heading and the value.
        percentage_text = selected_value_match.group(1)

        percentage_pattern = re.compile(
            rf"(?<!\d){re.escape(percentage_text)}\s*%"
        )

        value_chunk = next(
            (
                chunk
                for chunk in ordered_page_chunks
                if percentage_pattern.search(
                    str(chunk.get("text", ""))
                )
            ),
            ordered_page_chunks[-1],
        )

        evidence_chunk = dict(value_chunk)
        evidence_chunk["text"] = page_text

        chunk_id = str(
            evidence_chunk.get("chunk_id", "")
        ).strip()

        if not chunk_id or chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        matches.append(evidence_chunk)

    logger.info(
        "FinSight exact operating-margin lookup | "
        "symbols: %s | years: %s | basis: %s | "
        "pages scanned: %d | matches: %d",
        sorted(expected_symbols),
        sorted(expected_years),
        requested_basis,
        len(page_groups),
        len(matches),
    )

    return matches[:5]


# ---------------------------------------------------------------------
# Exact evidence prioritization
# ---------------------------------------------------------------------

def _prioritize_exact_metric_evidence(
    exact_evidence: list[dict[str, int | float | str]],
    reranked_chunks: list[dict[str, int | float | str]],
    top_k: int,
) -> list[dict[str, int | float | str]]:
    """Preserve exact metric evidence ahead of normal reranker results."""

    if not exact_evidence:
        return reranked_chunks

    prioritized: list[
        dict[str, int | float | str]
    ] = []

    seen_ids: set[str] = set()

    for chunk in exact_evidence:
        chunk_id = str(
            chunk.get("chunk_id", "")
        ).strip()

        if chunk_id and chunk_id not in seen_ids:
            prioritized.append(chunk)
            seen_ids.add(chunk_id)

    for chunk in reranked_chunks:
        chunk_id = str(
            chunk.get("chunk_id", "")
        ).strip()

        if chunk_id and chunk_id not in seen_ids:
            prioritized.append(chunk)
            seen_ids.add(chunk_id)

    selected = prioritized[:top_k]

    logger.info(
        "FinSight metric evidence retention | "
        "exact metric matches: %d | selected: %d | IDs: %s",
        len(exact_evidence),
        len(selected),
        [
            str(chunk.get("chunk_id", "missing"))
            for chunk in selected
        ],
    )

    return selected


# ---------------------------------------------------------------------
# Main FinSight agent
# ---------------------------------------------------------------------

class FinSightAgent:
    """Coordinate SQL answering, retrieval, reranking, and generation."""

    def __init__(
        self,
        *,
        chunks: list[dict[str, int | float | str]],
        embedding_model: SentenceTransformer,
        bm25_index: BM25Okapi,
        chunk_embeddings: np.ndarray,
        reranker: CrossEncoder,
        llm: LLMClient,
        sql_database_path: Path = DEFAULT_DATABASE_PATH,
        candidate_k: int = 50,
        hybrid_top_k: int = 50,
        rerank_top_k: int = 5,
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty.")

        if chunk_embeddings.ndim != 2:
            raise ValueError(
                "chunk_embeddings must be a 2D array."
            )

        if len(chunk_embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match the number of chunks."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero."
            )

        if hybrid_top_k <= 0:
            raise ValueError(
                "hybrid_top_k must be greater than zero."
            )

        if rerank_top_k <= 0:
            raise ValueError(
                "rerank_top_k must be greater than zero."
            )

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.bm25_index = bm25_index
        self.chunk_embeddings = chunk_embeddings
        self.reranker = reranker
        self.llm = llm
        self.sql_database_path = sql_database_path

        self.candidate_k = candidate_k
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k

    def _try_deterministic_financial_metric_answer(
        self,
        question: str,
        evidence: list[dict[str, int | float | str]],
        allowed_symbols: set[str] | None,
        allowed_fiscal_years: set[str] | None,
    ) -> GeneratedAnswer | None:
        """
        Try deterministic extraction before query expansion and reranking.

        Recover company and fiscal-year metadata from canonical indexed
        chunks because hybrid retrieval results may omit those fields.

        The fast path is limited to one resolved company and one explicit
        fiscal year. Comparisons, rankings and multi-year questions stay on
        the normal retrieval and generation path.
        """

        if not allowed_symbols or len(allowed_symbols) != 1:
            return None

        if not allowed_fiscal_years or len(allowed_fiscal_years) != 1:
            return None

        if re.search(
            r"\b(?:compare|comparison|versus|vs\.?|difference|"
            r"trend|historical|rank|ranking|highest|lowest)\b",
            question,
            re.IGNORECASE,
        ):
            return None

        expected_symbols = {
            str(symbol).strip().upper()
            for symbol in allowed_symbols
        }

        expected_years = {
            str(year).strip()
            for year in allowed_fiscal_years
        }

        # Build a canonical lookup from the original indexed chunks.
        canonical_chunks_by_id = {
            str(chunk["chunk_id"]).strip(): chunk
            for chunk in self.chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("chunk_id"), str)
            and chunk["chunk_id"].strip()
        }

        evidence_map: dict[
            str,
            dict[str, int | float | str],
        ] = {}

        for retrieved_chunk in evidence:
            if not isinstance(retrieved_chunk, dict):
                continue

            chunk_id = retrieved_chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue

            normalized_chunk_id = chunk_id.strip()

            canonical_chunk = canonical_chunks_by_id.get(
                normalized_chunk_id
            )

            if canonical_chunk is None:
                continue

            # Retain retrieved evidence, while restoring canonical metadata.
            merged_chunk = {
                **canonical_chunk,
                **retrieved_chunk,
            }

            # An empty retrieval field must not overwrite valid canonical
            # company/year metadata.
            for metadata_field in (
                "symbol",
                "fiscal_year",
                "company_name",
            ):
                if not retrieved_chunk.get(metadata_field):
                    if canonical_chunk.get(metadata_field):
                        merged_chunk[metadata_field] = (
                            canonical_chunk[metadata_field]
                        )

            source_doc = merged_chunk.get("source_doc")
            page_number = merged_chunk.get("page_number")
            text = merged_chunk.get("text")

            if (
                not isinstance(source_doc, str)
                or not source_doc.strip()
                or not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number <= 0
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue

            chunk_symbol = str(
                merged_chunk.get("symbol", "")
            ).strip().upper()

            if chunk_symbol not in expected_symbols:
                continue

            chunk_year = str(
                merged_chunk.get("fiscal_year", "")
            ).strip()

            if chunk_year not in expected_years:
                continue

            evidence_id = f"S{len(evidence_map) + 1}"
            evidence_map[evidence_id] = merged_chunk

        if not evidence_map:
            return None

        extraction_started = time.perf_counter()

        metric_result = extract_financial_metric(
            question,
            evidence_map,
        )

        extraction_elapsed = (
            time.perf_counter() - extraction_started
        )

        if metric_result is None:
            logger.info(
                "FinSight early financial metric extraction | "
                "no verified metric found | elapsed: %.3fs",
                extraction_elapsed,
            )
            return None

        evidence_chunk = evidence_map.get(
            metric_result.evidence_id
        )

        if evidence_chunk is None:
            logger.error(
                "FinSight early metric extraction returned "
                "an unknown evidence ID: %s",
                metric_result.evidence_id,
            )
            return None

        citation = Citation(
            document_name=str(
                evidence_chunk["source_doc"]
            ),
            page_number=int(
                evidence_chunk["page_number"]
            ),
            chunk_id=str(
                evidence_chunk["chunk_id"]
            ),
            evidence_text=str(
                evidence_chunk["text"]
            ),
        )

        answer = GeneratedAnswer(
            answer=metric_result.answer,
            citations=[citation],
            abstained=False,
        )

        logger.info(
            "FinSight early deterministic financial answer | "
            "metric: %s | company: %s | year: %d | basis: %s | "
            "value: %s | evidence: %s | elapsed: %.3fs | LLM skipped",
            metric_result.metric_name,
            next(iter(expected_symbols)),
            metric_result.fiscal_year_end,
            metric_result.reporting_basis,
            metric_result.display_value,
            evidence_chunk["chunk_id"],
            extraction_elapsed,
        )

        return answer

    async def answer(
        self,
        question: str,
    ) -> GeneratedAnswer:
        """Answer using verified SQL or year-aware grounded RAG."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be blank.")

        sql_started = time.perf_counter()

        if self.sql_database_path.is_file():
            connection = get_connection(
                self.sql_database_path
            )

            try:
                sql_answer = build_employee_count_answer_for_question(
                    connection,
                    question,
                )

                if sql_answer is not None:
                    return sql_answer

                unverified_employee_match = (
                    find_unverified_employee_fact_match(
                        connection,
                        question,
                    )
                )

                if unverified_employee_match is not None:
                    company = unverified_employee_match["company"]
                    fiscal_year = unverified_employee_match["fiscal_year"]

                    return GeneratedAnswer(
                        answer=(
                            "I couldn't provide a verified employee count "
                            f"for {company} in {fiscal_year} because one or "
                            "more employee-count records have not been "
                            "verified."
                        ),
                        citations=[],
                        abstained=True,
                    )

                unverified_match = (
                    find_unverified_financial_fact_match(
                        connection,
                        question,
                    )
                )

                if unverified_match is not None:
                    company = unverified_match["company"]
                    metric = unverified_match["metric"]
                    fiscal_year = unverified_match["fiscal_year"]

                    return GeneratedAnswer(
                        answer=(
                            "I couldn't provide a verified figure for "
                            f"{company}'s {metric.replace('_', ' ')} "
                            f"in {fiscal_year} because the available "
                            "financial record has not been verified."
                        ),
                        citations=[],
                        abstained=True,
                    )

            finally:
                connection.close()

                logger.info(
                    "FinSight timing | SQL lookup: %.3f seconds",
                    time.perf_counter() - sql_started,
                )

        # Deterministic cross-company revenue comparison.
        revenue_answer = _find_highest_revenue_answer(
            question=question,
            chunks=self.chunks,
        )

        if revenue_answer is not None:
            logger.info(
                "FinSight deterministic revenue comparison completed | "
                "citations: %d | abstained: %s",
                len(revenue_answer.citations),
                revenue_answer.abstained,
            )

            return revenue_answer

        # Resolve company and fiscal-year scope.
        scope_started = time.perf_counter()

        allowed_symbols = resolve_company_symbols(
            question,
            self.chunks,
        )

        allowed_fiscal_years = resolve_fiscal_years(
            question
        )

        logger.info(
            "FinSight timing | Company and year scope: %.3f seconds",
            time.perf_counter() - scope_started,
        )

        logger.info(
            "FinSight scope diagnostics | symbols: %s | years: %s",
            sorted(allowed_symbols) if allowed_symbols else None,
            sorted(allowed_fiscal_years)
            if allowed_fiscal_years
            else None,
        )

        # Verify that explicitly requested fiscal years are indexed.
        if allowed_fiscal_years is not None:
            scoped_chunks = [
                chunk
                for chunk in self.chunks
                if (
                    allowed_symbols is None
                    or str(
                        chunk.get("symbol", "")
                    ).strip().upper() in allowed_symbols
                )
            ]

            available_years = {
                str(chunk.get("fiscal_year", "")).strip()
                for chunk in scoped_chunks
                if chunk.get("fiscal_year")
            }

            missing_years = (
                allowed_fiscal_years - available_years
            )

            has_complete_year_metadata = (
                bool(scoped_chunks)
                and all(
                    bool(str(chunk.get("fiscal_year", "")).strip())
                    for chunk in scoped_chunks
                )
            )

            if missing_years and (
                not scoped_chunks or has_complete_year_metadata
            ):
                missing_text = ", ".join(
                    sorted(missing_years)
                )

                if allowed_symbols:
                    company_names = sorted(
                        {
                            str(
                                chunk.get(
                                    "company_name",
                                    chunk.get("symbol", ""),
                                )
                            )
                            for chunk in scoped_chunks
                            if str(
                                chunk.get("symbol", "")
                            ).strip().upper() in allowed_symbols
                        }
                    )

                    company_text = (
                        ", ".join(company_names)
                        or ", ".join(sorted(allowed_symbols))
                    )

                    message = (
                        f"I couldn't answer this question for "
                        f"{company_text} because the requested "
                        f"fiscal year(s) {missing_text} are not "
                        "available in the indexed annual reports."
                    )

                else:
                    message = (
                        "I couldn't answer this question because the "
                        f"requested fiscal year(s) {missing_text} are "
                        "not available in the indexed annual reports."
                    )

                return GeneratedAnswer(
                    answer=message,
                    citations=[],
                    abstained=True,
                )

        # Preserve the exact total-assets evidence path.
        exact_total_assets_evidence = (
            _find_exact_total_assets_evidence(
                question=question,
                chunks=self.chunks,
                allowed_symbols=allowed_symbols,
                allowed_fiscal_years=allowed_fiscal_years,
            )
        )

        if exact_total_assets_evidence:
            _log_ranked_chunks(
                "EXACT TOTAL-ASSETS EVIDENCE FROM FULL INDEX",
                exact_total_assets_evidence,
            )

        # Find exact operating/EBIT-margin evidence across the full index.
        # This prevents a relevant annual-report section from being missed
        # when ordinary hybrid retrieval or reranking favors unrelated text.
        exact_operating_margin_evidence = (
            _find_exact_operating_margin_evidence(
                question=question,
                chunks=self.chunks,
                allowed_symbols=allowed_symbols,
                allowed_fiscal_years=allowed_fiscal_years,
            )
        )

        if exact_operating_margin_evidence:
            _log_ranked_chunks(
                "EXACT OPERATING-MARGIN EVIDENCE FROM FULL INDEX",
                exact_operating_margin_evidence,
            )

        # First hybrid retrieval.
        retrieval_started = time.perf_counter()

        original_results = await asyncio.to_thread(
            hybrid_search,
            query=question,
            chunks=self.chunks,
            model=self.embedding_model,
            bm25_index=self.bm25_index,
            chunk_embeddings=self.chunk_embeddings,
            top_k=self.hybrid_top_k,
            candidate_k=self.candidate_k,
            allowed_symbols=allowed_symbols,
            allowed_fiscal_years=allowed_fiscal_years,
        )

        if (
            not original_results
            and (
                exact_total_assets_evidence
                or exact_operating_margin_evidence
            )
        ):
            original_results = [
                dict(chunk)
                for chunk in (
                    exact_operating_margin_evidence
                    + exact_total_assets_evidence
                )
            ]

        if (
            not original_results
            and not exact_total_assets_evidence
            and not exact_operating_margin_evidence
        ):
            logger.info(
                "FinSight timing | Hybrid retrieval: %.3f seconds | "
                "results: 0",
                time.perf_counter() - retrieval_started,
            )

            generation_started = time.perf_counter()

            answer = await generate_answer(
                question=question,
                evidence=[],
                llm=self.llm,
                top_k=self.rerank_top_k,
            )

            logger.info(
                "FinSight timing | Empty-evidence generation: %.3f seconds",
                time.perf_counter() - generation_started,
            )

            return answer

        logger.info(
            "FinSight timing | Initial hybrid retrieval: %.3f seconds | "
            "results: %d",
            time.perf_counter() - retrieval_started,
            len(original_results),
        )

        # -------------------------------------------------------------
        # Early deterministic financial metric extraction.
        # -------------------------------------------------------------

        # Put full-index margin evidence first so deterministic extraction
        # can use it even when hybrid retrieval omitted the relevant chunk.
        early_metric_evidence = (
            exact_operating_margin_evidence + original_results
        )

        early_metric_answer = (
            self._try_deterministic_financial_metric_answer(
                question=question,
                evidence=early_metric_evidence,
                allowed_symbols=allowed_symbols,
                allowed_fiscal_years=allowed_fiscal_years,
            )
        )

        if early_metric_answer is not None:
            logger.info(
                "FinSight early metric fast path completed | "
                "query expansion skipped | reranking skipped | "
                "LLM generation skipped."
            )

            return early_metric_answer

        # Continue through the normal retrieval path if early extraction
        # cannot verify a supported metric.
        candidates = original_results[
            :min(
                len(original_results),
                RERANK_CANDIDATE_LIMIT,
            )
        ]

        _log_ranked_chunks(
            "BEFORE RERANKING",
            candidates,
        )

        rerank_started = time.perf_counter()

        reranked_chunks = await asyncio.to_thread(
            rerank_chunks,
            query=question,
            chunks=candidates,
            model=self.reranker,
            top_k=self.rerank_top_k,
        )

        logger.info(
            "FinSight timing | Cross-encoder reranking: %.3f seconds | "
            "candidates: %d | returned: %d",
            time.perf_counter() - rerank_started,
            len(candidates),
            len(reranked_chunks),
        )

        exact_metric_evidence = (
            exact_operating_margin_evidence
            + exact_total_assets_evidence
        )

        reranked_chunks = _prioritize_exact_metric_evidence(
            exact_evidence=exact_metric_evidence,
            reranked_chunks=reranked_chunks,
            top_k=self.rerank_top_k,
        )

        _log_ranked_chunks(
            "AFTER RERANKING - FINAL GENERATION EVIDENCE",
            reranked_chunks,
        )

        generation_started = time.perf_counter()

        answer = await generate_answer(
            question=question,
            evidence=reranked_chunks,
            llm=self.llm,
            top_k=self.rerank_top_k,
        )

        logger.info(
            "FinSight timing | Answer generation: %.3f seconds",
            time.perf_counter() - generation_started,
        )

        logger.info(
            "FinSight answer diagnostics | abstained: %s | "
            "citation count: %d | answer: %s",
            answer.abstained,
            len(answer.citations),
            answer.answer[:300],
        )

        return answer
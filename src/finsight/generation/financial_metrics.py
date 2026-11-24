
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any


@dataclass(frozen=True)
class FinancialMetricResult:
    """A financial figure extracted from a labelled annual-report row."""

    metric_name: str
    evidence_id: str
    fiscal_year_end: int
    reporting_basis: str
    reported_value: str
    reported_unit: str | None
    display_value: str
    answer: str
    row_label: str


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "revenue_growth": {
        "question_patterns": (r"\brevenue\s+growth\b",),
        "row_patterns": (
            r"revenue\s+growth\s*\(\%\)",
            r"revenue\s+growth",
        ),
        "answer_label": "revenue growth",
        "value_type": "percentage",
        "ratio_table": True,
    },
    "operating_margin": {
        "question_patterns": (
            r"\boperating\s+margin\b",
            r"\bEBIT\s+margin\b",
        ),
        "row_patterns": (
            r"operating\s+margin\s*\(\%\)",
            r"operating\s+margin",
            r"EBIT\s+margin\s*\(\%\)",
            r"EBIT\s+margin",
        ),
        "answer_label": "operating margin",
        "value_type": "percentage",
        "ratio_table": True,
    },
    "net_profit_margin": {
        "question_patterns": (r"\bnet\s+profit\s+margin\b",),
        "row_patterns": (
            r"net\s+profit\s+margin\s*\(\%\)",
            r"net\s+profit\s+margin",
        ),
        "answer_label": "net profit margin",
        "value_type": "percentage",
        "ratio_table": True,
    },
    "return_on_equity": {
        "question_patterns": (
            r"\breturn\s+on\s+equity\b",
            r"\bROE\b",
        ),
        "row_patterns": (
            r"return\s+on\s+equity\s*\(\%\)",
            r"return\s+on\s+equity",
            r"ROE\s*\(\%\)",
            r"ROE",
        ),
        "answer_label": "return on equity",
        "value_type": "percentage",
        "ratio_table": True,
    },
    "current_ratio": {
        "question_patterns": (r"\bcurrent\s+ratio\b",),
        "row_patterns": (
            r"current\s+ratio\s*\(times\)",
            r"current\s+ratio",
        ),
        "answer_label": "current ratio",
        "value_type": "times",
        "ratio_table": True,
    },
    "debt_equity_ratio": {
        "question_patterns": (
            r"\bdebt[\s-]+equity\s+ratio\b",
            r"\bdebt[\s-]+to[\s-]+equity\s+ratio\b",
        ),
        "row_patterns": (
            r"debt[\s-]+equity\s+ratio\s*\(times\)",
            r"debt[\s-]+equity\s+ratio",
            r"debt[\s-]+to[\s-]+equity\s+ratio",
        ),
        "answer_label": "debt-equity ratio",
        "value_type": "times",
        "ratio_table": True,
    },
    "revenue": {
        "question_patterns": (r"\brevenue\b",),
        "row_patterns": (
            r"revenue\s+from\s+operations",
            r"revenues?",
        ),
        "answer_label": "revenue from operations",
        "value_type": "amount",
        "ratio_table": False,
    },
    "operating_profit": {
        "question_patterns": (
            r"\boperating\s+profit\b",
            r"\bprofit\s+from\s+operations\b",
        ),
        "row_patterns": (
            r"operating\s+profit",
            r"profit\s+from\s+operations",
        ),
        "answer_label": "operating profit",
        "value_type": "amount",
        "ratio_table": False,
    },
    "profit_before_tax": {
        "question_patterns": (
            r"\bprofit\s+before\s+tax\b",
            r"\bPBT\b",
        ),
        "row_patterns": (
            r"profit\s+before\s+tax",
            r"profit\s+before\s+exceptional\s+items\s+and\s+tax",
        ),
        "answer_label": "profit before tax",
        "value_type": "amount",
        "ratio_table": False,
    },
    "net_income": {
        "question_patterns": (
            r"\bnet\s+income\b",
            r"\bnet\s+profit\b",
            r"\bprofit\s+after\s+tax\b",
            r"\bPAT\b",
            r"\bprofit\s+for\s+the\s+year\b",
        ),
        "row_patterns": (
            r"profit\s+after\s+tax(?:\s*\(PAT\))?",
            r"profit\s+for\s+the\s+year",
            r"net\s+income",
            r"net\s+profit",
            r"PAT",
        ),
        "answer_label": "net income (profit after tax)",
        "value_type": "amount",
        "ratio_table": False,
    },
}


_YEAR_RANGE_PATTERN = re.compile(
    r"\b(?:FY\s*)?(20\d{2})\s*[-–—/]\s*(\d{2}|20\d{2})\b",
    re.I,
)

_YEAR_PATTERN = re.compile(r"\bFY\s*(20\d{2})\b", re.I)

_TABLE_YEAR_PATTERN = re.compile(
    r"\b(?:for\s+the\s+)?year\s+ended\s+"
    r"(?:march\s+31|31\s+march)\s*,?\s*(20\d{2})\b",
    re.I,
)

_BARE_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

_NUMERIC_PATTERN = re.compile(
    r"\(?-?\d[\d,]*(?:\.\d+)?\)?"
)

_ALLOWED_NUMERIC_CELL_REMAINDER = re.compile(
    r"\b(?:INR|USD|EUR|GBP|Rs\.?|I|C|K)\b",
    re.I,
)

_SUPPORTED_UNITS = {
    "million": "million",
    "millions": "million",
    "crore": "crore",
    "crores": "crore",
    "lakh": "lakh",
    "lakhs": "lakh",
}


def _requested_fiscal_year_end(question: str) -> int | None:
    """Resolve the ending calendar year of the requested financial year."""

    match = _YEAR_RANGE_PATTERN.search(question)

    if match:
        ending_year = match.group(2)

        if len(ending_year) == 2:
            return int(f"20{ending_year}")

        return int(ending_year)

    match = _YEAR_PATTERN.search(question)

    if match:
        return int(match.group(1))

    return None


def _requested_reporting_basis(question: str) -> str:
    """Default to consolidated unless standalone is explicitly requested."""

    if re.search(r"\bstandalone\b", question, re.I):
        return "standalone"

    return "consolidated"


def _identify_metric(question: str) -> str | None:
    """Identify a registered metric without guessing ambiguous variants."""

    if re.search(
        r"\battributable\b|"
        r"\bequity\s+holders\b|"
        r"\bowners\s+of\s+the\s+company\b",
        question,
        re.I,
    ):
        return None

    for metric_name, specification in METRIC_SPECS.items():
        if any(
            re.search(pattern, question, re.I)
            for pattern in specification["question_patterns"]
        ):
            return metric_name

    return None


def _normalise_row_label(line: str) -> str:
    """Remove annual-report footnote markers from a table row label."""

    normalized = " ".join(line.split())

    normalized = re.sub(
        r"(?:\s*\(\d+\))+\s*$",
        "",
        normalized,
    )

    return normalized.strip()


def _has_requested_statement_basis(
    text: str,
    reporting_basis: str,
) -> bool:
    """Verify that the chunk belongs to the requested statement basis."""

    escaped_basis = re.escape(reporting_basis)

    financial_statement_pattern = re.compile(
        rf"\b{escaped_basis}\s+financial\s+statements?\b",
        re.I,
    )

    income_statement_pattern = re.compile(
        rf"\b{escaped_basis}\s+statements?\s+of\s+"
        r"(?:income|profit\s+and\s+loss)\b",
        re.I,
    )

    financial_results_pattern = re.compile(
        rf"\b{escaped_basis}\s+financial\s+results?\b",
        re.I,
    )

    # Preserve the existing formal financial-statement checks.
    if (
        financial_statement_pattern.search(text)
        or income_statement_pattern.search(text)
        or financial_results_pattern.search(text)
    ):
        return True

    # Also recognize Financial Results tables where the reporting
    # basis is represented by column headings.
    results_heading = re.search(
        r"\bfinancial\s+results?\b",
        text,
        re.I,
    )

    if not results_heading:
        return False

    header_window = text[
        results_heading.start():
        results_heading.start() + 2000
    ]

    table_header = re.search(
        r"\bparticulars\b"
        r"[\s\S]{0,250}\bconsolidated\b"
        r"[\s\S]{0,250}\bstandalone\b",
        header_window,
        re.I,
    )

    if not table_header:
        return False

    return bool(
        re.search(
            rf"\b{escaped_basis}\b",
            table_header.group(0),
            re.I,
        )
    )


def _find_year_columns(
    text: str,
    row_position: int,
    target_year: int,
) -> tuple[list[int], int] | None:
    """Find the requested year column in an income-statement table."""

    header_start = max(0, row_position - 3500)
    header_text = text[header_start:row_position]

    year_matches = list(
        _TABLE_YEAR_PATTERN.finditer(header_text)
    )

    if not year_matches:
        return None

    years = [
        int(match.group(1))
        for match in year_matches
    ]

    try:
        target_column = years.index(target_year)
    except ValueError:
        return None

    return years, target_column


def _find_ratio_year_columns(
    text: str,
    row_position: int,
    target_year: int,
    reporting_basis: str,
) -> tuple[list[int], int] | None:
    """
    Resolve columns from a table structured like:

        Particulars | Standalone | Consolidated
                     2026  2025    2026  2025
    """

    header_start = max(0, row_position - 2500)
    header_text = text[header_start:row_position]

    standalone_matches = list(
        re.finditer(r"\bstandalone\b", header_text, re.I)
    )

    consolidated_matches = list(
        re.finditer(r"\bconsolidated\b", header_text, re.I)
    )

    if not standalone_matches or not consolidated_matches:
        return None

    standalone_match = standalone_matches[-1]
    consolidated_match = consolidated_matches[-1]

    if abs(
        standalone_match.start() - consolidated_match.start()
    ) > 1000:
        return None

    first_basis = (
        "standalone"
        if standalone_match.start() < consolidated_match.start()
        else "consolidated"
    )

    years_start = max(
        standalone_match.end(),
        consolidated_match.end(),
    )

    years_text = header_text[years_start:]

    all_years = [
        int(value)
        for value in re.findall(
            r"\b(?:19|20)\d{2}\b",
            years_text,
        )
    ]

    selected_group: list[int] | None = None
    selected_column_count = 0

    for column_count in range(len(all_years), 3, -2):
        if column_count % 2:
            continue

        group_size = column_count // 2

        first_group = all_years[:group_size]
        second_group = all_years[group_size:column_count]

        if first_group == second_group:
            selected_group = first_group
            selected_column_count = column_count
            break

    if not selected_group or target_year not in selected_group:
        return None

    year_index = selected_group.index(target_year)
    group_size = len(selected_group)

    if reporting_basis == first_basis:
        target_column = year_index
    else:
        target_column = group_size + year_index

    years = all_years[:selected_column_count]

    return years, target_column


def _is_numeric_cell_line(line: str) -> bool:
    """Check whether a line contains only numeric table-cell content."""

    stripped = line.strip()

    if not stripped:
        return False

    if stripped in {"-", "–", "—"}:
        return True

    remainder = _NUMERIC_PATTERN.sub("", stripped)
    remainder = _ALLOWED_NUMERIC_CELL_REMAINDER.sub("", remainder)

    remainder = re.sub(
        r"[\s₹$€£,.\-–—()%]",
        "",
        remainder,
    )

    return not remainder


def _parse_numeric_value(value: str) -> Decimal | None:
    """Convert an annual-report numeric cell to Decimal."""

    if not value:
        return None

    is_negative = (
        value.startswith("(")
        and value.endswith(")")
    )

    normalized = (
        value.replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )

    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None

    return -number if is_negative else number


def _collect_row_values(
    lines: list[str],
    row_index: int,
    expected_columns: int,
    row_patterns: tuple[str, ...],
    allow_small_values: bool = False,
) -> list[str | None] | None:
    """Collect numeric values following a labelled financial row."""

    if expected_columns <= 0:
        return None

    values: list[str | None] = []
    current_line = lines[row_index]

    for pattern in row_patterns:
        tail_match = re.match(
            rf"^\s*(?:{pattern})"
            r"(?:\s*\(\d+\))*"
            r"\s*:?\s*(.*)$",
            current_line,
            re.I,
        )

        if not tail_match:
            continue

        tail_values = _NUMERIC_PATTERN.findall(
            tail_match.group(1)
        )

        if not allow_small_values:
            while len(tail_values) > expected_columns and tail_values:
                first_value = _parse_numeric_value(tail_values[0])

                if first_value is None or abs(first_value) > 100:
                    break

                tail_values.pop(0)

            if len(tail_values) == 1 and expected_columns > 1:
                first_value = _parse_numeric_value(tail_values[0])

                if first_value is not None and abs(first_value) <= 100:
                    tail_values = []

        if len(tail_values) == expected_columns:
            values.extend(tail_values)

        break

    started = bool(values)

    following_lines = lines[
        row_index + 1:
        row_index + 1 + expected_columns * 3 + 8
    ]

    for line in following_lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped in {"-", "–", "—"}:
            values.append(None)

        elif _is_numeric_cell_line(stripped):
            parsed_values = _NUMERIC_PATTERN.findall(stripped)

            if not parsed_values:
                continue

            if (
                not allow_small_values
                and not started
                and len(parsed_values) == 1
                and expected_columns > 1
            ):
                first_value = _parse_numeric_value(parsed_values[0])

                if first_value is not None and abs(first_value) <= 100:
                    continue

            values.extend(parsed_values)
            started = True

        else:
            break

        if len(values) >= expected_columns:
            break

    if len(values) != expected_columns:
        return None

    return values


def _format_indian_integer(value: Decimal) -> str:
    """Format an integer using Indian comma grouping."""

    if value != value.to_integral_value():
        raise ValueError("Expected an integer value.")

    integer_value = str(abs(int(value)))
    sign = "-" if value < 0 else ""

    if len(integer_value) <= 3:
        return sign + integer_value

    last_three = integer_value[-3:]
    remaining = integer_value[:-3]
    groups = []

    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    return sign + ",".join(groups + [last_three])


def _format_indian_decimal(value: Decimal) -> str:
    """Format a number with Indian grouping and trimmed decimal places."""

    integer_part = int(value)
    fraction = abs(value - Decimal(integer_part))

    formatted_integer = _format_indian_integer(
        Decimal(integer_part)
    )

    if fraction == 0:
        return formatted_integer

    fraction_text = (
        format(fraction, "f")
        .split(".")[-1]
        .rstrip("0")
    )

    return f"{formatted_integer}.{fraction_text}"


def _extract_reported_unit(
    text: str,
    chunk: dict[str, Any],
) -> tuple[str | None, bool]:
    """Detect the reporting unit and primary currency."""

    unit_match = re.search(
        r"\b(?:all\s+amounts|amounts|figures)\s+in\s+"
        r"(millions?|crores?|lakhs?)\b"
        r"|\bin\s+(millions?|crores?|lakhs?)\b",
        text,
        re.I,
    )

    unit = None

    if unit_match:
        raw_unit = next(
            (
                group
                for group in unit_match.groups()
                if group
            ),
            "",
        ).casefold()

        unit = _SUPPORTED_UNITS.get(raw_unit)

    inr_marker = bool(
        re.search(r"₹|\bINR\b|\bRs\.?", text, re.I)
        or re.search(
            r"\bI\s+in\s+(?:millions?|crores?|lakhs?)\b",
            text,
            re.I,
        )
    )

    usd_convenience_column = bool(
        re.search(
            r"convenience\s+translation\s+into\s+U\.?S\.?\s+Dollar",
            text,
            re.I,
        )
    )

    indian_issuer = bool(
        str(chunk.get("symbol", "")).strip()
        or str(chunk.get("company_name", "")).strip()
    )

    is_inr = inr_marker or (
        indian_issuer and usd_convenience_column
    )

    return unit, is_inr


def _format_display_value(
    value: Decimal,
    unit: str | None,
    is_inr: bool,
) -> str:
    """Format reported amounts and convert INR millions to crore."""

    formatted = _format_indian_decimal(value)
    currency = "₹" if is_inr else ""

    if unit == "million":
        reported_amount = f"{currency}{formatted} million"

        if is_inr:
            crore_value = value / Decimal("10")
            converted = _format_indian_decimal(crore_value)

            return (
                f"{reported_amount} "
                f"({currency}{converted} crore)"
            )

        return reported_amount

    if unit:
        return f"{currency}{formatted} {unit}"

    return f"{currency}{formatted}"


def _format_fiscal_year(target_year: int) -> str:
    """Format a fiscal year ending in the supplied calendar year."""

    return f"FY {target_year - 1}–{str(target_year)[-2:]}"


def _extract_narrative_operating_margin(
    question: str,
    evidence_id: str,
    chunk: dict[str, Any],
) -> FinancialMetricResult | None:
    """
    Extract an explicitly reported EBIT margin from a financial-performance
    narrative section.
    """

    if _identify_metric(question) != "operating_margin":
        return None

    target_year = _requested_fiscal_year_end(question)

    if target_year is None:
        return None

    reporting_basis = _requested_reporting_basis(question)

    text = chunk.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return None

    normalized_text = " ".join(text.split())

    heading = re.search(
        rf"\b{re.escape(reporting_basis)}\s+financial\s+performance\b",
        normalized_text,
        re.I,
    )

    if heading is None:
        return None

    section = normalized_text[heading.end():]

    year_range = (
        rf"{target_year - 1}\s*[-–—/]\s*{str(target_year)[-2:]}"
    )

    if not re.search(
        rf"\bfinancial\s+year\s+{year_range}\b",
        section,
        re.I,
    ):
        return None

    margin_match = re.search(
        r"\bEBIT\s+Margin\b.{0,250}?"
        r"\b(?:stood\s+at|was|is)\s+"
        r"(-?\d+(?:\.\d+)?)\s*%",
        section,
        re.I,
    )

    if margin_match is None:
        return None

    reported_value = margin_match.group(1)
    numeric_value = _parse_numeric_value(reported_value)

    if numeric_value is None:
        return None

    display_value = f"{_format_indian_decimal(numeric_value)}%"
    fiscal_year = _format_fiscal_year(target_year)

    answer = (
        f"The {reporting_basis} operating margin "
        f"(reported as EBIT Margin) for {fiscal_year} "
        f"was {display_value}."
    )

    return FinancialMetricResult(
        metric_name="operating_margin",
        evidence_id=evidence_id,
        fiscal_year_end=target_year,
        reporting_basis=reporting_basis,
        reported_value=reported_value,
        reported_unit="%",
        display_value=display_value,
        answer=answer,
        row_label="EBIT Margin (narrative)",
    )


def extract_financial_metric(
    question: str,
    evidence_map: dict[str, dict[str, Any]],
) -> FinancialMetricResult | None:
    """
    Extract a registered metric from annual-report evidence.

    Supports statement tables, standalone/consolidated ratio tables,
    and explicitly reported operating margins in financial-performance
    narrative sections.

    Returns None when the requested year, basis, row, or values cannot
    be reliably identified.
    """

    if not isinstance(question, str) or not question.strip():
        return None

    metric_name = _identify_metric(question)

    if metric_name is None:
        return None

    target_year = _requested_fiscal_year_end(question)

    if target_year is None:
        return None

    reporting_basis = _requested_reporting_basis(question)
    specification = METRIC_SPECS[metric_name]

    row_patterns = tuple(specification["row_patterns"])
    value_type = specification["value_type"]
    allow_ratio_table = bool(specification.get("ratio_table", False))

    candidates: list[tuple[int, FinancialMetricResult]] = []

    for evidence_id, chunk in evidence_map.items():
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        # First check for an explicitly reported operating margin in
        # a financial-performance narrative section.
        narrative_result = _extract_narrative_operating_margin(
            question=question,
            evidence_id=evidence_id,
            chunk=chunk,
        )

        if narrative_result is not None:
            # Conventional financial statement or ratio-table evidence
            # retains precedence over narrative evidence.
            candidates.append((1, narrative_result))

        lines = text.splitlines()

        statement_basis_available = _has_requested_statement_basis(
            text,
            reporting_basis,
        )

        # Ratio tables do not necessarily contain a conventional
        # income-statement heading. Their column headings are verified
        # below at the actual position of each matched metric row.
        if not statement_basis_available and not allow_ratio_table:
            continue

        if (
            value_type == "amount"
            and re.search(
                r"statements?\s+of\s+comprehensive\s+income",
                text,
                re.I,
            )
        ):
            continue

        for row_index, line in enumerate(lines):
            normalized_line = _normalise_row_label(line)
            matched_label = None

            for pattern in row_patterns:
                if re.fullmatch(
                    pattern,
                    normalized_line,
                    re.I,
                ):
                    matched_label = normalized_line
                    break

            if matched_label is None:
                continue

            row_position = sum(
                len(previous_line) + 1
                for previous_line in lines[:row_index]
            )

            ratio_columns = None

            if allow_ratio_table:
                ratio_columns = _find_ratio_year_columns(
                    text=text,
                    row_position=row_position,
                    target_year=target_year,
                    reporting_basis=reporting_basis,
                )

            if ratio_columns is not None:
                header_years, target_column = ratio_columns
                is_ratio_table = True
            else:
                if not statement_basis_available:
                    continue

                header_result = _find_year_columns(
                    text,
                    row_position,
                    target_year,
                )

                if header_result is None:
                    continue

                header_years, target_column = header_result
                is_ratio_table = False

            if target_column >= len(header_years):
                continue

            values = _collect_row_values(
                lines=lines,
                row_index=row_index,
                expected_columns=len(header_years),
                row_patterns=row_patterns,
                allow_small_values=(
                    value_type in {"percentage", "times"}
                ),
            )

            if values is None:
                continue

            reported_cell = values[target_column]

            if reported_cell is None:
                continue

            numeric_value = _parse_numeric_value(reported_cell)

            if numeric_value is None:
                continue

            if value_type == "percentage":
                reported_unit = "%"
                display_value = (
                    f"{_format_indian_decimal(numeric_value)}%"
                )

            elif value_type == "times":
                reported_unit = "times"
                display_value = (
                    f"{_format_indian_decimal(numeric_value)} times"
                )

            else:
                reported_unit, is_inr = _extract_reported_unit(
                    text,
                    chunk,
                )

                display_value = _format_display_value(
                    numeric_value,
                    reported_unit,
                    is_inr,
                )

            fiscal_year = _format_fiscal_year(target_year)
            answer_label = specification["answer_label"]

            answer = (
                f"The {reporting_basis} {answer_label} "
                f"for {fiscal_year} was {display_value}."
            )

            result = FinancialMetricResult(
                metric_name=metric_name,
                evidence_id=evidence_id,
                fiscal_year_end=target_year,
                reporting_basis=reporting_basis,
                reported_value=reported_cell,
                reported_unit=reported_unit,
                display_value=display_value,
                answer=answer,
                row_label=matched_label,
            )

            # Prefer a standard financial statement over a ratio-table
            # value when both provide the same requested metric.
            priority = 0 if is_ratio_table else 1

            if re.search(
                r"statements?\s+of\s+(?:income|profit\s+and\s+loss)",
                text,
                re.I,
            ):
                priority -= 2

            candidates.append((priority, result))
            break

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]

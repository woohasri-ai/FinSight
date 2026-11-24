import re


def resolve_fiscal_years(question: str) -> set[str] | None:
    """
    Resolve explicitly requested fiscal years.

    Supports both hyphenated ranges (e.g. 'FY 2025-26', '2024-25')
    and single-year fiscal notations (e.g. 'FY2026', 'FY 2026', 'FY26').

    Returns normalized YYYY-YY values (e.g. {'2025-26'}), or None when
    no fiscal year is explicitly mentioned.
    """
    if not isinstance(question, str) or not question.strip():
        return None

    years: set[str] = set()

    # Step 1: Resolve explicit year ranges (e.g. '2025-26', 'FY 2024-25', '2023/24').
    range_patterns = [
        r"\b(?:FY|financial\s+year|fiscal\s+year)\s*"
        r"(20\d{2})\s*[-/–—]\s*(\d{2}|20\d{2})\b",
        r"\b(20\d{2})\s*[-/–—]\s*(\d{2}|20\d{2})\b",
    ]

    masked_question = question

    for pattern in range_patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            start_year = int(match.group(1))
            end_text = match.group(2)

            end_year = (
                int(end_text)
                if len(end_text) == 4
                else (start_year // 100) * 100 + int(end_text)
            )

            if end_year == start_year + 1:
                years.add(f"{start_year}-{str(end_year)[-2:]}")

            # Mask out the matched range so single-year matching does not double-count it.
            span = match.span()
            masked_question = (
                masked_question[: span[0]]
                + " " * (span[1] - span[0])
                + masked_question[span[1] :]
            )

    # Step 2: Resolve single-year fiscal notations (e.g. 'FY2026', 'FY 2026', 'FY26', 'FY'26').
    # In Indian financial reporting, FY2026 / FY26 denotes the year ending March 31, 2026
    # (i.e. '2025-26').
    single_year_pattern = (
        r"\b(?:FY|financial\s+year|fiscal\s+year)\s*['’]?\s*(20\d{2}|\d{2})\b"
    )

    for match in re.finditer(single_year_pattern, masked_question, flags=re.IGNORECASE):
        raw_year = match.group(1)

        if len(raw_year) == 4:
            end_year = int(raw_year)
        else:
            end_year = 2000 + int(raw_year)

        if 2000 <= end_year <= 2100:
            start_year = end_year - 1
            years.add(f"{start_year}-{str(end_year)[-2:]}")

    return years or None

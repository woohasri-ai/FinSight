from finsight.retrieval.year_scope import resolve_fiscal_years


def test_resolves_hyphenated_range():
    assert resolve_fiscal_years("What was TCS revenue in FY 2025-26?") == {"2025-26"}
    assert resolve_fiscal_years("What was Infosys revenue in 2024-25?") == {"2024-25"}
    assert resolve_fiscal_years("What was Wipro revenue in financial year 2023-24?") == {"2023-24"}


def test_resolves_full_four_digit_range():
    assert resolve_fiscal_years("What was TCS revenue in FY 2024-2025?") == {"2024-25"}
    assert resolve_fiscal_years("Revenue in 2025/2026") == {"2025-26"}


def test_resolves_four_digit_single_year_notation():
    assert resolve_fiscal_years("What was TCS revenue in FY2026?") == {"2025-26"}
    assert resolve_fiscal_years("What was TCS revenue in FY 2026?") == {"2025-26"}
    assert resolve_fiscal_years("What was Infosys revenue in FY2025?") == {"2024-25"}
    assert resolve_fiscal_years("What was Wipro profit in financial year 2024?") == {"2023-24"}
    assert resolve_fiscal_years("What was LTM margin in fiscal year 2026?") == {"2025-26"}


def test_resolves_two_digit_single_year_notation():
    assert resolve_fiscal_years("What was TCS revenue in FY26?") == {"2025-26"}
    assert resolve_fiscal_years("What was TCS revenue in FY 26?") == {"2025-26"}
    assert resolve_fiscal_years("What was Infosys revenue in FY'25?") == {"2024-25"}
    assert resolve_fiscal_years("What was Wipro revenue in FY24?") == {"2023-24"}


def test_resolves_multiple_years():
    assert resolve_fiscal_years("Compare TCS revenue in FY25 and FY26") == {"2024-25", "2025-26"}
    expected = {"2023-24", "2024-25"}
    assert resolve_fiscal_years("Revenue growth between FY 2023-24 and FY 2024-25") == expected


def test_returns_none_when_no_fiscal_year_mentioned():
    assert resolve_fiscal_years("What is the IT market outlook?") is None
    assert resolve_fiscal_years("What is TCS revenue?") is None
    assert resolve_fiscal_years("As of March 31, 2026") is None


def test_handles_empty_or_invalid_inputs():
    assert resolve_fiscal_years("") is None
    assert resolve_fiscal_years("   ") is None
    assert resolve_fiscal_years(None) is None  # type: ignore[arg-type]

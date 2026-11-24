import pytest

from finsight.retrieval.company_scope import resolve_company_symbols


@pytest.fixture
def chunks():
    return [
        {
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services Ltd.",
        },
        {
            "symbol": "INFY",
            "company_name": "Infosys Ltd.",
        },
        {
            "symbol": "HCLTECH",
            "company_name": "HCL Technologies Ltd.",
        },
    ]


def test_resolves_company_by_symbol(chunks):
    assert resolve_company_symbols("What is TCS revenue?", chunks) == {"TCS"}


def test_resolves_company_by_name(chunks):
    assert resolve_company_symbols("What is Infosys revenue?", chunks) == {"INFY"}


def test_resolves_multiple_companies(chunks):
    assert resolve_company_symbols(
        "Compare Infosys and HCL Technologies",
        chunks,
    ) == {"INFY", "HCLTECH"}


def test_general_question_returns_unrestricted_scope(chunks):
    assert resolve_company_symbols(
        "What is the IT market outlook?",
        chunks,
    ) is None


def test_unknown_company_returns_unrestricted_scope(chunks):
    assert resolve_company_symbols(
        "What is Reliance revenue?",
        chunks,
    ) is None


def test_does_not_match_partial_symbol(chunks):
    assert resolve_company_symbols("What is TCSXYZ revenue?", chunks) is None

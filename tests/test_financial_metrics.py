
import pytest

from finsight.generation.financial_metrics import extract_financial_metric


@pytest.fixture
def wipro_consolidated_evidence():
    return {
        "S1": {
            "source_doc": "wipro_annual_report_2025_26.pdf",
            "page_number": 178,
            "chunk_id": "wipro_annual_report_2025_26.pdf:page-178:chunk-1",
            "symbol": "WIPRO",
            "fiscal_year": "2025-26",
            "text": (
                "WIPRO INTEGRATED ANNUAL REPORT 2025-26\n"
                "STANDALONE FINANCIAL STATEMENT UNDER IND AS\n"
                "Statements of Profit and Loss\n"
                "(I in millions, except share and per share data, "
                "unless otherwise stated)\n"
                "Year ended\nMarch 31, 2026\n"
                "Year ended\nMarch 31, 2025\n"
                "Profit for the year\n"
                "121,296\n"
                "109,131\n"
            ),
        },
        "S2": {
            "source_doc": "wipro_annual_report_2025_26.pdf",
            "page_number": 396,
            "chunk_id": "wipro_annual_report_2025_26.pdf:page-396:chunk-1",
            "symbol": "WIPRO",
            "fiscal_year": "2025-26",
            "text": (
                "WIPRO INTEGRATED ANNUAL REPORT 2025-26\n"
                "CONSOLIDATED FINANCIAL STATEMENTS UNDER IFRS\n"
                "(I in millions, except share and per share data, "
                "unless otherwise stated)\n"
                "Year ended\nMarch 31, 2024\n"
                "Year ended\nMarch 31, 2025\n"
                "Year ended\nMarch 31, 2026\n"
                "Year ended\nMarch 31, 2026\n"
                "Convenience translation into U.S. Dollar in millions "
                "(unaudited)\n"
                "Profit for the year\n"
                "111,121\n"
                "132,180\n"
                "132,655\n"
                "1,414\n"
            ),
        },
    }


def test_wipro_fy2026_selects_correct_consolidated_net_income(
    wipro_consolidated_evidence,
):
    result = extract_financial_metric(
        "What was Wipro net income in FY 2025-26?",
        wipro_consolidated_evidence,
    )

    assert result is not None
    assert result.metric_name == "net_income"
    assert result.evidence_id == "S2"
    assert result.fiscal_year_end == 2026
    assert result.reporting_basis == "consolidated"
    assert result.reported_value == "132,655"

    assert result.display_value == (
        "₹1,32,655 million (₹13,265.5 crore)"
    )

    assert "13,265.5 crore" in result.answer


def test_wipro_fy2025_selects_correct_year_column(
    wipro_consolidated_evidence,
):
    result = extract_financial_metric(
        "What was Wipro net income in FY 2024-25?",
        wipro_consolidated_evidence,
    )

    assert result is not None
    assert result.evidence_id == "S2"
    assert result.fiscal_year_end == 2025
    assert result.reported_value == "132,180"
    assert result.display_value == (
        "₹1,32,180 million (₹13,218 crore)"
    )


def test_wipro_default_basis_does_not_select_standalone_evidence(
    wipro_consolidated_evidence,
):
    standalone_only = {
        "S1": wipro_consolidated_evidence["S1"],
    }

    result = extract_financial_metric(
        "What was Wipro net income in FY 2025-26?",
        standalone_only,
    )

    assert result is None


def test_wipro_explicit_standalone_question_selects_standalone_evidence(
    wipro_consolidated_evidence,
):
    standalone_only = {
        "S1": wipro_consolidated_evidence["S1"],
    }

    result = extract_financial_metric(
        "What was Wipro standalone net income in FY 2025-26?",
        standalone_only,
    )

    assert result is not None
    assert result.reporting_basis == "standalone"
    assert result.evidence_id == "S1"
    assert result.reported_value == "121,296"
    assert result.display_value == (
        "₹1,21,296 million (₹12,129.6 crore)"
    )


def test_wipro_net_income_does_not_select_profit_attributable(
    wipro_consolidated_evidence,
):
    result = extract_financial_metric(
        "What was Wipro net income attributable to equity holders "
        "in FY 2025-26?",
        wipro_consolidated_evidence,
    )

    assert result is None


def test_wipro_net_income_requires_explicit_fiscal_year(
    wipro_consolidated_evidence,
):
    result = extract_financial_metric(
        "What was Wipro net income?",
        wipro_consolidated_evidence,
    )

    assert result is None


def test_wipro_net_income_never_selects_usd_convenience_column(
    wipro_consolidated_evidence,
):
    result = extract_financial_metric(
        "What was Wipro net income in FY 2025-26?",
        wipro_consolidated_evidence,
    )

    assert result is not None
    assert result.reported_value == "132,655"
    assert result.reported_value != "1,414"
def test_tech_mahindra_extracts_consolidated_ebit_margin_from_narrative():
    evidence = {
        "TECHM_PAGE_127": {
            "source_doc": "techmahindra_annual_report_2025_26.pdf",
            "page_number": 127,
            "chunk_id": (
                "techmahindra_annual_report_2025_26.pdf:"
                "page-127:chunk-2"
            ),
            "company_name": "Tech Mahindra Ltd.",
            "symbol": "TECHM",
            "fiscal_year": "2025-26",
            "text": (
                "Consolidated Financial Performance\n"
                "For the financial year 2025-26, the Company "
                "reported Revenue from Operations of ` 568,154 Mn. "
                "Operating Profit for the financial year under "
                "review, rose to ` 71,525 Mn. The resultant "
                "EBIT Margin for the financial year 2025-26 "
                "stood at 12.6%, an expansion of 290 basis "
                "points over the previous year."
            ),
        }
    }

    result = extract_financial_metric(
        "What was Tech Mahindra operating margin in FY 2025-26?",
        evidence,
    )

    assert result is not None
    assert result.metric_name == "operating_margin"
    assert result.reporting_basis == "consolidated"
    assert result.fiscal_year_end == 2026
    assert result.reported_value == "12.6"
    assert result.display_value == "12.6%"
    assert result.evidence_id == "TECHM_PAGE_127"
    assert "12.6%" in result.answer
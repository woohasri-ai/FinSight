
import json

import pytest

from finsight.generation.financial_metrics import extract_financial_metric
from finsight.generation.generator import generate_answer


def test_hcltech_net_income_fy2026():
    chunk = {
        "source_doc": "hcltech_annual_report_2025_26.pdf",
        "page_number": 129,
        "chunk_id": "hcltech_annual_report_2025_26.pdf:page-129:chunk-1",
        "text": (
            "Financial Results\n"
            "Particulars\n"
            "Consolidated\n"
            "Standalone\n"
            "Year ended March 31, 2026\n"
            "Year ended March 31, 2025\n"
            "Profit for the year\n"
            "16,652\n"
            "17,399\n"
        ),
        "metadata": {
            "symbol": "HCLTECH",
            "fiscal_year": "2025-26",
        },
    }

    result = extract_financial_metric(
        question="What was HCLTech net income in FY 2025-26?",
        evidence_map={"S5": chunk},
    )

    assert result is not None
    assert result.metric_name == "net_income"
    assert result.evidence_id == "S5"
    assert result.reporting_basis == "consolidated"
    assert result.fiscal_year_end == 2026
    assert result.reported_value == "16,652"


def test_infosys_net_profit_fy2026():
    chunk = {
        "source_doc": "infosys_annual_report_2025_26.pdf",
        "page_number": 288,
        "chunk_id": "infosys_annual_report_2025_26.pdf:page-288:chunk-1",
        "text": (
            "Consolidated Statement of Profit and Loss\n"
            "Year ended March 31, 2026\n"
            "Year ended March 31, 2025\n"
            "Profit for the year\n"
            "29,474\n"
            "26,750\n"
        ),
        "metadata": {
            "symbol": "INFY",
            "fiscal_year": "2025-26",
        },
    }

    result = extract_financial_metric(
        question="What was Infosys net profit in FY 2025-26?",
        evidence_map={"S4": chunk},
    )

    assert result is not None
    assert result.metric_name == "net_income"
    assert result.evidence_id == "S4"
    assert result.reporting_basis == "consolidated"
    assert result.fiscal_year_end == 2026
    assert result.reported_value == "29,474"


class S1ReturningLLM:
    """Mock LLM returning an invalid internal evidence ID as its answer."""

    async def generate(
        self,
        prompt,
        *,
        system=None,
        response_format=None,
        num_predict=None,
    ):
        return json.dumps(
            {
                "answer": "S1",
                "excerpts": [{"evidence_id": "S1"}],
                "abstained": False,
            }
        )


@pytest.mark.asyncio
async def test_answer_must_not_be_internal_evidence_id():
    chunk = {
        "source_doc": "tcs_annual_report_2025_26.pdf",
        "page_number": 10,
        "chunk_id": "tcs_annual_report_2025_26.pdf:page-10:chunk-1",
        "text": "TCS reported revenue for the financial year.",
        "metadata": {
            "symbol": "TCS",
            "fiscal_year": "2025-26",
        },
    }

    result = await generate_answer(
        question="What was TCS revenue?",
        evidence=[chunk],
        llm=S1ReturningLLM(),
    )

    assert result.answer.strip() != "S1"

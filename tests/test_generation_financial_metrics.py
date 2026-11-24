
from unittest.mock import AsyncMock

import pytest

from finsight.generation.generator import generate_answer


@pytest.fixture
def wipro_evidence():
    return [
        {
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
        {
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
    ]


@pytest.mark.asyncio
async def test_generator_returns_wipro_fy2026_consolidated_net_income(
    wipro_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was Wipro net income in FY 2025-26?",
        evidence=wipro_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "₹1,32,655 million" in result.answer
    assert "₹13,265.5 crore" in result.answer

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.page_number == 396
    assert citation.chunk_id == (
        "wipro_annual_report_2025_26.pdf:page-396:chunk-1"
    )
    assert "132,655" in citation.evidence_text

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generator_respects_explicit_standalone_net_income(
    wipro_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was Wipro standalone net income in FY 2025-26?",
        evidence=wipro_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "₹12,129.6 crore" in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 178

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generator_abstains_when_only_standalone_evidence_is_available(
    wipro_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was Wipro net income in FY 2025-26?",
        evidence=[wipro_evidence[0]],
        llm=llm,
    )

    assert result.abstained is True
    assert result.citations == []

    llm.generate.assert_not_awaited()

import pytest
from unittest.mock import AsyncMock

from finsight.generation.generator import generate_answer


@pytest.fixture
def fy2026_evidence():
    return [
        {
            "source_doc": "tcs_annual_report_2025_26.pdf",
            "page_number": 58,
            "chunk_id": "tcs:page-58:chunk-1",
            "text": (
                "The following table gives an overview of the consolidated "
                "financial results of the Company:\n"
                "(H crore)\n"
                "FY 2026 FY 2025\n"
                "Revenue from operations\n"
                "267,021 255,324\n"
                "Profit after tax attributable to shareholders of the Company\n"
                "52,820 48,553\n"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_fy2025_26_profit_question_uses_deterministic_extraction(
    fy2026_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was TCS's net profit for FY2025–26?",
        evidence=fy2026_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "52,820" in result.answer
    assert "48,553" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 58
    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_fy2026_profit_question_uses_deterministic_extraction(
    fy2026_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was TCS's net profit for FY2026?",
        evidence=fy2026_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "52,820" in result.answer
    llm.generate.assert_not_awaited()

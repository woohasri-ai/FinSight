
import json
from unittest.mock import AsyncMock

import pytest

from finsight.generation.generator import (
    GenerationError,
    build_evidence_map,
    build_response_schema,
    generate_answer,
    handle_empty_evidence,
    parse_llm_response,
    validate_evidence_excerpts,
)
from finsight.generation.prompts import SYSTEM_PROMPT
from finsight.generation.schemas import (
    EvidenceExcerpt,
    LLMGenerationResponse,
)
from finsight.llm.ollama import (
    OllamaConnectionError,
    OllamaTimeoutError,
)


@pytest.fixture
def sample_evidence():
    return [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "Total employee strength was 584,519.",
        },
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 58,
            "chunk_id": "tcs:page-58:chunk-2",
            "text": "Revenue from operations was INR 267,021 crore.",
        },
    ]


@pytest.fixture
def techmahindra_operating_margin_evidence():
    return [
        {
            "source_doc": "techmahindra_annual_report_2025_26.pdf",
            "page_number": 127,
            "chunk_id": (
                "techmahindra_annual_report_2025_26.pdf:"
                "page-127:chunk-2"
            ),
            "text": (
                "Consolidated Financial Performance\n"
                "For the financial year 2025-26, the Company reported "
                "Revenue from Operations of ` 568,154 Mn, representing "
                "a 7.2% growth over the previous year. Operating Profit "
                "for the financial year under review, rose to ` 71,525 Mn, "
                "up by 39.2% from ` 51,382 Mn in the previous year. "
                "The resultant EBIT Margin for the financial year "
                "2025-26 stood at 12.6%, an expansion of 290 basis "
                "points over the previous year."
            ),
            "company_name": "Tech Mahindra Ltd.",
            "symbol": "TECHM",
            "fiscal_year": "2025-26",
        }
    ]


@pytest.fixture
def hcltech_assets_evidence():
    """Put misleading net-assets evidence before the correct balance sheet."""
    return [
        {
            "source_doc": "hcltech_annual_report_2025_26.pdf",
            "page_number": 448,
            "chunk_id": "hcltech_annual_report_2025_26.pdf:page-448:chunk-2",
            "symbol": "HCLTECH",
            "fiscal_year": "2025-26",
            "text": (
                "Total\n"
                "100.00 75,197 100.00 16,512 100.00 (1,862) "
                "100.00 14,650\n"
                "Non controlling interest\n"
                "(32) (10) (4) (14)\n"
                "Consolidation adjustments\n"
                "140 5,571 5,711\n"
                "Consolidated Net assets / Profit after tax\n"
                "75,165 16,642 3,705 20,347\n"
                "Note: Dividend received from subsidiaries has been "
                "excluded from profits."
            ),
        },
        {
            "source_doc": "hcltech_annual_report_2025_26.pdf",
            "page_number": 373,
            "chunk_id": "hcltech_annual_report_2025_26.pdf:page-373:chunk-1",
            "symbol": "HCLTECH",
            "fiscal_year": "2025-26",
            "text": (
                "370\n"
                "Consolidated Financial Statements\n"
                "Note\n"
                "No.\n"
                "As at\n"
                "31 March 2026\n"
                "31 March 2025\n"
                "I\n"
                "ASSETS\n"
                "(1)\n"
                "Non-current assets\n"
                "Total non-current assets\n"
                "45,716\n"
                "43,435\n"
                "(2)\n"
                "Current assets\n"
                "Total current assets\n"
                "70,542\n"
                "62,109\n"
                "TOTAL ASSETS\n"
                "116,258\n"
                "105,544\n"
                "II\n"
                "EQUITY\n"
                "Equity attributable to owners of the Company\n"
                "75,165\n"
                "69,655\n"
                "TOTAL EQUITY\n"
                "75,197\n"
                "69,673\n"
                "Consolidated Balance Sheet\n"
                "(All amounts in crores of ₹, except share data and as)"
            ),
        },
    ]


def test_empty_evidence_returns_abstention():
    result = handle_empty_evidence()

    assert result.abstained is True
    assert result.citations == []
    assert "does not provide sufficient information" in result.answer.lower()


def test_empty_evidence_returns_generated_answer():
    result = handle_empty_evidence()

    assert result.answer
    assert isinstance(result.citations, list)


def test_build_evidence_map_assigns_ids(sample_evidence):
    result = build_evidence_map(sample_evidence)

    assert list(result.keys()) == ["S1", "S2"]
    assert result["S1"]["chunk_id"] == "tcs:page-25:chunk-1"
    assert result["S2"]["chunk_id"] == "tcs:page-58:chunk-2"


def test_build_evidence_map_preserves_original_chunks(sample_evidence):
    result = build_evidence_map(sample_evidence)

    assert result["S1"] is sample_evidence[0]
    assert result["S2"] is sample_evidence[1]


def test_build_evidence_map_respects_top_k(sample_evidence):
    result = build_evidence_map(sample_evidence, top_k=1)

    assert list(result.keys()) == ["S1"]
    assert result["S1"]["chunk_id"] == "tcs:page-25:chunk-1"


def test_build_evidence_map_rejects_empty_evidence():
    with pytest.raises(ValueError, match="Evidence cannot be empty"):
        build_evidence_map([])


def test_build_evidence_map_rejects_invalid_top_k(sample_evidence):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        build_evidence_map(sample_evidence, top_k=0)


def test_build_evidence_map_rejects_missing_field(sample_evidence):
    invalid_evidence = [sample_evidence[0].copy()]
    del invalid_evidence[0]["chunk_id"]

    with pytest.raises(ValueError, match="missing 'chunk_id'"):
        build_evidence_map(invalid_evidence)


def test_build_evidence_map_rejects_blank_text(sample_evidence):
    invalid_evidence = [sample_evidence[0].copy()]
    invalid_evidence[0]["text"] = "   "

    with pytest.raises(ValueError, match="non-blank string"):
        build_evidence_map(invalid_evidence)


def test_build_evidence_map_rejects_invalid_page_number(sample_evidence):
    invalid_evidence = [sample_evidence[0].copy()]
    invalid_evidence[0]["page_number"] = 0

    with pytest.raises(ValueError, match="positive integer"):
        build_evidence_map(invalid_evidence)


def test_build_response_schema_restricts_evidence_ids():
    result = build_response_schema(["S1", "S2", "S3"])

    allowed_ids = result["$defs"]["EvidenceExcerpt"]["properties"][
        "evidence_id"
    ]["enum"]

    assert allowed_ids == ["S1", "S2", "S3"]


def test_build_response_schema_does_not_modify_original_schema():
    original_schema = LLMGenerationResponse.model_json_schema()

    build_response_schema(["S1", "S2"])

    assert (
        "enum"
        not in original_schema["$defs"]["EvidenceExcerpt"]["properties"][
            "evidence_id"
        ]
    )


@pytest.mark.parametrize(
    "evidence_ids",
    [
        [],
        ["S1", ""],
        ["S1", "   "],
        ["S1", "S1"],
    ],
)
def test_build_response_schema_rejects_invalid_ids(evidence_ids):
    with pytest.raises(ValueError):
        build_response_schema(evidence_ids)


def test_generation_error_is_runtime_error():
    error = GenerationError("Invalid LLM response.")

    assert isinstance(error, RuntimeError)
    assert str(error) == "Invalid LLM response."


def test_parse_llm_response_accepts_valid_json():
    raw_response = json.dumps(
        {
            "answer": "TCS reported an employee strength of 584,519.",
            "excerpts": [{"evidence_id": "S1"}],
            "abstained": False,
        }
    )

    result = parse_llm_response(raw_response)

    assert result.answer == (
        "TCS reported an employee strength of 584,519."
    )
    assert result.excerpts[0].evidence_id == "S1"
    assert result.abstained is False


def test_parse_llm_response_rejects_malformed_json():
    raw_response = '{"excerpts": ["Incomplete JSON"'

    with pytest.raises(
        GenerationError,
        match="invalid generation response",
    ):
        parse_llm_response(raw_response)


def test_parse_llm_response_rejects_missing_fields():
    raw_response = json.dumps({"excerpts": []})

    with pytest.raises(
        GenerationError,
        match="invalid generation response",
    ):
        parse_llm_response(raw_response)


def test_parse_llm_response_rejects_unexpected_fields():
    raw_response = json.dumps(
        {
            "answer": "A valid answer.",
            "excerpts": [],
            "abstained": False,
            "extra_field": "unexpected",
        }
    )

    with pytest.raises(
        GenerationError,
        match="invalid generation response",
    ):
        parse_llm_response(raw_response)


def test_validate_evidence_excerpts_accepts_valid_ids(sample_evidence):
    response = LLMGenerationResponse(
        answer="TCS reported 584,519 employees.",
        excerpts=[EvidenceExcerpt(evidence_id="S1")],
        abstained=False,
    )

    evidence_map = {"S1": sample_evidence[0]}

    validate_evidence_excerpts(response, evidence_map)


def test_validate_evidence_excerpts_accepts_multiple_valid_ids(
    sample_evidence,
):
    response = LLMGenerationResponse(
        answer="TCS reported its employee strength and revenue.",
        excerpts=[
            EvidenceExcerpt(evidence_id="S1"),
            EvidenceExcerpt(evidence_id="S2"),
        ],
        abstained=False,
    )

    evidence_map = {
        "S1": sample_evidence[0],
        "S2": sample_evidence[1],
    }

    validate_evidence_excerpts(response, evidence_map)


def test_validate_evidence_excerpts_rejects_unknown_evidence_id(
    sample_evidence,
):
    response = LLMGenerationResponse(
        answer="TCS reported 584,519 employees.",
        excerpts=[EvidenceExcerpt(evidence_id="S99")],
        abstained=False,
    )

    evidence_map = {"S1": sample_evidence[0]}

    with pytest.raises(
        GenerationError,
        match="unknown evidence ID",
    ):
        validate_evidence_excerpts(response, evidence_map)


@pytest.mark.asyncio
async def test_generate_answer_returns_valid_answer(sample_evidence):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": (
                "TCS reported a total employee strength of "
                "584,519 in the supplied annual report."
            ),
            "excerpts": [{"evidence_id": "S1"}],
            "abstained": False,
        }
    )

    result = await generate_answer(
        question="What was TCS's employee strength?",
        evidence=sample_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert result.answer == (
        "TCS reported a total employee strength of "
        "584,519 in the supplied annual report."
    )
    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.document_name == "tcs_annual_report_2025_26"
    assert citation.page_number == 25
    assert citation.chunk_id == "tcs:page-25:chunk-1"
    assert citation.evidence_text == "Total employee strength was 584,519."

    llm.generate.assert_awaited_once()

    generation_call = llm.generate.await_args

    assert generation_call.kwargs["system"] == SYSTEM_PROMPT

    response_schema = generation_call.kwargs["response_format"]
    allowed_ids = response_schema["$defs"]["EvidenceExcerpt"][
        "properties"
    ]["evidence_id"]["enum"]

    assert allowed_ids == ["S1", "S2"]


@pytest.mark.asyncio
async def test_generate_answer_uses_natural_language_not_just_quotes(
    sample_evidence,
):
    llm = AsyncMock()

    natural_answer = (
        "TCS reported a workforce of 584,519 employees. "
        "This figure is stated in the supplied annual report."
    )

    llm.generate.return_value = json.dumps(
        {
            "answer": natural_answer,
            "excerpts": [{"evidence_id": "S1"}],
            "abstained": False,
        }
    )

    result = await generate_answer(
        question="Summarize TCS employee strength.",
        evidence=sample_evidence,
        llm=llm,
    )

    assert result.answer == natural_answer
    assert result.answer != "Total employee strength was 584,519."
    assert len(result.citations) == 1


@pytest.mark.asyncio
async def test_generate_answer_skips_llm_when_evidence_is_empty():
    llm = AsyncMock()

    result = await generate_answer(
        question="What was TCS's revenue?",
        evidence=[],
        llm=llm,
    )

    assert result.abstained is True
    assert result.citations == []

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_answer_rejects_unknown_evidence_id(
    sample_evidence,
):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": "TCS reported an employee strength of 584,519.",
            "excerpts": [{"evidence_id": "S99"}],
            "abstained": False,
        }
    )

    with pytest.raises(
        GenerationError,
        match="unknown evidence ID",
    ):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )


@pytest.mark.asyncio
async def test_generate_answer_accepts_abstention(sample_evidence):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": (
                "The supplied annual report evidence does not "
                "provide sufficient information about monthly attrition."
            ),
            "excerpts": [],
            "abstained": True,
        }
    )

    result = await generate_answer(
        question="What was the monthly attrition rate?",
        evidence=sample_evidence,
        llm=llm,
    )

    assert result.abstained is True
    assert result.citations == []


@pytest.mark.asyncio
async def test_generate_answer_rejects_excerpts_in_abstention(
    sample_evidence,
):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": "The report does not provide sufficient information.",
            "excerpts": [{"evidence_id": "S1"}],
            "abstained": True,
        }
    )

    with pytest.raises(
        GenerationError,
        match="abstained response cannot contain evidence excerpts",
    ):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )


@pytest.mark.asyncio
async def test_generate_answer_rejects_missing_excerpts(
    sample_evidence,
):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": "TCS reported a strong workforce.",
            "excerpts": [],
            "abstained": False,
        }
    )

    with pytest.raises(
        GenerationError,
        match="non-abstained response must contain evidence excerpts",
    ):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )


@pytest.mark.asyncio
async def test_generate_answer_supports_multiple_citations(
    sample_evidence,
):
    llm = AsyncMock()

    natural_answer = (
        "TCS reported an employee strength of 584,519. "
        "Its revenue from operations was INR 267,021 crore."
    )

    llm.generate.return_value = json.dumps(
        {
            "answer": natural_answer,
            "excerpts": [
                {"evidence_id": "S1"},
                {"evidence_id": "S2"},
            ],
            "abstained": False,
        }
    )

    result = await generate_answer(
        question="What were TCS's employee strength and revenue?",
        evidence=sample_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert result.answer == natural_answer
    assert len(result.citations) == 2

    assert result.citations[0].page_number == 25
    assert result.citations[0].chunk_id == "tcs:page-25:chunk-1"

    assert result.citations[1].page_number == 58
    assert result.citations[1].chunk_id == "tcs:page-58:chunk-2"


@pytest.mark.asyncio
async def test_generate_answer_respects_top_k(sample_evidence):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": "TCS reported an employee strength of 584,519.",
            "excerpts": [{"evidence_id": "S1"}],
            "abstained": False,
        }
    )

    result = await generate_answer(
        question="What was TCS's employee strength?",
        evidence=sample_evidence,
        llm=llm,
        top_k=1,
    )

    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 25

    prompt_sent = llm.generate.await_args.args[0]

    assert "[S1]" in prompt_sent
    assert "[S2]" not in prompt_sent

    response_schema = llm.generate.await_args.kwargs["response_format"]
    allowed_ids = response_schema["$defs"]["EvidenceExcerpt"][
        "properties"
    ]["evidence_id"]["enum"]

    assert allowed_ids == ["S1"]


@pytest.mark.asyncio
async def test_generate_answer_rejects_evidence_outside_top_k(
    sample_evidence,
):
    llm = AsyncMock()

    llm.generate.return_value = json.dumps(
        {
            "answer": "TCS reported revenue of INR 267,021 crore.",
            "excerpts": [{"evidence_id": "S2"}],
            "abstained": False,
        }
    )

    with pytest.raises(
        GenerationError,
        match="unknown evidence ID",
    ):
        await generate_answer(
            question="What was TCS's revenue?",
            evidence=sample_evidence,
            llm=llm,
            top_k=1,
        )


@pytest.mark.asyncio
async def test_generate_answer_rejects_invalid_llm_json(
    sample_evidence,
):
    llm = AsyncMock()
    llm.generate.return_value = '{"excerpts": ["Incomplete JSON"'

    with pytest.raises(
        GenerationError,
        match="invalid generation response",
    ):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )

    llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_propagates_connection_error(
    sample_evidence,
):
    llm = AsyncMock()
    llm.generate.side_effect = OllamaConnectionError(
        "Could not connect to Ollama."
    )

    with pytest.raises(OllamaConnectionError):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )

    llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_propagates_timeout_error(
    sample_evidence,
):
    llm = AsyncMock()
    llm.generate.side_effect = OllamaTimeoutError(
        "Ollama request timed out."
    )

    with pytest.raises(OllamaTimeoutError):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
        )

    llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_rejects_invalid_top_k(sample_evidence):
    llm = AsyncMock()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await generate_answer(
            question="What was TCS's employee strength?",
            evidence=sample_evidence,
            llm=llm,
            top_k=0,
        )

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_techmahindra_fy2026_operating_margin_uses_exact_evidence(
    techmahindra_operating_margin_evidence,
):
    llm = AsyncMock()

    result = await generate_answer(
        question="What was Tech Mahindra operating margin in FY 2025-26?",
        evidence=techmahindra_operating_margin_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "12.6%" in result.answer
    assert "consolidated" in result.answer.lower()
    assert "operating margin" in result.answer.lower()

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.document_name == (
        "techmahindra_annual_report_2025_26.pdf"
    )
    assert citation.page_number == 127
    assert citation.chunk_id == (
        "techmahindra_annual_report_2025_26.pdf:"
        "page-127:chunk-2"
    )
    assert "stood at 12.6%" in citation.evidence_text

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_hcltech_total_assets_uses_balance_sheet_not_net_assets(
    hcltech_assets_evidence,
):
    """A misleading net-assets chunk before the correct row must not win."""

    llm = AsyncMock()

    result = await generate_answer(
        question="What were HCLTech total assets in FY 2025-26?",
        evidence=hcltech_assets_evidence,
        llm=llm,
    )

    assert result.abstained is False
    assert "₹1,16,258 crore" in result.answer
    assert "75,165" not in result.answer

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.page_number == 373
    assert citation.chunk_id == (
        "hcltech_annual_report_2025_26.pdf:"
        "page-373:chunk-1"
    )
    assert "TOTAL ASSETS" in citation.evidence_text
    assert "116,258" in citation.evidence_text

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_hcltech_net_assets_chunk_cannot_answer_total_assets(
    hcltech_assets_evidence,
):
    """If only net-assets evidence is supplied, FinSight must abstain."""

    llm = AsyncMock()

    result = await generate_answer(
        question="What were HCLTech total assets in FY 2025-26?",
        evidence=[hcltech_assets_evidence[0]],
        llm=llm,
    )

    assert result.abstained is True
    assert result.citations == []
    assert "sufficient information" in result.answer.lower()

    llm.generate.assert_not_awaited()

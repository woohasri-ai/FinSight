
import pytest
from pydantic import ValidationError

from finsight.generation.schemas import (
    Citation,
    EvidenceExcerpt,
    GeneratedAnswer,
    LLMGenerationResponse,
)


def test_citation_accepts_valid_data():
    citation = Citation(
        document_name="tcs_annual_report_2025_26",
        page_number=25,
        chunk_id="tcs:page-25:chunk-1",
        evidence_text="Total employee strength was 584,519.",
    )

    assert citation.page_number == 25
    assert citation.chunk_id == "tcs:page-25:chunk-1"


def test_citation_rejects_zero_page_number():
    with pytest.raises(ValidationError):
        Citation(
            document_name="tcs_report",
            page_number=0,
            chunk_id="chunk-1",
            evidence_text="Some evidence.",
        )


@pytest.mark.parametrize(
    "field",
    ["document_name", "chunk_id", "evidence_text"],
)
def test_citation_rejects_blank_text(field):
    data = {
        "document_name": "tcs_report",
        "page_number": 25,
        "chunk_id": "chunk-1",
        "evidence_text": "Some evidence.",
    }
    data[field] = "   "

    with pytest.raises(ValidationError):
        Citation(**data)


def test_citation_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        Citation(
            document_name="tcs_report",
            page_number=25,
            chunk_id="chunk-1",
            evidence_text="Some evidence.",
            unknown_field="unexpected",
        )


def test_generated_answer_accepts_valid_data():
    answer = GeneratedAnswer(
        answer="TCS reported 584,519 employees.",
        citations=[
            Citation(
                document_name="tcs_report",
                page_number=25,
                chunk_id="chunk-1",
                evidence_text="Total employee strength was 584,519.",
            )
        ],
        abstained=False,
    )

    assert answer.abstained is False
    assert len(answer.citations) == 1


def test_generated_answer_defaults_to_empty_citations():
    answer = GeneratedAnswer(
        answer="The report does not provide sufficient information.",
        abstained=True,
    )

    assert answer.citations == []


def test_generated_answer_rejects_empty_answer():
    with pytest.raises(ValidationError):
        GeneratedAnswer(
            answer="",
            abstained=True,
        )


def test_generated_answer_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        GeneratedAnswer(
            answer="Some answer.",
            abstained=False,
            unexpected="value",
        )


def test_evidence_excerpt_accepts_valid_data():
    excerpt = EvidenceExcerpt(evidence_id="S1")

    assert excerpt.evidence_id == "S1"


def test_evidence_excerpt_rejects_blank_evidence_id():
    with pytest.raises(ValidationError):
        EvidenceExcerpt(evidence_id="   ")


def test_evidence_excerpt_rejects_missing_evidence_id():
    with pytest.raises(ValidationError):
        EvidenceExcerpt()


def test_evidence_excerpt_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        EvidenceExcerpt(
            evidence_id="S1",
            quote="Some evidence.",
        )


def test_llm_response_accepts_valid_data():
    result = LLMGenerationResponse(
        answer=(
            "TCS reported 584,519 employees and revenue "
            "of INR 267,021 crore."
        ),
        excerpts=[
            EvidenceExcerpt(evidence_id="S1"),
            EvidenceExcerpt(evidence_id="S2"),
        ],
        abstained=False,
    )

    assert result.answer
    assert len(result.excerpts) == 2
    assert result.excerpts[0].evidence_id == "S1"
    assert result.excerpts[1].evidence_id == "S2"
    assert result.abstained is False


def test_llm_response_defaults_to_empty_excerpts():
    result = LLMGenerationResponse(
        answer="The report does not provide sufficient information.",
        abstained=True,
    )

    assert result.excerpts == []
    assert result.abstained is True


def test_llm_response_rejects_blank_answer():
    with pytest.raises(ValidationError):
        LLMGenerationResponse(
            answer="   ",
            excerpts=[],
            abstained=True,
        )


def test_llm_response_rejects_duplicate_evidence_references():
    with pytest.raises(
        ValidationError,
        match="Duplicate evidence references",
    ):
        LLMGenerationResponse(
            answer="TCS reported 584,519 employees.",
            excerpts=[
                EvidenceExcerpt(evidence_id="S1"),
                EvidenceExcerpt(evidence_id="S1"),
            ],
            abstained=False,
        )


def test_llm_response_allows_different_evidence_references():
    result = LLMGenerationResponse(
        answer="The report describes the company's workforce.",
        excerpts=[
            EvidenceExcerpt(evidence_id="S1"),
            EvidenceExcerpt(evidence_id="S2"),
        ],
        abstained=False,
    )

    assert len(result.excerpts) == 2


def test_llm_response_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        LLMGenerationResponse(
            answer="Some answer.",
            excerpts=[],
            abstained=True,
            unexpected="value",
        )


def test_llm_response_rejects_missing_abstained_field():
    with pytest.raises(ValidationError):
        LLMGenerationResponse(
            answer="Some answer.",
            excerpts=[],
        )


def test_llm_response_rejects_missing_answer():
    with pytest.raises(ValidationError):
        LLMGenerationResponse(
            excerpts=[],
            abstained=True,
        )

import pytest

from finsight.generation.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
)


@pytest.fixture
def sample_evidence():
    return [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "Total employee strength was 584,519.",
            "score": 0.95,
        },
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 58,
            "chunk_id": "tcs:page-58:chunk-2",
            "text": "Revenue from operations was ₹267,021 crore.",
            "score": 0.90,
        },
    ]


def test_system_prompt_contains_grounding_rules():
    prompt = SYSTEM_PROMPT.lower()

    assert "only the supplied evidence" in prompt
    assert "never invent" in prompt
    assert "abstain" in prompt
    assert "evidence" in prompt
    assert "evidence ids" in prompt


def test_system_prompt_requires_evidence_id_references():
    prompt = SYSTEM_PROMPT.lower()

    assert "evidence id" in prompt
    assert "support the answer" in prompt
    assert "do not invent" in prompt
    assert "quotation text" in prompt


def test_system_prompt_requires_natural_language_answers():
    prompt = SYSTEM_PROMPT.lower()

    assert "natural language" in prompt
    assert "answer the actual question directly" in prompt
    assert "clear, professional" in prompt


def test_system_prompt_requires_new_response_structure():
    assert '"answer"' in SYSTEM_PROMPT
    assert '"excerpts"' in SYSTEM_PROMPT
    assert '"evidence_id"' in SYSTEM_PROMPT
    assert '"quote"' not in SYSTEM_PROMPT
    assert '"abstained"' in SYSTEM_PROMPT
    assert "do not include quotation text or additional fields" in (
        SYSTEM_PROMPT.lower()
    )


def test_system_prompt_supports_abstention():
    prompt = SYSTEM_PROMPT.lower()

    assert "empty excerpts array" in prompt
    assert "abstained=true" in prompt


def test_prompt_includes_user_question(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "What was TCS's employee strength?" in prompt


def test_prompt_assigns_evidence_ids(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert '"evidence_id": "S1"' in prompt
    assert '"evidence_id": "S2"' in prompt


def test_prompt_includes_source_metadata(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "tcs_annual_report_2025_26" in prompt
    assert "tcs:page-25:chunk-1" in prompt
    assert "Page: 25" in prompt
    assert '"page_number": 25' in prompt


def test_prompt_includes_evidence_text(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "Total employee strength was 584,519." in prompt
    assert "Revenue from operations was ₹267,021 crore." in prompt


def test_prompt_requests_natural_language_answer(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "Answer the financial research question" in prompt
    assert "clear, professional, natural language" in SYSTEM_PROMPT
    assert "Answer" in prompt


def test_prompt_requests_evidence_id_references(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "appropriate evidence IDs" in prompt
    assert "only IDs that appear" in prompt
    assert "evidence_id" in prompt


def test_prompt_requests_new_response_structure(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
    )

    assert "answer" in prompt.lower()
    assert "evidence_id" in prompt
    assert "quotation" not in prompt.lower()
    assert "JSON output format" in prompt


def test_prompt_respects_top_k(sample_evidence):
    prompt = build_generation_prompt(
        "What was TCS's employee strength?",
        sample_evidence,
        top_k=1,
    )

    assert "[S1]" in prompt
    assert "[S2]" not in prompt
    assert '"evidence_id": "S1"' in prompt
    assert '"evidence_id": "S2"' not in prompt
    assert "Total employee strength was 584,519." in prompt
    assert "Revenue from operations" not in prompt


def test_prompt_rejects_blank_question(sample_evidence):
    with pytest.raises(ValueError, match="Question cannot be blank"):
        build_generation_prompt("   ", sample_evidence)


def test_prompt_rejects_invalid_top_k(sample_evidence):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        build_generation_prompt(
            "What was TCS's employee strength?",
            sample_evidence,
            top_k=0,
        )


def test_prompt_rejects_empty_evidence():
    with pytest.raises(ValueError, match="Evidence cannot be empty"):
        build_generation_prompt(
            "What was TCS's employee strength?",
            [],
        )


def test_prompt_rejects_missing_evidence_field(sample_evidence):
    invalid_evidence = [sample_evidence[0].copy()]
    del invalid_evidence[0]["chunk_id"]

    with pytest.raises(ValueError, match="missing 'chunk_id'"):
        build_generation_prompt(
            "What was TCS's employee strength?",
            invalid_evidence,
        )

import re
from typing import Any


def _extract_fact_value(fact: str) -> str:
    """Extract the value portion after the first colon."""
    if ":" not in fact:
        return fact.strip()

    return fact.split(":", maxsplit=1)[1].strip()


def _normalize_text(text: str) -> str:
    """Normalize text for case-insensitive value matching."""
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_fact_coverage(
    answer: str,
    required_facts: list[str],
) -> dict[str, Any]:
    """Evaluate how many required fact values appear in an answer."""

    if not answer.strip():
        return {
            "matched_facts": [],
            "missing_facts": required_facts.copy(),
            "coverage": 0.0 if required_facts else 1.0,
        }

    normalized_answer = _normalize_text(answer)

    matched_facts = []
    missing_facts = []

    for fact in required_facts:
        expected_value = _normalize_text(_extract_fact_value(fact))

        if expected_value and expected_value in normalized_answer:
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    coverage = len(matched_facts) / len(required_facts) if required_facts else 1.0

    return {
        "matched_facts": matched_facts,
        "missing_facts": missing_facts,
        "coverage": coverage,
    }


def evaluate_citations(
    expected_pages: list[int],
    generated_pages: list[int],
) -> dict[str, Any]:
    """Evaluate citation precision and recall using page numbers."""

    expected = set(expected_pages)
    generated = set(generated_pages)

    correct_pages = sorted(expected & generated)
    missing_pages = sorted(expected - generated)
    unexpected_pages = sorted(generated - expected)

    precision = len(correct_pages) / len(generated) if generated else 0.0

    recall = len(correct_pages) / len(expected) if expected else 1.0

    return {
        "correct_pages": correct_pages,
        "missing_pages": missing_pages,
        "unexpected_pages": unexpected_pages,
        "citation_precision": precision,
        "citation_recall": recall,
    }


def evaluate_abstention(
    expected_answerable: bool,
    actually_answered: bool,
) -> dict[str, Any]:
    """Evaluate whether FinSight answered or abstained appropriately."""

    correctly_handled = expected_answerable == actually_answered

    if expected_answerable and actually_answered:
        result = "correctly_answered"
    elif not expected_answerable and not actually_answered:
        result = "correctly_abstained"
    elif not expected_answerable and actually_answered:
        result = "incorrectly_answered"
    else:
        result = "incorrectly_abstained"

    return {
        "correct": correctly_handled,
        "result": result,
    }


def evaluate_answer_quality(
    answer: str,
    required_facts: list[str],
    expected_pages: list[int],
    generated_pages: list[int],
    expected_answerable: bool,
    actually_answered: bool,
) -> dict[str, Any]:
    """Combine fact, citation, and abstention evaluation results."""

    fact_results = evaluate_fact_coverage(
        answer=answer,
        required_facts=required_facts,
    )

    citation_results = evaluate_citations(
        expected_pages=expected_pages,
        generated_pages=generated_pages,
    )

    abstention_results = evaluate_abstention(
        expected_answerable=expected_answerable,
        actually_answered=actually_answered,
    )

    return {
        "fact_coverage": fact_results,
        "citations": citation_results,
        "abstention": abstention_results,
    }

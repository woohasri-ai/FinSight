import pytest

from finsight.evaluation.deterministic import (
    evaluate_abstention,
    evaluate_answer_quality,
    evaluate_citations,
    evaluate_fact_coverage,
)


def test_all_required_facts_are_present():
    answer = "TCS reported revenue from operations of ₹267,021 crore and EBIT of ₹66,838 crore."

    required_facts = [
        "Revenue from operations: ₹267,021 crore",
        "EBIT: ₹66,838 crore",
    ]

    result = evaluate_fact_coverage(answer, required_facts)

    assert result["coverage"] == 1.0
    assert len(result["matched_facts"]) == 2
    assert result["missing_facts"] == []


def test_some_required_facts_are_missing():
    answer = "TCS reported revenue from operations of ₹267,021 crore."

    required_facts = [
        "Revenue from operations: ₹267,021 crore",
        "EBIT: ₹66,838 crore",
    ]

    result = evaluate_fact_coverage(answer, required_facts)

    assert result["coverage"] == 0.5
    assert len(result["matched_facts"]) == 1
    assert len(result["missing_facts"]) == 1


def test_empty_required_facts():
    result = evaluate_fact_coverage("Some answer", [])

    assert result["coverage"] == 1.0
    assert result["matched_facts"] == []
    assert result["missing_facts"] == []


def test_empty_answer():
    required_facts = ["Revenue: ₹267,021 crore"]

    result = evaluate_fact_coverage("", required_facts)

    assert result["coverage"] == 0.0
    assert result["missing_facts"] == required_facts


def test_whitespace_only_answer():
    result = evaluate_fact_coverage("   \n\t ", ["Revenue: ₹267,021 crore"])

    assert result["coverage"] == 0.0
    assert result["missing_facts"] == ["Revenue: ₹267,021 crore"]


def test_fact_matching_is_case_insensitive():
    answer = "TCS REPORTED REVENUE OF ₹267,021 CRORE."

    result = evaluate_fact_coverage(
        answer,
        ["Revenue: ₹267,021 crore"],
    )

    assert result["coverage"] == 1.0


def test_fact_matching_normalizes_whitespace():
    answer = "TCS reported revenue of ₹267,021\n\ncrore."

    result = evaluate_fact_coverage(
        answer,
        ["Revenue: ₹267,021 crore"],
    )

    assert result["coverage"] == 1.0


def test_fact_without_colon_is_matched_as_written():
    answer = "TCS reported revenue of ₹267,021 crore."

    result = evaluate_fact_coverage(
        answer,
        ["TCS reported revenue of ₹267,021 crore."],
    )

    assert result["coverage"] == 1.0


def test_empty_fact_value_is_not_counted_as_matched():
    result = evaluate_fact_coverage(
        "TCS reported revenue.",
        ["Revenue:"],
    )

    assert result["coverage"] == 0.0
    assert result["missing_facts"] == ["Revenue:"]


def test_all_citations_are_correct():
    result = evaluate_citations([58, 59], [58, 59])

    assert result["correct_pages"] == [58, 59]
    assert result["missing_pages"] == []
    assert result["unexpected_pages"] == []
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0


def test_missing_and_unexpected_citations():
    result = evaluate_citations([58, 59], [58, 60])

    assert result["correct_pages"] == [58]
    assert result["missing_pages"] == [59]
    assert result["unexpected_pages"] == [60]
    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 0.5


def test_no_generated_citations():
    result = evaluate_citations([58], [])

    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 0.0


def test_no_expected_citations():
    result = evaluate_citations([], [58])

    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 1.0


def test_both_expected_and_generated_citations_empty():
    result = evaluate_citations([], [])

    assert result["correct_pages"] == []
    assert result["missing_pages"] == []
    assert result["unexpected_pages"] == []
    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 1.0


def test_duplicate_citation_pages_are_counted_once():
    result = evaluate_citations([58, 58, 59], [58, 58, 59, 59])

    assert result["correct_pages"] == [58, 59]
    assert result["missing_pages"] == []
    assert result["unexpected_pages"] == []
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0


@pytest.mark.parametrize(
    ("expected_answerable", "actually_answered", "expected_result"),
    [
        (True, True, "correctly_answered"),
        (False, False, "correctly_abstained"),
        (False, True, "incorrectly_answered"),
        (True, False, "incorrectly_abstained"),
    ],
)
def test_abstention_outcomes(
    expected_answerable,
    actually_answered,
    expected_result,
):
    result = evaluate_abstention(
        expected_answerable=expected_answerable,
        actually_answered=actually_answered,
    )

    assert result["result"] == expected_result
    assert result["correct"] is (expected_answerable == actually_answered)


def test_combined_answer_quality_evaluation():
    answer = "TCS reported revenue from operations of ₹267,021 crore and EBIT of ₹66,838 crore."

    result = evaluate_answer_quality(
        answer=answer,
        required_facts=[
            "Revenue from operations: ₹267,021 crore",
            "EBIT: ₹66,838 crore",
        ],
        expected_pages=[58],
        generated_pages=[58],
        expected_answerable=True,
        actually_answered=True,
    )

    assert result["fact_coverage"]["coverage"] == 1.0
    assert result["citations"]["citation_precision"] == 1.0
    assert result["citations"]["citation_recall"] == 1.0
    assert result["abstention"]["correct"] is True


def test_combined_evaluation_for_correct_abstention():
    result = evaluate_answer_quality(
        answer="",
        required_facts=[],
        expected_pages=[],
        generated_pages=[],
        expected_answerable=False,
        actually_answered=False,
    )

    assert result["fact_coverage"]["coverage"] == 1.0
    assert result["citations"]["citation_precision"] == 0.0
    assert result["citations"]["citation_recall"] == 1.0
    assert result["abstention"]["correct"] is True
    assert result["abstention"]["result"] == "correctly_abstained"

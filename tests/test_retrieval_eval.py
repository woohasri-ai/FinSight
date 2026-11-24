import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
SPEC = importlib.util.spec_from_file_location(
    "run_retrieval_eval",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

recall_at_k = MODULE.recall_at_k
chunk_recall_at_k = MODULE.chunk_recall_at_k
reciprocal_rank = MODULE.reciprocal_rank
ndcg_at_k = MODULE.ndcg_at_k


# Page-level Recall@K


def test_recall_at_k_returns_one_when_all_expected_pages_are_found():
    results = [
        {"page_number": 14},
        {"page_number": 62},
        {"page_number": 27},
    ]

    assert recall_at_k(results, [14, 62], 3) == 1.0


def test_recall_at_k_returns_fraction_when_one_expected_page_is_found():
    results = [
        {"page_number": 14},
        {"page_number": 27},
    ]

    assert recall_at_k(results, [14, 62], 2) == 0.5


def test_recall_at_k_returns_zero_when_no_expected_pages_are_found():
    results = [
        {"page_number": 14},
        {"page_number": 27},
    ]

    assert recall_at_k(results, [62], 2) == 0.0


def test_recall_at_k_only_checks_top_k_results():
    results = [
        {"page_number": 14},
        {"page_number": 27},
        {"page_number": 62},
    ]

    assert recall_at_k(results, [62], 2) == 0.0


def test_recall_at_k_does_not_double_count_duplicate_expected_pages():
    results = [
        {"page_number": 14},
        {"page_number": 62},
    ]

    assert recall_at_k(results, [14, 14, 62], 2) == 1.0


def test_recall_at_k_returns_zero_for_empty_results():
    assert recall_at_k([], [14], 5) == 0.0


def test_recall_at_k_returns_zero_for_empty_expected_pages():
    results = [{"page_number": 14}]

    assert recall_at_k(results, [], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_invalid_cutoff(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k([], [14], k)


# Chunk-level Recall@K


def test_chunk_recall_at_k_returns_one_when_all_relevant_chunks_are_found():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert chunk_recall_at_k(results, ["a", "b"], 3) == 1.0


def test_chunk_recall_at_k_returns_fraction_for_partial_retrieval():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "c"},
    ]

    assert chunk_recall_at_k(results, ["a", "b"], 2) == 0.5


def test_chunk_recall_at_k_returns_zero_when_no_relevant_chunks_are_found():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert chunk_recall_at_k(results, ["c"], 2) == 0.0


def test_chunk_recall_at_k_only_checks_top_k_results():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert chunk_recall_at_k(results, ["c"], 2) == 0.0


def test_chunk_recall_at_k_does_not_double_count_duplicate_chunks():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert chunk_recall_at_k(results, ["a", "b"], 3) == 1.0


def test_chunk_recall_at_k_ignores_results_without_chunk_id():
    results = [
        {"page_number": 14},
        {"chunk_id": "a"},
    ]

    assert chunk_recall_at_k(results, ["a", "b"], 2) == 0.5


def test_chunk_recall_at_k_returns_zero_for_empty_results():
    assert chunk_recall_at_k([], ["a"], 5) == 0.0


def test_chunk_recall_at_k_returns_zero_for_empty_relevant_ids():
    results = [{"chunk_id": "a"}]

    assert chunk_recall_at_k(results, [], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_chunk_recall_at_k_rejects_invalid_cutoff(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        chunk_recall_at_k([], ["a"], k)


# Mean Reciprocal Rank


def test_reciprocal_rank_returns_one_for_first_result():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert reciprocal_rank(results, {"a"}) == 1.0


def test_reciprocal_rank_rewards_higher_relevant_rank():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert reciprocal_rank(results, {"b"}) == 0.5


def test_reciprocal_rank_returns_zero_when_no_relevant_chunk_is_found():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert reciprocal_rank(results, {"c"}) == 0.0


def test_reciprocal_rank_respects_cutoff():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert reciprocal_rank(results, {"c"}, k=2) == 0.0


def test_reciprocal_rank_returns_zero_for_empty_results():
    assert reciprocal_rank([], {"a"}) == 0.0


def test_reciprocal_rank_returns_zero_for_empty_relevant_ids():
    results = [{"chunk_id": "a"}]

    assert reciprocal_rank(results, set()) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_reciprocal_rank_rejects_invalid_cutoff(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank([], {"a"}, k=k)


# Normalized Discounted Cumulative Gain


def test_ndcg_returns_one_for_ideal_ranking():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert ndcg_at_k(results, {"a", "b"}) == 1.0


def test_ndcg_rewards_relevant_chunks_ranked_higher():
    good_ranking = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]
    poor_ranking = [
        {"chunk_id": "c"},
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    good_score = ndcg_at_k(good_ranking, {"a", "b"})
    poor_score = ndcg_at_k(poor_ranking, {"a", "b"})

    assert good_score > poor_score


def test_ndcg_returns_zero_when_no_relevant_chunk_is_found():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert ndcg_at_k(results, {"c"}) == 0.0


def test_ndcg_respects_cutoff():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert ndcg_at_k(results, {"c"}, k=2) == 0.0


def test_ndcg_does_not_count_duplicate_relevant_chunks_twice():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    score = ndcg_at_k(results, {"a", "b"})

    assert 0.0 < score < 1.0


def test_ndcg_returns_zero_for_empty_results():
    assert ndcg_at_k([], {"a"}) == 0.0


def test_ndcg_returns_zero_for_empty_relevant_ids():
    results = [{"chunk_id": "a"}]

    assert ndcg_at_k(results, set()) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_rejects_invalid_cutoff(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        ndcg_at_k([], {"a"}, k=k)

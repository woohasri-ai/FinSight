import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
SPEC = importlib.util.spec_from_file_location("run_retrieval_eval", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

recall_at_k = MODULE.recall_at_k
reciprocal_rank = MODULE.reciprocal_rank
ndcg_at_k = MODULE.ndcg_at_k


def test_recall_at_k_returns_one_when_all_expected_pages_are_found() -> None:
    results = [{"page_number": 14}, {"page_number": 62}, {"page_number": 27}]
    assert recall_at_k(results, [14, 62], 3) == 1.0


def test_recall_at_k_returns_fraction_when_one_expected_page_is_found() -> None:
    results = [{"page_number": 14}, {"page_number": 27}]
    assert recall_at_k(results, [14, 62], 2) == 0.5


def test_recall_at_k_returns_zero_when_no_expected_pages_are_found() -> None:
    results = [{"page_number": 14}, {"page_number": 27}]
    assert recall_at_k(results, [62], 2) == 0.0


def test_recall_at_k_only_checks_the_top_k_results() -> None:
    results = [{"page_number": 14}, {"page_number": 27}, {"page_number": 62}]
    assert recall_at_k(results, [62], 2) == 0.0


def test_reciprocal_rank_returns_one_for_first_result() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert reciprocal_rank(results, {"a"}) == 1.0


def test_reciprocal_rank_rewards_higher_relevant_rank() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert reciprocal_rank(results, {"b"}) == 0.5


def test_reciprocal_rank_returns_zero_when_no_relevant_chunk_is_found() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert reciprocal_rank(results, {"c"}) == 0.0


def test_reciprocal_rank_respects_cutoff() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert reciprocal_rank(results, {"c"}, k=2) == 0.0


def test_ndcg_returns_one_for_ideal_ranking() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert ndcg_at_k(results, {"a", "b"}) == 1.0


def test_ndcg_rewards_relevant_chunks_ranked_higher() -> None:
    good_ranking = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    poor_ranking = [{"chunk_id": "c"}, {"chunk_id": "a"}, {"chunk_id": "b"}]

    good_score = ndcg_at_k(good_ranking, {"a", "b"})
    poor_score = ndcg_at_k(poor_ranking, {"a", "b"})

    assert good_score > poor_score


def test_ndcg_returns_zero_when_no_relevant_chunk_is_found() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert ndcg_at_k(results, {"c"}) == 0.0


def test_ndcg_respects_cutoff() -> None:
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert ndcg_at_k(results, {"c"}, k=2) == 0.0

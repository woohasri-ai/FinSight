import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
SPEC = importlib.util.spec_from_file_location("run_retrieval_eval", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
recall_at_k = MODULE.recall_at_k


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

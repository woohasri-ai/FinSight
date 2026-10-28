import numpy as np
import pytest

from finsight.retrieval.rerank import rerank_chunks


class FakeReranker:
    def predict(
        self,
        sentences: list[tuple[str, str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return deterministic relevance scores for testing."""
        score_map = {
            "TCS revenue increased": 0.9,
            "AI strategy": 0.3,
            "Unrelated passage": 0.1,
        }
        return np.array(
            [score_map.get(text, 0.5) for _, text in sentences],
            dtype=np.float32,
        )


@pytest.fixture
def sample_chunks() -> list[dict[str, int | float | str]]:
    return [
        {
            "chunk_id": "A",
            "text": "AI strategy",
            "source_doc": "report.pdf",
            "page_number": 20,
            "score": 0.016,
        },
        {
            "chunk_id": "B",
            "text": "TCS revenue increased",
            "source_doc": "report.pdf",
            "page_number": 10,
            "score": 0.015,
        },
    ]


def test_rerank_chunks_orders_by_relevance(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    results = rerank_chunks(
        "What is relevant?",
        sample_chunks,
        FakeReranker(),  # type: ignore[arg-type]
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "B"
    assert results[0]["page_number"] == 10
    assert results[0]["score"] > results[1]["score"]


def test_rerank_chunks_rejects_blank_query(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        rerank_chunks(
            "   ",
            sample_chunks,
            FakeReranker(),  # type: ignore[arg-type]
        )


def test_rerank_chunks_rejects_invalid_top_k(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        rerank_chunks(
            "TCS revenue",
            sample_chunks,
            FakeReranker(),  # type: ignore[arg-type]
            top_k=0,
        )


def test_rerank_chunks_returns_empty_for_no_chunks() -> None:
    results = rerank_chunks(
        "TCS revenue",
        [],
        FakeReranker(),  # type: ignore[arg-type]
    )

    assert results == []


def test_rerank_chunks_limits_results_and_preserves_metadata(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    results = rerank_chunks(
        "What is relevant?",
        sample_chunks,
        FakeReranker(),  # type: ignore[arg-type]
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "B"
    assert results[0]["source_doc"] == "report.pdf"
    assert results[0]["page_number"] == 10
    assert results[0]["text"] == "TCS revenue increased"

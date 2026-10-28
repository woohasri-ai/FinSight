import numpy as np
import pytest

from finsight.retrieval.hybrid import hybrid_search


class FakeModel:
    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return deterministic vectors for testing."""
        vectors = {
            "revenue growth": [1.0, 0.0],
            "TCS revenue": [0.9, 0.1],
            "TCS revenue increased": [0.8, 0.2],
            "AI strategy": [0.0, 1.0],
        }
        return np.array(
            [vectors.get(text, [0.5, 0.5]) for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def sample_chunks() -> list[dict[str, int | str]]:
    return [
        {
            "chunk_id": "A",
            "text": "TCS revenue increased",
            "source_doc": "report.pdf",
            "page_number": 10,
        },
        {
            "chunk_id": "B",
            "text": "AI strategy",
            "source_doc": "report.pdf",
            "page_number": 20,
        },
    ]


def test_hybrid_search_returns_ranked_results(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    model = FakeModel()
    index = build_bm25_index(sample_chunks)
    embeddings = np.array([[0.8, 0.2], [0.0, 1.0]], dtype=np.float32)

    results = hybrid_search(
        "TCS revenue",
        sample_chunks,
        model,  # type: ignore[arg-type]
        index,
        embeddings,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "A"
    assert results[0]["page_number"] == 10
    assert isinstance(results[0]["score"], float)


def test_hybrid_search_rejects_blank_query(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    with pytest.raises(ValueError, match="query must not be blank"):
        hybrid_search(
            "   ",
            sample_chunks,
            FakeModel(),  # type: ignore[arg-type]
            build_bm25_index(sample_chunks),
            np.array([[0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        )


def test_hybrid_search_rejects_invalid_top_k(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    with pytest.raises(ValueError, match="greater than zero"):
        hybrid_search(
            "TCS revenue",
            sample_chunks,
            FakeModel(),  # type: ignore[arg-type]
            build_bm25_index(sample_chunks),
            np.array([[0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
            top_k=0,
        )


def test_hybrid_search_returns_empty_for_no_chunks() -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    chunks = [
        {
            "chunk_id": "dummy",
            "text": "dummy text",
            "source_doc": "report.pdf",
            "page_number": 1,
        }
    ]

    results = hybrid_search(
        "anything",
        [],
        FakeModel(),  # type: ignore[arg-type]
        build_bm25_index(chunks),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert results == []


def test_hybrid_search_rejects_invalid_candidate_k(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    with pytest.raises(ValueError, match="greater than zero"):
        hybrid_search(
            "TCS revenue",
            sample_chunks,
            FakeModel(),  # type: ignore[arg-type]
            build_bm25_index(sample_chunks),
            np.array([[0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
            candidate_k=0,
        )


def test_hybrid_search_rejects_mismatched_embedding_dimensions(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    from finsight.retrieval.bm25 import build_bm25_index

    with pytest.raises(ValueError, match="dimensions must match"):
        hybrid_search(
            "TCS revenue",
            sample_chunks,
            FakeModel(),  # type: ignore[arg-type]
            build_bm25_index(sample_chunks),
            np.array([[0.8, 0.2, 0.1], [0.0, 1.0, 0.5]], dtype=np.float32),
        )

from pathlib import Path

import numpy as np
import pytest

from finsight.retrieval.dense import embed_texts, load_saved_embeddings, search_chunks


class FakeModel:
    """Provide deterministic embeddings without downloading a model."""

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        vectors = {
            "revenue growth": [1.0, 0.0],
            "employee wellbeing": [0.0, 1.0],
            "What drove revenue growth?": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts], dtype=np.float32)


def sample_chunks():
    return [
        {
            "chunk_id": "report:page-1:chunk-1",
            "text": "revenue growth",
            "source_doc": "report.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "report:page-2:chunk-1",
            "text": "employee wellbeing",
            "source_doc": "report.pdf",
            "page_number": 2,
        },
    ]


def test_embed_texts_returns_numpy_array():
    embeddings = embed_texts(["revenue growth"], FakeModel())

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 2)


def test_embed_texts_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        embed_texts([], FakeModel())


def test_embed_texts_rejects_blank_text():
    with pytest.raises(ValueError, match="blank"):
        embed_texts(["   "], FakeModel())


def test_search_chunks_returns_most_relevant_first():
    results = search_chunks(
        "What drove revenue growth?",
        sample_chunks(),
        FakeModel(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "report:page-1:chunk-1"
    assert results[0]["page_number"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_search_chunks_rejects_blank_query():
    with pytest.raises(ValueError, match="blank"):
        search_chunks("   ", sample_chunks(), FakeModel())


def test_search_chunks_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        search_chunks("revenue", sample_chunks(), FakeModel(), top_k=0)


def test_search_chunks_returns_empty_for_no_chunks():
    assert search_chunks("revenue", [], FakeModel()) == []


def test_load_saved_embeddings_returns_numpy_array(tmp_path: Path) -> None:
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    result = load_saved_embeddings(embeddings_path)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, embeddings)


def test_load_saved_embeddings_raises_for_missing_file(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
        load_saved_embeddings(embeddings_path)


def test_load_saved_embeddings_rejects_empty_array(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "empty.npy"
    np.save(embeddings_path, np.empty((0, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="must not be empty"):
        load_saved_embeddings(embeddings_path)


def test_load_saved_embeddings_rejects_one_dimensional_array(
    tmp_path: Path,
) -> None:
    embeddings_path = tmp_path / "one_dimensional.npy"
    np.save(embeddings_path, np.array([0.1, 0.2], dtype=np.float32))

    with pytest.raises(ValueError, match="two-dimensional"):
        load_saved_embeddings(embeddings_path)

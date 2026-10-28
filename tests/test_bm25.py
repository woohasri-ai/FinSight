import pytest
from rank_bm25 import BM25Okapi

from finsight.retrieval.bm25 import (
    build_bm25_index,
    search_bm25,
    tokenize_text,
)


@pytest.fixture
def sample_chunks() -> list[dict[str, int | str]]:
    return [
        {
            "chunk_id": "chunk-1",
            "text": "TCS revenue increased significantly",
            "source_doc": "tcs_report.pdf",
            "page_number": 10,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Infosys operating profit and margins",
            "source_doc": "infosys_report.pdf",
            "page_number": 20,
        },
        {
            "chunk_id": "chunk-3",
            "text": "TCS artificial intelligence strategy",
            "source_doc": "tcs_report.pdf",
            "page_number": 30,
        },
    ]


def test_tokenize_text_lowercases_words() -> None:
    assert tokenize_text("TCS Revenue Increased") == [
        "tcs",
        "revenue",
        "increased",
    ]


def test_build_bm25_index_returns_index(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    index = build_bm25_index(sample_chunks)

    assert isinstance(index, BM25Okapi)


def test_build_bm25_index_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        build_bm25_index([])


def test_build_bm25_index_rejects_blank_chunk() -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "   ",
            "source_doc": "report.pdf",
            "page_number": 1,
        }
    ]

    with pytest.raises(ValueError, match="blank text"):
        build_bm25_index(chunks)


def test_search_bm25_returns_relevant_chunk_first(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    index = build_bm25_index(sample_chunks)

    results = search_bm25("TCS revenue", sample_chunks, index)

    assert results
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["page_number"] == 10


def test_search_bm25_rejects_blank_query(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    index = build_bm25_index(sample_chunks)

    with pytest.raises(ValueError, match="query must not be blank"):
        search_bm25("   ", sample_chunks, index)


def test_search_bm25_rejects_invalid_top_k(
    sample_chunks: list[dict[str, int | str]],
) -> None:
    index = build_bm25_index(sample_chunks)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        search_bm25("TCS revenue", sample_chunks, index, top_k=0)


def test_search_bm25_returns_empty_for_no_chunks() -> None:
    index = build_bm25_index(
        [{"chunk_id": "x", "text": "sample text", "source_doc": "x.pdf", "page_number": 1}]
    )

    assert search_bm25("anything", [], index) == []

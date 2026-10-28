import pytest

from finsight.retrieval.fusion import reciprocal_rank_fusion


def make_result(
    chunk_id: str,
    score: float = 0.5,
) -> dict[str, int | float | str]:
    return {
        "chunk_id": chunk_id,
        "text": f"Text for {chunk_id}",
        "source_doc": "report.pdf",
        "page_number": 1,
        "score": score,
    }


def test_fusion_combines_ranked_lists() -> None:
    dense = [make_result("A"), make_result("B")]
    bm25 = [make_result("B"), make_result("C")]

    results = reciprocal_rank_fusion(dense, bm25)

    assert results[0]["chunk_id"] == "B"
    assert {result["chunk_id"] for result in results} == {"A", "B", "C"}


def test_fusion_ranks_shared_chunk_higher() -> None:
    dense = [make_result("A"), make_result("B")]
    bm25 = [make_result("B"), make_result("A")]

    results = reciprocal_rank_fusion(dense, bm25)

    assert results[0]["chunk_id"] == "A"
    assert results[1]["chunk_id"] == "B"
    assert results[0]["score"] == pytest.approx(results[1]["score"])


def test_fusion_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_fusion_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        reciprocal_rank_fusion([], k=0)


def test_fusion_does_not_duplicate_chunks() -> None:
    dense = [make_result("A"), make_result("A")]
    results = reciprocal_rank_fusion(dense)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "A"

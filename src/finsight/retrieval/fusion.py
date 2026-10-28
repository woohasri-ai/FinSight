from collections.abc import Sequence


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[dict[str, int | float | str]],
    k: int = 60,
) -> list[dict[str, int | float | str]]:
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, int | float | str]] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = str(result["chunk_id"])

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            chunks[chunk_id] = result

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)

    fused_results: list[dict[str, int | float | str]] = []

    for chunk_id in ranked_ids:
        result = dict(chunks[chunk_id])
        result["score"] = scores[chunk_id]
        fused_results.append(result)

    return fused_results

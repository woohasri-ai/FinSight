from collections.abc import Sequence

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
) -> CrossEncoder:
    """Load the cross-encoder reranking model on the CPU."""
    return CrossEncoder(model_name, device="cpu")


def rerank_chunks(
    query: str,
    chunks: Sequence[dict[str, int | float | str]],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict[str, int | float | str]]:
    """Rerank retrieved chunks by their relevance to the query."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    pairs = [(query, str(chunk["text"])) for chunk in chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    if len(scores) != len(chunks):
        raise ValueError("Reranker must return one score per chunk")

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:top_k]

    results: list[dict[str, int | float | str]] = []

    for index in ranked_indices:
        result = dict(chunks[index])
        result["score"] = float(scores[index])
        results.append(result)

    return results

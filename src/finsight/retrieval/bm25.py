from collections.abc import Sequence

from rank_bm25 import BM25Okapi


def tokenize_text(text: str) -> list[str]:
    """Normalize text into lowercase word tokens."""
    return text.lower().split()


def build_bm25_index(
    chunks: Sequence[dict[str, int | str]],
) -> BM25Okapi:
    """Build a BM25 index from report chunks."""
    if not chunks:
        raise ValueError("chunks must contain at least one item")

    tokenized_chunks = [tokenize_text(str(chunk["text"])) for chunk in chunks]

    if any(not tokens for tokens in tokenized_chunks):
        raise ValueError("chunks must not contain blank text")

    return BM25Okapi(tokenized_chunks)


def search_bm25(
    query: str,
    chunks: Sequence[dict[str, int | str]],
    index: BM25Okapi,
    top_k: int = 5,
) -> list[dict[str, int | float | str]]:
    """Retrieve report chunks using BM25 keyword relevance."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    scores = index.get_scores(tokenize_text(query))
    ranked_indices = scores.argsort()[::-1][:top_k]

    results: list[dict[str, int | float | str]] = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        chunk = chunks[int(idx)]
        results.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"]),
                "source_doc": str(chunk["source_doc"]),
                "page_number": int(chunk["page_number"]),
                "score": score,
            }
        )

    return results


import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi


def tokenize_text(text: str) -> list[str]:
    """Normalize text into lowercase word tokens."""
    return text.lower().split()


def normalize_fiscal_year(value: object) -> str | None:
    """Normalize fiscal-year metadata to YYYY-YY format."""
    text = str(value).strip().upper()
    text = re.sub(r"^FY\s*", "", text)
    text = re.sub(r"[^0-9]", "", text)

    if len(text) == 6:
        return f"{text[:4]}-{text[4:]}"
    if len(text) == 8 and text[4:6] == "20":
        return f"{text[:4]}-{text[6:]}"

    return None


def build_bm25_index(
    chunks: Sequence[dict[str, object]],
) -> BM25Okapi:
    """Build a BM25 index from report chunks."""
    if not chunks:
        raise ValueError("chunks must contain at least one item")

    tokenized_chunks = [
        tokenize_text(str(chunk["text"])) for chunk in chunks
    ]

    if any(not tokens for tokens in tokenized_chunks):
        raise ValueError("chunks must not contain blank text")

    return BM25Okapi(tokenized_chunks)


def search_bm25(
    query: str,
    chunks: Sequence[dict[str, object]],
    index: BM25Okapi,
    top_k: int = 5,
    allowed_symbols: set[str] | None = None,
    allowed_fiscal_years: set[str] | None = None,
) -> list[dict[str, int | float | str]]:
    """Retrieve chunks with optional company and fiscal-year filters."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    scores = index.get_scores(tokenize_text(query))

    if len(scores) != len(chunks):
        raise ValueError("BM25 score count must match the number of chunks")

    normalized_symbols = (
        {symbol.strip().upper() for symbol in allowed_symbols}
        if allowed_symbols is not None
        else None
    )

    normalized_years = (
        {
            year
            for value in allowed_fiscal_years
            if (year := normalize_fiscal_year(value)) is not None
        }
        if allowed_fiscal_years is not None
        else None
    )

    ranked_indices = sorted(
        (
            position
            for position, chunk in enumerate(chunks)
            if (
                normalized_symbols is None
                or str(chunk.get("symbol", "")).strip().upper()
                in normalized_symbols
            )
            and (
                normalized_years is None
                or normalize_fiscal_year(chunk.get("fiscal_year", ""))
                in normalized_years
            )
        ),
        key=lambda position: scores[position],
        reverse=True,
    )

    results: list[dict[str, int | float | str]] = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"]),
                "source_doc": str(chunk["source_doc"]),
                "page_number": int(chunk["page_number"]),
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results

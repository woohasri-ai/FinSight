
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


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


def load_embedding_model(
    model_name: str = MODEL_NAME,
) -> SentenceTransformer:
    """Load the embedding model on the CPU."""
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Convert texts into normalized embedding vectors."""
    if not texts:
        raise ValueError("texts must contain at least one item")

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain blank items")

    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def load_saved_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load precomputed embeddings from a NumPy file."""
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path, allow_pickle=False)

    if embeddings.ndim != 2:
        raise ValueError("Saved embeddings must be a two-dimensional array")

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("Saved embeddings must not be empty")

    return np.asarray(embeddings, dtype=np.float32)


def search_chunks(
    query: str,
    chunks: Sequence[dict[str, object]],
    model: SentenceTransformer,
    top_k: int = 5,
    chunk_embeddings: np.ndarray | None = None,
    allowed_symbols: set[str] | None = None,
    allowed_fiscal_years: set[str] | None = None,
) -> list[dict[str, int | float | str]]:
    """Search chunks with optional company and fiscal-year filters."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    if chunk_embeddings is None:
        chunk_embeddings = embed_texts(
            [str(chunk["text"]) for chunk in chunks],
            model,
        )

    if chunk_embeddings.ndim != 2:
        raise ValueError("Chunk embeddings must be a two-dimensional array")

    if chunk_embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count must match the number of chunks")

    query_embedding = embed_texts([query], model)[0]

    if chunk_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("Query and chunk embedding dimensions must match")

    scores = chunk_embeddings @ query_embedding

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

    candidate_indices = np.asarray(
        [
            index
            for index, chunk in enumerate(chunks)
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
        ],
        dtype=np.int64,
    )

    if candidate_indices.size == 0:
        return []

    ranked_positions = np.argsort(scores[candidate_indices])[::-1][:top_k]
    ranked_indices = candidate_indices[ranked_positions]

    results: list[dict[str, int | float | str]] = []

    for index in ranked_indices:
        chunk = chunks[int(index)]
        results.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"]),
                "source_doc": str(chunk["source_doc"]),
                "page_number": int(chunk["page_number"]),
                "score": float(scores[index]),
            }
        )

    return results

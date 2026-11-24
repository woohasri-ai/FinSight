
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from finsight.retrieval.bm25 import search_bm25
from finsight.retrieval.dense import search_chunks
from finsight.retrieval.fusion import reciprocal_rank_fusion


def hybrid_search(
    query: str,
    chunks: list[dict[str, int | float | str]],
    model: SentenceTransformer,
    bm25_index: BM25Okapi,
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
    candidate_k: int = 10,
    allowed_symbols: set[str] | None = None,
    allowed_fiscal_years: set[str] | None = None,
) -> list[dict[str, int | float | str]]:
    """Retrieve and fuse passages with optional company and year filters."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be greater than zero")

    if not chunks:
        return []

    dense_results = search_chunks(
        query,
        chunks,
        model,
        top_k=candidate_k,
        chunk_embeddings=chunk_embeddings,
        allowed_symbols=allowed_symbols,
        allowed_fiscal_years=allowed_fiscal_years,
    )

    bm25_results = search_bm25(
        query,
        chunks,
        bm25_index,
        top_k=candidate_k,
        allowed_symbols=allowed_symbols,
        allowed_fiscal_years=allowed_fiscal_years,
    )

    fused_results = reciprocal_rank_fusion(dense_results, bm25_results)

    return fused_results[:top_k]

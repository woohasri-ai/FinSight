import json
from pathlib import Path

from finsight.agent.pipeline import FinSightAgent
from finsight.llm.client import LLMClient
from finsight.llm.ollama import OllamaClient
from finsight.retrieval.bm25 import build_bm25_index
from finsight.retrieval.dense import (
    load_embedding_model,
    load_saved_embeddings,
)
from finsight.retrieval.rerank import load_reranker
from finsight.sql.database import DEFAULT_DATABASE_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_company"
    / "tier1_combined_chunks.json"
)

DEFAULT_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_company"
    / "tier1_combined_embeddings.npy"
)


def load_chunks(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> list[dict[str, int | float | str]]:
    """Load processed document chunks from a JSON file."""

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("The chunk file must contain a JSON list.")

    if not chunks:
        raise ValueError("The chunk file contains no chunks.")

    if not all(isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Every chunk must be a JSON object.")

    return chunks


def create_agent(
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    ollama_model: str = "qwen2.5:1.5b",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: float = 300.0,
    ollama_client: LLMClient | None = None,
    sql_database_path: Path = DEFAULT_DATABASE_PATH,
    candidate_k: int = 15,
    hybrid_top_k: int = 15,
    rerank_top_k: int = 5,
) -> FinSightAgent:
    """Load runtime resources and construct a FinSight agent."""

    chunks = load_chunks(chunks_path)
    chunk_embeddings = load_saved_embeddings(embeddings_path)

    embedding_model = load_embedding_model()
    bm25_index = build_bm25_index(chunks)
    reranker = load_reranker()

    llm = ollama_client or OllamaClient(
        model=ollama_model,
        base_url=ollama_base_url,
        timeout=ollama_timeout,
    )

    return FinSightAgent(
        chunks=chunks,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        chunk_embeddings=chunk_embeddings,
        reranker=reranker,
        llm=llm,
        sql_database_path=sql_database_path,
        candidate_k=candidate_k,
        hybrid_top_k=hybrid_top_k,
        rerank_top_k=rerank_top_k,
    )




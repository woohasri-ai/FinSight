
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

import finsight.agent.factory as factory
from finsight.agent.factory import load_chunks


def test_load_chunks_reads_json_file(tmp_path):
    expected_chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "Total employee strength was 584,519.",
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    chunks_path.write_text(
        json.dumps(expected_chunks),
        encoding="utf-8",
    )

    result = load_chunks(chunks_path)

    assert result == expected_chunks


def test_load_chunks_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing_chunks.json"

    with pytest.raises(FileNotFoundError, match="Chunk file not found"):
        load_chunks(missing_path)


def test_load_chunks_rejects_empty_list(tmp_path):
    chunks_path = tmp_path / "empty_chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="chunk file contains no chunks",
    ):
        load_chunks(chunks_path)


def test_load_chunks_rejects_non_list_json(tmp_path):
    chunks_path = tmp_path / "invalid_chunks.json"

    chunks_path.write_text(
        '{"chunk_id": "chunk-1"}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a JSON list",
    ):
        load_chunks(chunks_path)


def test_load_chunks_rejects_non_dictionary_entries(tmp_path):
    chunks_path = tmp_path / "invalid_entries.json"

    chunks_path.write_text(
        '["Some document text", 123]',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Every chunk must be a JSON object",
    ):
        load_chunks(chunks_path)


def test_load_chunks_rejects_malformed_json(tmp_path):
    chunks_path = tmp_path / "malformed_chunks.json"

    chunks_path.write_text(
        '{"chunk_id": "chunk-1"',
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        load_chunks(chunks_path)


def test_create_agent_initializes_and_injects_dependencies(
    tmp_path,
    monkeypatch,
):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "chunk-1",
            "text": "Total employee strength was 584,519.",
        }
    ]

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks),
        encoding="utf-8",
    )

    embeddings_path = tmp_path / "embeddings.npy"

    embeddings = np.array([[0.1, 0.2]], dtype=np.float32)
    embedding_model = MagicMock()
    bm25_index = MagicMock()
    reranker = MagicMock()
    llm = MagicMock()
    expected_agent = MagicMock()

    monkeypatch.setattr(
        factory,
        "load_saved_embeddings",
        MagicMock(return_value=embeddings),
    )
    monkeypatch.setattr(
        factory,
        "load_embedding_model",
        MagicMock(return_value=embedding_model),
    )
    monkeypatch.setattr(
        factory,
        "build_bm25_index",
        MagicMock(return_value=bm25_index),
    )
    monkeypatch.setattr(
        factory,
        "load_reranker",
        MagicMock(return_value=reranker),
    )
    monkeypatch.setattr(
        factory,
        "OllamaClient",
        MagicMock(return_value=llm),
    )
    monkeypatch.setattr(
        factory,
        "FinSightAgent",
        MagicMock(return_value=expected_agent),
    )

    result = factory.create_agent(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
    )

    assert result is expected_agent

    factory.load_saved_embeddings.assert_called_once_with(embeddings_path)
    factory.load_embedding_model.assert_called_once()
    factory.build_bm25_index.assert_called_once_with(chunks)
    factory.load_reranker.assert_called_once()
    factory.OllamaClient.assert_called_once()

    factory.FinSightAgent.assert_called_once_with(
        chunks=chunks,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        chunk_embeddings=embeddings,
        reranker=reranker,
        llm=llm,
        sql_database_path=factory.DEFAULT_DATABASE_PATH,
        candidate_k=15,
        hybrid_top_k=15,
        rerank_top_k=5,
    )


def test_create_agent_passes_custom_configuration(tmp_path, monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "chunk-1",
            "text": "Total employee strength was 584,519.",
        }
    ]

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    embeddings = np.array([[0.1, 0.2]], dtype=np.float32)
    custom_sql_path = tmp_path / "custom_finsight.db"

    monkeypatch.setattr(
        factory,
        "load_saved_embeddings",
        MagicMock(return_value=embeddings),
    )
    monkeypatch.setattr(factory, "load_embedding_model", MagicMock())
    monkeypatch.setattr(factory, "build_bm25_index", MagicMock())
    monkeypatch.setattr(factory, "load_reranker", MagicMock())

    ollama_mock = MagicMock()
    agent_mock = MagicMock()

    monkeypatch.setattr(factory, "OllamaClient", ollama_mock)
    monkeypatch.setattr(factory, "FinSightAgent", agent_mock)

    factory.create_agent(
        chunks_path=chunks_path,
        embeddings_path=tmp_path / "embeddings.npy",
        ollama_model="qwen2.5:3b",
        ollama_base_url="http://localhost:11434",
        ollama_timeout=180.0,
        sql_database_path=custom_sql_path,
        candidate_k=30,
        hybrid_top_k=20,
        rerank_top_k=3,
    )

    ollama_mock.assert_called_once_with(
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
        timeout=180.0,
    )

    agent_kwargs = agent_mock.call_args.kwargs

    assert agent_kwargs["sql_database_path"] == custom_sql_path
    assert agent_kwargs["candidate_k"] == 30
    assert agent_kwargs["hybrid_top_k"] == 20
    assert agent_kwargs["rerank_top_k"] == 3

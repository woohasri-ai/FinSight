from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import finsight.agent.pipeline as pipeline
from finsight.agent.pipeline import FinSightAgent
from finsight.generation.schemas import Citation, GeneratedAnswer
from finsight.sql.database import get_connection
from finsight.sql.seeding import seed_verified_facts


def test_agent_initializes_with_dependencies():
    chunks = [
        {"text": "First document chunk."},
        {"text": "Second document chunk."},
    ]

    embedding_model = MagicMock()
    bm25_index = MagicMock()
    embeddings = np.array(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float32,
    )
    reranker = MagicMock()
    llm = MagicMock()

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        chunk_embeddings=embeddings,
        reranker=reranker,
        llm=llm,
    )

    assert agent.chunks is chunks
    assert agent.embedding_model is embedding_model
    assert agent.bm25_index is bm25_index
    assert agent.chunk_embeddings is embeddings
    assert agent.reranker is reranker
    assert agent.llm is llm
    assert agent.sql_database_path == pipeline.DEFAULT_DATABASE_PATH

    assert agent.candidate_k == 50
    assert agent.hybrid_top_k == 50
    assert agent.rerank_top_k == 5


def test_agent_rejects_empty_chunks():
    with pytest.raises(ValueError, match="chunks cannot be empty"):
        FinSightAgent(
            chunks=[],
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.empty((0, 2), dtype=np.float32),
            reranker=MagicMock(),
            llm=MagicMock(),
        )


def test_agent_rejects_mismatched_embedding_count():
    chunks = [
        {"text": "First."},
        {"text": "Second."},
    ]

    with pytest.raises(
        ValueError,
        match="Number of embeddings must match",
    ):
        FinSightAgent(
            chunks=chunks,
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.array(
                [[0.1, 0.2]],
                dtype=np.float32,
            ),
            reranker=MagicMock(),
            llm=MagicMock(),
        )


def test_agent_rejects_non_2d_embeddings():
    with pytest.raises(
        ValueError,
        match="chunk_embeddings must be a 2D array",
    ):
        FinSightAgent(
            chunks=[{"text": "First document chunk."}],
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.array(
                [0.1, 0.2],
                dtype=np.float32,
            ),
            reranker=MagicMock(),
            llm=MagicMock(),
        )


def test_agent_rejects_invalid_candidate_k():
    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than zero",
    ):
        FinSightAgent(
            chunks=[{"text": "First."}],
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.array(
                [[0.1, 0.2]],
                dtype=np.float32,
            ),
            reranker=MagicMock(),
            llm=MagicMock(),
            candidate_k=0,
        )


def test_agent_rejects_invalid_hybrid_top_k():
    with pytest.raises(
        ValueError,
        match="hybrid_top_k must be greater than zero",
    ):
        FinSightAgent(
            chunks=[{"text": "First."}],
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.array(
                [[0.1, 0.2]],
                dtype=np.float32,
            ),
            reranker=MagicMock(),
            llm=MagicMock(),
            hybrid_top_k=0,
        )


def test_agent_rejects_invalid_rerank_top_k():
    with pytest.raises(
        ValueError,
        match="rerank_top_k must be greater than zero",
    ):
        FinSightAgent(
            chunks=[{"text": "First."}],
            embedding_model=MagicMock(),
            bm25_index=MagicMock(),
            chunk_embeddings=np.array(
                [[0.1, 0.2]],
                dtype=np.float32,
            ),
            reranker=MagicMock(),
            llm=MagicMock(),
            rerank_top_k=0,
        )


@pytest.mark.asyncio
async def test_agent_answer_runs_complete_pipeline(monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "TCS revenue information.",
        }
    ]

    reranked_chunks = [
        {
            **chunks[0],
            "score": 0.95,
        }
    ]

    expected_answer = GeneratedAnswer(
        answer="TCS revenue information was retrieved.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26",
                page_number=25,
                chunk_id="tcs:page-25:chunk-1",
                evidence_text="TCS revenue information.",
            )
        ],
        abstained=False,
    )

    call_order = []

    def fake_hybrid_search(**kwargs):
        call_order.append("retrieval")
        return chunks

    def fake_rerank_chunks(**kwargs):
        call_order.append("reranking")
        return reranked_chunks

    async def fake_generate_answer(**kwargs):
        call_order.append("generation")
        return expected_answer

    hybrid_mock = MagicMock(side_effect=fake_hybrid_search)
    rerank_mock = MagicMock(side_effect=fake_rerank_chunks)
    generation_mock = AsyncMock(side_effect=fake_generate_answer)

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
    )

    result = await agent.answer("What was TCS's revenue?")

    assert call_order == [
        "retrieval",
        "reranking",
        "generation",
    ]

    assert result is expected_answer
    assert result.abstained is False

    hybrid_mock.assert_called_once()
    rerank_mock.assert_called_once()
    generation_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_abstains_when_retrieval_returns_no_chunks(
    monkeypatch,
):
    retrieval_mock = MagicMock(return_value=[])

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        retrieval_mock,
    )

    reranker = MagicMock()

    # AsyncMock is required because generate() is awaited.
    llm = AsyncMock()

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=reranker,
        llm=llm,
    )

    result = await agent.answer("What was TCS's monthly attrition rate?")

    assert result.abstained is True
    assert result.citations == []
    assert result.answer

    retrieval_mock.assert_called_once()
    reranker.predict.assert_not_called()
    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_rejects_blank_question(monkeypatch):
    retrieval_mock = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        retrieval_mock,
    )

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be blank",
    ):
        await agent.answer("   ")

    retrieval_mock.assert_not_called()


@pytest.mark.asyncio
async def test_agent_passes_configured_limits(monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "TCS revenue information.",
        }
    ]

    hybrid_mock = MagicMock(return_value=chunks)
    rerank_mock = MagicMock(return_value=chunks)

    expected_answer = GeneratedAnswer(
        answer="TCS revenue information was retrieved.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26",
                page_number=25,
                chunk_id="tcs:page-25:chunk-1",
                evidence_text="TCS revenue information.",
            )
        ],
        abstained=False,
    )

    generation_mock = AsyncMock(return_value=expected_answer)

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        candidate_k=20,
        hybrid_top_k=15,
        rerank_top_k=3,
    )

    await agent.answer("What was TCS's revenue?")

    assert hybrid_mock.call_args.kwargs["candidate_k"] == 20
    assert hybrid_mock.call_args.kwargs["top_k"] == 15
    assert rerank_mock.call_args.kwargs["top_k"] == 3
    assert generation_mock.call_args.kwargs["top_k"] == 3


@pytest.mark.asyncio
async def test_agent_propagates_retrieval_error(monkeypatch):
    retrieval_mock = MagicMock(side_effect=RuntimeError("Hybrid retrieval failed."))

    rerank_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        retrieval_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
    )

    with pytest.raises(
        RuntimeError,
        match="Hybrid retrieval failed",
    ):
        await agent.answer("What was TCS's revenue?")

    retrieval_mock.assert_called_once()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_propagates_reranking_error(monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "TCS revenue information.",
        }
    ]

    hybrid_mock = MagicMock(return_value=chunks)

    rerank_mock = MagicMock(side_effect=RuntimeError("Reranking failed."))

    generation_mock = AsyncMock()

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
    )

    with pytest.raises(
        RuntimeError,
        match="Reranking failed",
    ):
        await agent.answer("What was TCS's revenue?")

    hybrid_mock.assert_called_once()
    rerank_mock.assert_called_once()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_propagates_generation_error(monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "TCS revenue information.",
        }
    ]

    hybrid_mock = MagicMock(return_value=chunks)
    rerank_mock = MagicMock(return_value=chunks)

    generation_mock = AsyncMock(side_effect=RuntimeError("Generation failed."))

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
    )

    with pytest.raises(
        RuntimeError,
        match="Generation failed",
    ):
        await agent.answer("What was TCS's revenue?")

    hybrid_mock.assert_called_once()
    rerank_mock.assert_called_once()
    generation_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_passes_reranked_chunks_to_generation(monkeypatch):
    retrieved_chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "chunk-1",
            "text": "Original retrieval result.",
        }
    ]

    reranked_chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 58,
            "chunk_id": "chunk-2",
            "text": "Reranked revenue evidence.",
            "score": 0.95,
        }
    ]

    expected_answer = GeneratedAnswer(
        answer="TCS revenue information was retrieved.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26",
                page_number=58,
                chunk_id="chunk-2",
                evidence_text="Reranked revenue evidence.",
            )
        ],
        abstained=False,
    )

    hybrid_mock = MagicMock(return_value=retrieved_chunks)
    rerank_mock = MagicMock(return_value=reranked_chunks)
    generation_mock = AsyncMock(return_value=expected_answer)

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    llm = MagicMock()

    agent = FinSightAgent(
        chunks=retrieved_chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=llm,
    )

    question = "What was TCS's revenue?"
    result = await agent.answer(question)

    generation_kwargs = generation_mock.call_args.kwargs

    assert generation_kwargs["question"] == question
    assert generation_kwargs["evidence"] is reranked_chunks
    assert generation_kwargs["llm"] is llm
    assert generation_kwargs["top_k"] == 5

    assert result.citations[0].page_number == 58


@pytest.mark.asyncio
async def test_agent_passes_correct_dependencies(monkeypatch):
    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "chunk-1",
            "text": "TCS revenue information.",
        }
    ]

    hybrid_mock = MagicMock(return_value=chunks)
    rerank_mock = MagicMock(return_value=chunks)

    expected_answer = GeneratedAnswer(
        answer="TCS revenue information was retrieved.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26",
                page_number=25,
                chunk_id="chunk-1",
                evidence_text="TCS revenue information.",
            )
        ],
        abstained=False,
    )

    generation_mock = AsyncMock(return_value=expected_answer)

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    embedding_model = MagicMock()
    bm25_index = MagicMock()
    embeddings = np.array(
        [[0.1, 0.2]],
        dtype=np.float32,
    )
    reranker = MagicMock()

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        chunk_embeddings=embeddings,
        reranker=reranker,
        llm=MagicMock(),
    )

    question = "What was TCS's revenue?"
    await agent.answer(question)

    hybrid_kwargs = hybrid_mock.call_args.kwargs

    assert hybrid_kwargs["query"] == question
    assert hybrid_kwargs["model"] is embedding_model
    assert hybrid_kwargs["bm25_index"] is bm25_index
    assert hybrid_kwargs["chunk_embeddings"] is embeddings

    rerank_kwargs = rerank_mock.call_args.kwargs

    assert rerank_kwargs["query"] == question
    assert rerank_kwargs["model"] is reranker


@pytest.mark.asyncio
async def test_agent_answers_employee_count_directly_from_sql(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"
    seed_verified_facts(db_path)

    hybrid_mock = MagicMock()
    rerank_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    result = await agent.answer("What was TCS's total employee strength in FY2026?")

    assert result.abstained is False
    assert "617,437" in result.answer
    assert "584,519" in result.answer
    assert "32,918" in result.answer

    assert len(result.citations) == 1

    assert result.citations[0].page_number == 108
    assert result.citations[0].chunk_id == ("tcs_annual_report_2025_26.pdf:page-108:chunk-1")

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_falls_back_to_rag_when_sql_has_no_matching_facts(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"
    seed_verified_facts(db_path)

    chunks = [
        {
            "source_doc": "tcs_annual_report_2025_26",
            "page_number": 25,
            "chunk_id": "tcs:page-25:chunk-1",
            "text": "TCS annual report employee information.",
        }
    ]

    expected_answer = GeneratedAnswer(
        answer="TCS employee information was retrieved through RAG.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26",
                page_number=25,
                chunk_id="tcs:page-25:chunk-1",
                evidence_text="TCS annual report employee information.",
            )
        ],
        abstained=False,
    )

    hybrid_mock = MagicMock(return_value=chunks)
    rerank_mock = MagicMock(return_value=chunks)
    generation_mock = AsyncMock(return_value=expected_answer)

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    result = await agent.answer("What was TCS's employee strength in FY2025?")

    assert result is expected_answer

    hybrid_mock.assert_called_once()
    rerank_mock.assert_called_once()
    generation_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_rejects_inconsistent_sql_data_without_rag_fallback(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"
    seed_verified_facts(db_path)

    connection = get_connection(db_path)

    try:
        connection.execute(
            """
            UPDATE verified_facts
            SET value = ?
            WHERE company = ?
            AND fiscal_year = ?
            AND metric = ?
            AND category = ?
            """,
            (600000, "TCS", "FY2026", "employee_count", "total"),
        )

        connection.commit()

    finally:
        connection.close()

    hybrid_mock = MagicMock()
    rerank_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        hybrid_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "rerank_chunks",
        rerank_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        generation_mock,
    )

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    with pytest.raises(
        ValueError,
        match="employee counts are inconsistent",
    ):
        await agent.answer("What was TCS's total employee strength in FY2026?")

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_answers_permanent_employee_question_from_sql(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    hybrid_mock = MagicMock()
    rerank_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "rerank_chunks", rerank_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    result = await agent.answer("How many permanent employees did TCS have in FY2026?")

    assert result.abstained is False
    assert "584,519" in result.answer
    assert "617,437" not in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 108
    assert result.citations[0].chunk_id == ("tcs_annual_report_2025_26.pdf:page-108:chunk-1")

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_answers_other_than_permanent_employee_question_from_sql(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "finsight.db"

    seed_verified_facts(db_path)

    hybrid_mock = MagicMock()
    rerank_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "rerank_chunks", rerank_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = FinSightAgent(
        chunks=[{"text": "TCS annual report content."}],
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=db_path,
    )

    result = await agent.answer("How many other than permanent employees did TCS have in FY2026?")

    assert result.abstained is False
    assert "32,918" in result.answer
    assert "584,519" not in result.answer
    assert "617,437" not in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].page_number == 108
    assert result.citations[0].chunk_id == ("tcs_annual_report_2025_26.pdf:page-108:chunk-1")

    hybrid_mock.assert_not_called()
    rerank_mock.assert_not_called()
    generation_mock.assert_not_awaited()


def _make_year_scope_agent(chunks, tmp_path):
    return FinSightAgent(
        chunks=chunks,
        embedding_model=MagicMock(),
        bm25_index=MagicMock(),
        chunk_embeddings=np.zeros((len(chunks), 2), dtype=np.float32),
        reranker=MagicMock(),
        llm=MagicMock(),
        sql_database_path=tmp_path / "nonexistent_test_database.db",
    )


@pytest.mark.asyncio
async def test_agent_passes_requested_fiscal_year_to_retrieval(
    tmp_path, monkeypatch
):
    chunks = [
        {
            "chunk_id": "tcs-2023",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2023_24.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2023-24",
        },
        {
            "chunk_id": "tcs-2024",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2024_25.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2024-25",
        },
    ]

    hybrid_mock = MagicMock(return_value=[])
    generation_mock = AsyncMock(
        return_value=GeneratedAnswer(
            answer="No matching evidence.",
            citations=[],
            abstained=True,
        )
    )

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = _make_year_scope_agent(chunks, tmp_path)

    await agent.answer("What was TCS revenue in FY 2023-24?")

    kwargs = hybrid_mock.call_args.kwargs
    assert kwargs["allowed_symbols"] == {"TCS"}
    assert kwargs["allowed_fiscal_years"] == {"2023-24"}


@pytest.mark.asyncio
async def test_agent_abstains_when_requested_company_year_is_unavailable(
    tmp_path, monkeypatch
):
    chunks = [
        {
            "chunk_id": "infy-2025",
            "text": "Infosys revenue information.",
            "source_doc": "infosys_annual_report_2025_26.pdf",
            "page_number": 10,
            "company_name": "Infosys Ltd.",
            "symbol": "INFY",
            "fiscal_year": "2025-26",
        }
    ]

    hybrid_mock = MagicMock()
    generation_mock = AsyncMock()

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = _make_year_scope_agent(chunks, tmp_path)

    result = await agent.answer(
        "What was Infosys revenue in FY 2023-24?"
    )

    assert result.abstained is True
    assert result.citations == []
    assert "2023-24" in result.answer
    assert "not available" in result.answer

    hybrid_mock.assert_not_called()
    generation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_passes_multiple_requested_years_to_retrieval(
    tmp_path, monkeypatch
):
    chunks = [
        {
            "chunk_id": "tcs-2023",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2023_24.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2023-24",
        },
        {
            "chunk_id": "tcs-2024",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2024_25.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2024-25",
        },
    ]

    hybrid_mock = MagicMock(return_value=[])
    generation_mock = AsyncMock(
        return_value=GeneratedAnswer(
            answer="No matching evidence.",
            citations=[],
            abstained=True,
        )
    )

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = _make_year_scope_agent(chunks, tmp_path)

    await agent.answer(
        "Compare TCS revenue in FY 2023-24 and FY 2024-25."
    )

    kwargs = hybrid_mock.call_args.kwargs
    assert kwargs["allowed_symbols"] == {"TCS"}
    assert kwargs["allowed_fiscal_years"] == {
        "2023-24",
        "2024-25",
    }


@pytest.mark.asyncio
async def test_agent_keeps_retrieval_unrestricted_when_year_is_not_specified(
    tmp_path, monkeypatch
):
    chunks = [
        {
            "chunk_id": "tcs-2023",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2023_24.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2023-24",
        },
        {
            "chunk_id": "tcs-2024",
            "text": "TCS revenue information.",
            "source_doc": "tcs_annual_report_2024_25.pdf",
            "page_number": 10,
            "company_name": "Tata Consultancy Services Ltd.",
            "symbol": "TCS",
            "fiscal_year": "2024-25",
        },
    ]

    hybrid_mock = MagicMock(return_value=[])
    generation_mock = AsyncMock(
        return_value=GeneratedAnswer(
            answer="No matching evidence.",
            citations=[],
            abstained=True,
        )
    )

    monkeypatch.setattr(pipeline, "hybrid_search", hybrid_mock)
    monkeypatch.setattr(pipeline, "generate_answer", generation_mock)

    agent = _make_year_scope_agent(chunks, tmp_path)

    await agent.answer("What was TCS revenue?")

    kwargs = hybrid_mock.call_args.kwargs
    assert kwargs["allowed_symbols"] == {"TCS"}
    assert kwargs["allowed_fiscal_years"] is None

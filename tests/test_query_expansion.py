import json

import pytest

from finsight.retrieval.query_expansion import expand_retrieval_queries, rewrite_retrieval_query


class FakeLLM:
    def __init__(
        self,
        response: str = '{"queries": ["alternative one", "alternative two"]}',
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.last_response_format = None

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict | str | None = None,
    ) -> str:
        self.last_response_format = response_format

        if self.error:
            raise self.error

        return self.response


@pytest.mark.asyncio
async def test_expands_question_and_preserves_original() -> None:
    llm = FakeLLM()

    queries = await expand_retrieval_queries(
        "What measurable improvements did LTM achieve?",
        llm,  # type: ignore[arg-type]
    )

    assert queries == [
        "What measurable improvements did LTM achieve?",
        "alternative one",
        "alternative two",
    ]
    assert llm.last_response_format is not None


@pytest.mark.asyncio
async def test_removes_duplicates_and_original_question() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "queries": [
                    "Alternative query",
                    "Alternative query",
                    "Original question",
                ]
            }
        )
    )

    queries = await expand_retrieval_queries(
        "Original question",
        llm,  # type: ignore[arg-type]
    )

    assert queries == ["Original question", "Alternative query"]


@pytest.mark.asyncio
async def test_ignores_invalid_and_blank_queries() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "queries": [
                    "",
                    "   ",
                    123,
                    "Useful alternative",
                ]
            }
        )
    )

    queries = await expand_retrieval_queries(
        "Original question",
        llm,  # type: ignore[arg-type]
    )

    assert queries == ["Original question", "Useful alternative"]


@pytest.mark.asyncio
async def test_returns_original_for_invalid_json() -> None:
    llm = FakeLLM(response="not valid json")

    queries = await expand_retrieval_queries(
        "Original question",
        llm,  # type: ignore[arg-type]
    )

    assert queries == ["Original question"]


@pytest.mark.asyncio
async def test_returns_original_when_llm_fails() -> None:
    llm = FakeLLM(error=RuntimeError("LLM unavailable"))

    queries = await expand_retrieval_queries(
        "Original question",
        llm,  # type: ignore[arg-type]
    )

    assert queries == ["Original question"]


@pytest.mark.asyncio
async def test_rejects_blank_question() -> None:
    llm = FakeLLM()

    with pytest.raises(ValueError, match="Question cannot be blank"):
        await expand_retrieval_queries(
            "   ",
            llm,  # type: ignore[arg-type]
        )



@pytest.mark.asyncio
async def test_rewrites_question_successfully() -> None:
    llm = FakeLLM(
        response=json.dumps(
            {
                "query": (
                    "LTM percentage reductions in digital marketing "
                    "time-to-market and operational costs"
                )
            }
        )
    )

    rewritten = await rewrite_retrieval_query(
        "What measurable improvements did LTM achieve?",
        llm,  # type: ignore[arg-type]
    )

    assert rewritten == (
        "LTM percentage reductions in digital marketing "
        "time-to-market and operational costs"
    )
    assert llm.last_response_format is not None


@pytest.mark.asyncio
async def test_returns_original_for_invalid_json_during_rewrite() -> None:
    llm = FakeLLM(response="not valid json")
    question = "What measurable improvements did LTM achieve?"

    rewritten = await rewrite_retrieval_query(
        question,
        llm,  # type: ignore[arg-type]
    )

    assert rewritten == question


@pytest.mark.asyncio
async def test_returns_original_for_missing_or_blank_rewrite() -> None:
    question = "What measurable improvements did LTM achieve?"

    for response in (
        json.dumps({}),
        json.dumps({"query": ""}),
        json.dumps({"query": "   "}),
        json.dumps({"query": 123}),
    ):
        llm = FakeLLM(response=response)

        rewritten = await rewrite_retrieval_query(
            question,
            llm,  # type: ignore[arg-type]
        )

        assert rewritten == question


@pytest.mark.asyncio
async def test_returns_original_when_rewrite_is_unchanged() -> None:
    question = "What measurable improvements did LTM achieve?"
    llm = FakeLLM(
        response=json.dumps({"query": question})
    )

    rewritten = await rewrite_retrieval_query(
        question,
        llm,  # type: ignore[arg-type]
    )

    assert rewritten == question


@pytest.mark.asyncio
async def test_returns_original_when_rewrite_llm_fails() -> None:
    question = "What measurable improvements did LTM achieve?"
    llm = FakeLLM(error=RuntimeError("LLM unavailable"))

    rewritten = await rewrite_retrieval_query(
        question,
        llm,  # type: ignore[arg-type]
    )

    assert rewritten == question


@pytest.mark.asyncio
async def test_rejects_blank_question_during_rewrite() -> None:
    llm = FakeLLM()

    with pytest.raises(ValueError, match="Question cannot be blank"):
        await rewrite_retrieval_query(
            "   ",
            llm,  # type: ignore[arg-type]
        )

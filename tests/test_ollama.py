import json

import httpx
import pytest

from finsight.llm.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)


@pytest.mark.asyncio
async def test_generate_returns_model_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen2.5:3b"
        assert payload["stream"] is False

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "RAG combines retrieval and generation.",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()
    response = await client.generate("What is RAG?")

    assert response == "RAG combines retrieval and generation."


@pytest.mark.asyncio
async def test_generate_includes_system_prompt(monkeypatch):
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Answer based on the provided evidence.",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    response = await client.generate(
        "What is RAG?",
        system="Answer only using the supplied context.",
    )

    assert response == "Answer based on the provided evidence."
    assert captured_payload["messages"][0] == {
        "role": "system",
        "content": "Answer only using the supplied context.",
    }
    assert captured_payload["messages"][1] == {
        "role": "user",
        "content": "What is RAG?",
    }


@pytest.mark.asyncio
async def test_generate_raises_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "Internal server error"},
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("What is RAG?")


def test_init_rejects_empty_model():
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        OllamaClient(model="   ")


def test_init_rejects_empty_base_url():
    with pytest.raises(ValueError, match="Base URL cannot be empty"):
        OllamaClient(base_url="   ")


def test_init_rejects_invalid_timeout():
    with pytest.raises(ValueError, match="Timeout must be greater than zero"):
        OllamaClient(timeout=0)


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt():
    client = OllamaClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await client.generate("   ")


@pytest.mark.asyncio
async def test_generate_rejects_empty_system_prompt():
    client = OllamaClient()

    with pytest.raises(ValueError, match="System prompt cannot be empty"):
        await client.generate(
            "What is RAG?",
            system="   ",
        )


@pytest.mark.asyncio
async def test_generate_raises_error_for_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not valid json",
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaResponseError,
        match="Ollama returned invalid JSON",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_error_when_message_is_missing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"unexpected": "data"},
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaResponseError,
        match="Ollama response is missing a valid message object",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_error_when_message_is_not_dict(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": "Unexpected text"},
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaResponseError,
        match="Ollama response is missing a valid message object",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_error_when_content_is_missing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaResponseError,
        match="Ollama response is missing valid content",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_error_when_content_is_empty(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "   ",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaResponseError,
        match="Ollama response is missing valid content",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaConnectionError,
        match="Could not connect to Ollama",
    ):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_generate_raises_timeout_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mocked_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.ollama.httpx.AsyncClient",
        mocked_async_client,
    )

    client = OllamaClient()

    with pytest.raises(
        OllamaTimeoutError,
        match="Ollama request timed out",
    ):
        await client.generate("What is RAG?")


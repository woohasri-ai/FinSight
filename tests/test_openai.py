import json

import httpx
import pytest

from finsight.llm.openai import (
    OpenAIClient,
    OpenAIConnectionError,
    OpenAIResponseError,
    OpenAITimeoutError,
)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


@pytest.mark.asyncio
async def test_generate_returns_response(api_key, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)

        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "gpt-4.1-mini"

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": "Test answer"}',
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.openai.httpx.AsyncClient",
        mocked_client,
    )

    client = OpenAIClient()
    result = await client.generate("Test question")

    assert result == '{"answer": "Test answer"}'


@pytest.mark.asyncio
async def test_generate_includes_structured_json_schema(
    api_key,
    monkeypatch,
):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": "Supported"}',
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.openai.httpx.AsyncClient",
        mocked_client,
    )

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    client = OpenAIClient()

    await client.generate(
        "Test question",
        system="Use evidence only",
        response_format=schema,
    )

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt(api_key):
    client = OpenAIClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await client.generate("   ")


@pytest.mark.asyncio
async def test_generate_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        OpenAIClient()


@pytest.mark.asyncio
async def test_generate_raises_connection_error(api_key, monkeypatch):
    def handler(request):
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.openai.httpx.AsyncClient",
        mocked_client,
    )

    with pytest.raises(OpenAIConnectionError):
        await OpenAIClient().generate("Test")


@pytest.mark.asyncio
async def test_generate_raises_timeout_error(api_key, monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.openai.httpx.AsyncClient",
        mocked_client,
    )

    with pytest.raises(OpenAITimeoutError):
        await OpenAIClient().generate("Test")


@pytest.mark.asyncio
async def test_generate_raises_response_error(api_key, monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={"unexpected": "response"},
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "finsight.llm.openai.httpx.AsyncClient",
        mocked_client,
    )

    with pytest.raises(OpenAIResponseError):
        await OpenAIClient().generate("Test")

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import finsight.api.main as api
from finsight.generation.generator import GenerationError
from finsight.generation.schemas import Citation, GeneratedAnswer


@pytest.fixture
def api_client(monkeypatch):
    expected_answer = GeneratedAnswer(
        answer="TCS reported 617,437 total employees in FY2026.",
        citations=[
            Citation(
                document_name="tcs_annual_report_2025_26.pdf",
                page_number=108,
                chunk_id="tcs_annual_report_2025_26.pdf:page-108:chunk-1",
                evidence_text="Total employees (D + E): 6,17,437",
            )
        ],
        abstained=False,
    )

    mock_agent = MagicMock()
    mock_agent.answer = AsyncMock(return_value=expected_answer)

    create_agent_mock = MagicMock(return_value=mock_agent)

    monkeypatch.setattr(api, "create_agent", create_agent_mock)

    with TestClient(api.app) as client:
        yield client, mock_agent, create_agent_mock


def test_health_endpoint_returns_ok(api_client):
    client, _, create_agent_mock = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    create_agent_mock.assert_called_once()


def test_ask_endpoint_returns_generated_answer(api_client):
    client, mock_agent, _ = api_client

    response = client.post(
        "/ask",
        json={"question": "What was TCS's total employee strength in FY2026?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "TCS reported 617,437 total employees in FY2026."
    assert data["abstained"] is False
    assert len(data["citations"]) == 1

    assert data["citations"][0]["page_number"] == 108
    assert data["citations"][0]["chunk_id"] == ("tcs_annual_report_2025_26.pdf:page-108:chunk-1")

    mock_agent.answer.assert_awaited_once()


def test_ask_endpoint_rejects_blank_question(api_client):
    client, mock_agent, _ = api_client

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422

    mock_agent.answer.assert_not_awaited()


def test_ask_endpoint_rejects_missing_question(api_client):
    client, mock_agent, _ = api_client

    response = client.post("/ask", json={})

    assert response.status_code == 422

    mock_agent.answer.assert_not_awaited()


def test_ask_endpoint_returns_abstention_on_generation_error(api_client):
    client, mock_agent, _ = api_client

    mock_agent.answer.side_effect = GenerationError(
        "The LLM returned a quotation that does not exactly match evidence S1."
    )

    response = client.post(
        "/ask",
        json={"question": "What was Apple's revenue in FY2026?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["abstained"] is True
    assert data["citations"] == []
    assert "does not provide sufficient information" in data["answer"]

    mock_agent.answer.assert_awaited_once()

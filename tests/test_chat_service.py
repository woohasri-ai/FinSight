import pytest
from pydantic import ValidationError

from finsight.chat.service import (
    ChatMessage,
    ChatRequest,
    GeneralChatService,
)


class FakeLLM:
    def __init__(self, response: str = "This is a test response.") -> None:
        self.response = response
        self.last_prompt = None
        self.last_system = None
        self.last_num_predict = None
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format=None,
        num_predict: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_num_predict = num_predict
        return self.response


@pytest.mark.asyncio
async def test_general_chat_returns_response_and_history():
    llm = FakeLLM("Python is a programming language.")
    service = GeneralChatService(llm)

    result = await service.chat(ChatRequest(message="What is Python?"))

    assert result.response == "Python is a programming language."
    assert len(result.conversation_history) == 2
    assert result.conversation_history[0].role == "user"
    assert result.conversation_history[0].content == "What is Python?"
    assert result.conversation_history[1].role == "assistant"


@pytest.mark.asyncio
async def test_general_chat_uses_previous_conversation_context():
    llm = FakeLLM("It is dynamically typed.")
    service = GeneralChatService(llm)

    history = [
        ChatMessage(role="user", content="What is Python?"),
        ChatMessage(
            role="assistant",
            content="Python is a programming language.",
        ),
    ]

    await service.chat(
        ChatRequest(
            message="Is it dynamically typed?",
            history=history,
        )
    )

    assert "What is Python?" in llm.last_prompt
    assert "Python is a programming language." in llm.last_prompt
    assert "Is it dynamically typed?" in llm.last_prompt
    assert "Career Guide" in llm.last_system


@pytest.mark.asyncio
async def test_old_roadmap_request_does_not_trigger_another_roadmap():
    llm = FakeLLM("Supervised learning uses labelled data.")
    service = GeneralChatService(llm)

    history = [
        ChatMessage(
            role="user",
            content="Create a 30-day roadmap for a fresher AI/ML Engineer.",
        ),
        ChatMessage(
            role="assistant",
            content="# 30-Day Learning Roadmap\n\n" + ("### Week 1: Python foundations\n" * 100),
        ),
    ]

    result = await service.chat(
        ChatRequest(
            message=("What is the difference between supervised and unsupervised learning?"),
            history=history,
        )
    )

    assert result.response == "Supervised learning uses labelled data."
    assert llm.call_count == 1
    assert "What is the difference between supervised" in llm.last_prompt


@pytest.mark.asyncio
async def test_ai_ml_roadmap_is_generated_for_a_new_roadmap_request():
    llm = FakeLLM()
    service = GeneralChatService(llm)

    result = await service.chat(
        ChatRequest(message="Create a 30-day roadmap for a fresher AI/ML Engineer.")
    )

    assert "30-Day Learning Roadmap" in result.response
    assert "| 30 |" in result.response
    assert "Customer Churn Prediction" in result.response
    assert llm.call_count == 0


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="Invalid role")


def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


@pytest.mark.asyncio
async def test_general_chat_rejects_empty_model_response():
    llm = FakeLLM("   ")
    service = GeneralChatService(llm)

    with pytest.raises(
        ValueError,
        match="language model returned an empty response",
    ):
        await service.chat(ChatRequest(message="Explain machine learning."))

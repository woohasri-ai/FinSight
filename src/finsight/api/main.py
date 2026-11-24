
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from finsight.agent.factory import create_agent
from finsight.chat.service import (
    ChatRequest,
    ChatResponse,
    GeneralChatService,
)
from finsight.generation.generator import GenerationError
from finsight.generation.schemas import GeneratedAnswer
from finsight.llm.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)

load_dotenv()

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """Validate an incoming Financial Research question."""

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize Financial Research and General Chat services."""

    ollama_client = OllamaClient()

    app.state.agent = create_agent(
        ollama_client=ollama_client,
    )
    app.state.chat_service = GeneralChatService(ollama_client)

    logger.info("Financial Research provider: local Ollama")
    logger.info("General Chat provider: local Ollama")

    yield

    app.state.agent = None
    app.state.chat_service = None


app = FastAPI(
    title="FinSight API",
    description=(
        "FinSight provides separate General Chat and evidence-grounded "
        "Financial Research experiences."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Check whether both FinSight services are initialized."""

    agent = getattr(request.app.state, "agent", None)
    chat_service = getattr(request.app.state, "chat_service", None)

    if agent is None or chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="FinSight services are not initialized.",
        )

    return {"status": "ok"}


@app.post("/ask", response_model=GeneratedAnswer)
async def ask_question(
    payload: AskRequest,
    request: Request,
) -> GeneratedAnswer:
    """Answer using the evidence-grounded Financial Research agent."""

    agent = getattr(request.app.state, "agent", None)

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Financial Research service is not initialized.",
        )

    try:
        return await agent.answer(payload.question)

    except GenerationError:
        logger.exception(
            "Answer generation failed validation; returning abstention."
        )
        return GeneratedAnswer(
            answer=(
                "The supplied evidence does not provide sufficient "
                "information to answer this question reliably."
            ),
            citations=[],
            abstained=True,
        )

    except OllamaConnectionError as exc:
        logger.exception("Ollama connection failed.")
        raise HTTPException(
            status_code=503,
            detail="The language model service is unavailable.",
        ) from exc

    except OllamaTimeoutError as exc:
        logger.exception("Ollama request timed out.")
        raise HTTPException(
            status_code=504,
            detail="The language model service timed out.",
        ) from exc

    except OllamaResponseError as exc:
        logger.exception("Ollama returned an invalid response.")
        raise HTTPException(
            status_code=502,
            detail="The language model service returned an invalid response.",
        ) from exc


@app.post("/chat", response_model=ChatResponse)
async def general_chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Answer general-purpose questions using local Ollama."""

    chat_service = getattr(request.app.state, "chat_service", None)

    if chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="General Chat service is not initialized.",
        )

    try:
        return await chat_service.chat(payload)

    except OllamaConnectionError as exc:
        logger.exception("Ollama connection failed during General Chat.")
        raise HTTPException(
            status_code=503,
            detail="The language model service is unavailable.",
        ) from exc

    except OllamaTimeoutError as exc:
        logger.exception("Ollama timed out during General Chat.")
        raise HTTPException(
            status_code=504,
            detail="The language model service timed out.",
        ) from exc

    except OllamaResponseError as exc:
        logger.exception(
            "Ollama returned an invalid General Chat response."
        )
        raise HTTPException(
            status_code=502,
            detail="The language model service returned an invalid response.",
        ) from exc

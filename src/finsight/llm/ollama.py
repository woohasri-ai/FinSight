import logging
import time
from typing import Any

import httpx


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

logger.propagate = False


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an invalid response."""


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama service cannot be reached."""


class OllamaTimeoutError(RuntimeError):
    """Raised when an Ollama request times out."""


class OllamaClient:
    """Async client for the Ollama chat API."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Model name cannot be empty")

        if not base_url.strip():
            raise ValueError("Base URL cannot be empty")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        num_predict: int | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if system is not None and not system.strip():
            raise ValueError("System prompt cannot be empty")

        if num_predict is not None and num_predict <= 0:
            raise ValueError("num_predict must be greater than zero")

        if response_format is not None and response_format != "json":
            if not isinstance(response_format, dict):
                raise ValueError(
                    "Response format must be a dictionary or 'json'"
                )

        messages = []

        if system is not None:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        options: dict[str, Any] = {
            "temperature": 0.0,
        }

        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        if response_format is not None:
            payload["format"] = response_format

        request_started = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "Ollama request timed out"
            ) from exc

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama"
            ) from exc

        request_elapsed = time.perf_counter() - request_started

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise OllamaResponseError(
                "Ollama response is missing a valid message object"
            )

        message = data.get("message")

        if not isinstance(message, dict):
            raise OllamaResponseError(
                "Ollama response is missing a valid message object"
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError(
                "Ollama response is missing valid content"
            )

        def duration_seconds(field: str) -> float | None:
            value = data.get(field)

            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                return value / 1_000_000_000

            return None

        logger.info(
            "Ollama performance | model: %s | request: %.3fs | "
            "total: %s | prompt evaluation: %s | generation: %s | "
            "prompt tokens: %s | generated tokens: %s | "
            "prompt length: %d characters | response length: %d characters",
            self.model,
            request_elapsed,
            duration_seconds("total_duration"),
            duration_seconds("prompt_eval_duration"),
            duration_seconds("eval_duration"),
            data.get("prompt_eval_count"),
            data.get("eval_count"),
            len(prompt) + (len(system) if system else 0),
            len(content),
        )

        return content.strip()

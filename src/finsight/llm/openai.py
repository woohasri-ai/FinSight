
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAIResponseError(RuntimeError):
    """Raised when the hosted LLM returns an invalid response."""


class OpenAIConnectionError(RuntimeError):
    """Raised when the hosted LLM cannot be reached."""


class OpenAITimeoutError(RuntimeError):
    """Raised when the hosted LLM request times out."""


class OpenAIClient:
    """Async client for the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 90.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Model name cannot be empty")
        if not base_url.strip():
            raise ValueError("Base URL cannot be empty")
        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero")

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError("OPENAI_API_KEY is not configured")

        self.model = model
        self.api_key = resolved_key.strip()
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

        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if num_predict is not None:
            payload["max_completion_tokens"] = num_predict

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, dict):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "finsight_generation",
                    "strict": True,
                    "schema": response_format,
                },
            }
        elif response_format is not None:
            raise ValueError(
                "Response format must be a dictionary or 'json'"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        request_started = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise OpenAITimeoutError(
                "Hosted LLM request timed out"
            ) from exc

        except httpx.ConnectError as exc:
            raise OpenAIConnectionError(
                "Could not connect to the hosted LLM"
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_message = "No additional error details provided."
            error_type = None
            error_code = None

            try:
                error_data = exc.response.json()
                error_object = (
                    error_data.get("error")
                    if isinstance(error_data, dict)
                    else None
                )

                if isinstance(error_object, dict):
                    error_message = str(
                        error_object.get("message") or error_message
                    )
                    error_type = error_object.get("type")
                    error_code = error_object.get("code")

            except ValueError:
                error_message = "The API returned a non-JSON error."

            logger.error(
                "Hosted LLM HTTP error | status: %s | "
                "type: %s | code: %s | message: %s",
                status_code,
                error_type,
                error_code,
                error_message,
            )

            raise OpenAIResponseError(
                f"Hosted LLM request failed with HTTP {status_code}: "
                f"{error_message}"
            ) from exc

        request_elapsed = time.perf_counter() - request_started

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise OpenAIResponseError(
                "Hosted LLM returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise OpenAIResponseError(
                "Hosted LLM returned an invalid response object"
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAIResponseError(
                "Hosted LLM response contains no choices"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAIResponseError(
                "Hosted LLM returned an invalid choice object"
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise OpenAIResponseError(
                "Hosted LLM response is missing a message"
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAIResponseError(
                "Hosted LLM response contains no text content"
            )

        usage = data.get("usage", {})
        prompt_tokens = (
            usage.get("prompt_tokens")
            if isinstance(usage, dict)
            else None
        )
        completion_tokens = (
            usage.get("completion_tokens")
            if isinstance(usage, dict)
            else None
        )

        logger.info(
            "Hosted LLM performance | model: %s | request: %.3fs | "
            "prompt tokens: %s | completion tokens: %s | "
            "prompt length: %d | response length: %d",
            self.model,
            request_elapsed,
            prompt_tokens,
            completion_tokens,
            len(prompt) + (len(system) if system else 0),
            len(content),
        )

        return content.strip()

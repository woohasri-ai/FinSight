from typing import Any, Protocol


class LLMClient(Protocol):
    """Common interface for language model providers."""

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Generate a text response from the language model."""
        ...

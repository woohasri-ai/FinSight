
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Citation(BaseModel):
    """A traceable source supporting a generated answer."""

    model_config = ConfigDict(extra="forbid")

    document_name: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)

    @field_validator("document_name", "chunk_id", "evidence_text")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be blank.")
        return value


class GeneratedAnswer(BaseModel):
    """Structured answer returned by the FinSight generation pipeline."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool


class EvidenceExcerpt(BaseModel):
    """An evidence reference selected from retrieved source material."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)

    @field_validator("evidence_id")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence ID cannot be blank.")
        return value


class LLMGenerationResponse(BaseModel):
    """Validate the generated answer and its supporting evidence references."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    excerpts: list[EvidenceExcerpt] = Field(default_factory=list)
    abstained: bool

    @field_validator("answer")
    @classmethod
    def validate_non_blank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer cannot be blank.")
        return value

    @field_validator("excerpts")
    @classmethod
    def validate_unique_excerpts(
        cls,
        value: list[EvidenceExcerpt],
    ) -> list[EvidenceExcerpt]:
        seen = set()

        for excerpt in value:
            if excerpt.evidence_id in seen:
                raise ValueError("Duplicate evidence references are not allowed.")

            seen.add(excerpt.evidence_id)

        return value

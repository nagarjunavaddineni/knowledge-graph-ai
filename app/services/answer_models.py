"""Validated models for grounded GraphRAG answers."""

from pydantic import BaseModel, ConfigDict, Field


class AnswerModel(BaseModel):
    """Base model for generated-answer data."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class AnswerSource(AnswerModel):
    """A source chunk supporting a generated answer."""

    source_id: str = Field(
        min_length=1,
        description="Display identifier such as S1.",
    )
    chunk_id: str = Field(min_length=1)
    document_id: str | None = None
    document_title: str | None = None
    chunk_index: int = Field(default=0, ge=0)
    excerpt: str = Field(min_length=1)
    retrieval_score: float = Field(gt=0.0)


class GroundedAnswer(AnswerModel):
    """Final grounded response returned to the user."""

    question: str = Field(min_length=1)
    answerable: bool
    answer: str = Field(min_length=1)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    insufficient_evidence_reason: str | None = None

    sources: list[AnswerSource] = Field(
        default_factory=list
    )

    graph_facts: list[str] = Field(
        default_factory=list
    )
"""Validated models for expanded graph retrieval context."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import GraphConnection


class ContextModel(BaseModel):
    """Base model for graph-context results."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ContextEntity(ContextModel):
    """An entity mentioned in a retrieved chunk."""

    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    description: str | None = None

    connections: list[GraphConnection] = Field(
        default_factory=list
    )


class ExpandedChunkContext(ContextModel):
    """A retrieved chunk expanded through graph relationships."""

    chunk_id: str = Field(min_length=1)
    document_id: str | None = None
    document_title: str | None = None
    chunk_index: int = Field(default=0, ge=0)
    text: str = Field(min_length=1)

    rrf_score: float = Field(gt=0.0)

    fulltext_score: float | None = Field(
        default=None,
        ge=0.0,
    )
    vector_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    fulltext_rank: int | None = Field(
        default=None,
        ge=1,
    )
    vector_rank: int | None = Field(
        default=None,
        ge=1,
    )

    matched_by: list[
        Literal["fulltext", "vector"]
    ] = Field(
        min_length=1
    )

    entities: list[ContextEntity] = Field(
        default_factory=list
    )


class GraphContextResult(ContextModel):
    """Complete context produced by graph expansion."""

    question: str = Field(min_length=1)
    retrieved_chunk_count: int = Field(ge=0)
    expanded_chunk_count: int = Field(ge=0)

    chunks: list[ExpandedChunkContext] = Field(
        default_factory=list
    )
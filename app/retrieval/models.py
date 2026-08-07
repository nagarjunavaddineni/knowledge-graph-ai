"""Validated models returned by GraphRAG retrievers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalModel(BaseModel):
    """Base model for retrieval results."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class EntityReference(RetrievalModel):
    """An entity mentioned by a retrieved document chunk."""

    entity_id: str
    name: str
    entity_type: str


class SourceChunkReference(RetrievalModel):
    """A source chunk connected to a retrieved entity."""

    chunk_id: str
    document_id: str | None = None
    document_title: str | None = None
    text: str


class GraphConnection(RetrievalModel):
    """A graph relationship connected to an entity."""

    relationship_type: str
    direction: str
    neighbor_id: str
    neighbor_name: str
    neighbor_type: str


class ChunkSearchResult(RetrievalModel):
    """A document chunk returned by one retrieval method."""

    chunk_id: str
    document_id: str | None = None
    document_title: str | None = None
    chunk_index: int = Field(default=0, ge=0)
    text: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    entities: list[EntityReference] = Field(
        default_factory=list
    )


class EntitySearchResult(RetrievalModel):
    """An entity returned by full-text retrieval."""

    entity_id: str
    name: str
    entity_type: str
    description: str | None = None
    score: float = Field(ge=0.0)
    source_chunks: list[SourceChunkReference] = Field(
        default_factory=list
    )
    connections: list[GraphConnection] = Field(
        default_factory=list
    )


class FullTextSearchResult(RetrievalModel):
    """Combined chunk and entity full-text search results."""

    question: str
    lucene_query: str
    chunks: list[ChunkSearchResult] = Field(
        default_factory=list
    )
    entities: list[EntitySearchResult] = Field(
        default_factory=list
    )


class HybridChunkSearchResult(RetrievalModel):
    """A chunk ranked using full-text and vector retrieval."""

    chunk_id: str
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

    entities: list[EntityReference] = Field(
        default_factory=list
    )


class HybridSearchResult(RetrievalModel):
    """Complete result from hybrid rank-fusion retrieval."""

    question: str = Field(min_length=1)
    rrf_constant: int = Field(gt=0)
    candidate_limit: int = Field(gt=0)
    fulltext_candidate_count: int = Field(ge=0)
    vector_candidate_count: int = Field(ge=0)

    results: list[HybridChunkSearchResult] = Field(
        default_factory=list
    )
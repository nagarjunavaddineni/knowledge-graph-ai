"""Retrieve chunks and entities from Neo4j full-text indexes."""

import logging
import re
from typing import Any, Final

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.retrieval.models import (
    ChunkSearchResult,
    EntityReference,
    EntitySearchResult,
    FullTextSearchResult,
    GraphConnection,
    SourceChunkReference,
)


logger = logging.getLogger(__name__)


class FullTextRetrievalError(RuntimeError):
    """Raised when full-text retrieval cannot be completed."""


QUESTION_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    }
)


ENTITY_LABELS: Final[tuple[str, ...]] = (
    "Person",
    "Project",
    "Customer",
    "Team",
    "Technology",
    "Document",
)


class FullTextRetriever:
    """Search Neo4j chunks and entities using full-text indexes."""

    def __init__(
        self,
        client: Neo4jClient,
        chunk_index_name: str,
        entity_index_name: str,
        default_top_k: int = 5,
    ) -> None:
        """Initialize the full-text retriever."""
        if not chunk_index_name:
            raise ValueError(
                "chunk_index_name cannot be empty."
            )

        if not entity_index_name:
            raise ValueError(
                "entity_index_name cannot be empty."
            )

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than zero."
            )

        self.client = client
        self.chunk_index_name = chunk_index_name
        self.entity_index_name = entity_index_name
        self.default_top_k = default_top_k

    @classmethod
    def from_settings(
        cls,
        client: Neo4jClient,
    ) -> "FullTextRetriever":
        """Create a retriever using application settings."""
        return cls(
            client=client,
            chunk_index_name=settings.chunk_fulltext_index,
            entity_index_name=settings.entity_fulltext_index,
            default_top_k=settings.retrieval_top_k,
        )

    def search(
        self,
        question: str,
        limit: int | None = None,
    ) -> FullTextSearchResult:
        """Search both chunks and graph entities."""
        normalized_question = self._validate_question(
            question
        )
        effective_limit = self._resolve_limit(limit)
        lucene_query = self._build_lucene_query(
            normalized_question
        )

        chunk_results = self._search_chunks_with_query(
            lucene_query=lucene_query,
            limit=effective_limit,
        )

        entity_results = self._search_entities_with_query(
            lucene_query=lucene_query,
            limit=effective_limit,
        )

        logger.info(
            "Full-text retrieval returned %s chunks and "
            "%s entities for question '%s'.",
            len(chunk_results),
            len(entity_results),
            normalized_question,
        )

        return FullTextSearchResult(
            question=normalized_question,
            lucene_query=lucene_query,
            chunks=chunk_results,
            entities=entity_results,
        )

    def search_chunks(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Search document chunks by keyword relevance."""
        normalized_question = self._validate_question(
            question
        )
        effective_limit = self._resolve_limit(limit)
        lucene_query = self._build_lucene_query(
            normalized_question
        )

        return self._search_chunks_with_query(
            lucene_query=lucene_query,
            limit=effective_limit,
        )

    def search_entities(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[EntitySearchResult]:
        """Search graph entities by keyword relevance."""
        normalized_question = self._validate_question(
            question
        )
        effective_limit = self._resolve_limit(limit)
        lucene_query = self._build_lucene_query(
            normalized_question
        )

        return self._search_entities_with_query(
            lucene_query=lucene_query,
            limit=effective_limit,
        )

    def _search_chunks_with_query(
        self,
        lucene_query: str,
        limit: int,
    ) -> list[ChunkSearchResult]:
        """Execute full-text search against Chunk nodes."""
        query = """
        CALL db.index.fulltext.queryNodes(
            $index_name,
            $lucene_query
        )
        YIELD node, score

        WITH node AS chunk, score
        WHERE chunk:Chunk

        OPTIONAL MATCH
            (document:Document)-[:HAS_CHUNK]->(chunk)

        OPTIONAL MATCH
            (chunk)-[:MENTIONS]->(entity)

        WITH
            chunk,
            score,
            document,
            collect(
                DISTINCT CASE
                    WHEN entity IS NULL
                    THEN NULL
                    ELSE {
                        entity_id: entity.id,
                        name: coalesce(
                            entity.name,
                            entity.title
                        ),
                        entity_type: head(labels(entity))
                    }
                END
            ) AS mentioned_entities

        RETURN
            chunk.id AS chunk_id,
            chunk.document_id AS document_id,
            document.title AS document_title,
            coalesce(
                chunk.chunk_index,
                0
            ) AS chunk_index,
            chunk.text AS text,
            score,
            mentioned_entities AS entities

        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            records = self.client.execute_query(
                query,
                {
                    "index_name": self.chunk_index_name,
                    "lucene_query": lucene_query,
                    "limit": limit,
                },
            )
        except Exception as error:
            logger.exception(
                "Chunk full-text retrieval failed."
            )

            raise FullTextRetrievalError(
                "Unable to search the chunk full-text index."
            ) from error

        results: list[ChunkSearchResult] = []

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("text")

            if not chunk_id or not text:
                continue

            entities = self._parse_entity_references(
                record.get("entities")
            )

            results.append(
                ChunkSearchResult(
                    chunk_id=str(chunk_id),
                    document_id=self._optional_string(
                        record.get("document_id")
                    ),
                    document_title=self._optional_string(
                        record.get("document_title")
                    ),
                    chunk_index=int(
                        record.get("chunk_index") or 0
                    ),
                    text=str(text),
                    score=float(
                        record.get("score") or 0.0
                    ),
                    entities=entities,
                )
            )

        return results

    def _search_entities_with_query(
        self,
        lucene_query: str,
        limit: int,
    ) -> list[EntitySearchResult]:
        """Execute full-text search against graph entities."""
        query = """
        CALL db.index.fulltext.queryNodes(
            $index_name,
            $lucene_query
        )
        YIELD node, score

        WITH node AS entity, score
        WHERE any(
            label IN labels(entity)
            WHERE label IN $entity_labels
        )

        OPTIONAL MATCH
            (chunk:Chunk)-[:MENTIONS]->(entity)

        OPTIONAL MATCH
            (document:Document)-[:HAS_CHUNK]->(chunk)

        WITH
            entity,
            score,
            collect(
                DISTINCT CASE
                    WHEN chunk IS NULL
                    THEN NULL
                    ELSE {
                        chunk_id: chunk.id,
                        document_id: document.id,
                        document_title: document.title,
                        text: chunk.text
                    }
                END
            )[0..$context_limit] AS source_chunks

        OPTIONAL MATCH
            (entity)-[relationship]-(neighbor)

        WHERE NOT type(relationship) IN [
            "MENTIONS",
            "HAS_CHUNK"
        ]

        WITH
            entity,
            score,
            source_chunks,
            collect(
                DISTINCT CASE
                    WHEN neighbor IS NULL
                    THEN NULL
                    ELSE {
                        relationship_type:
                            type(relationship),
                        direction:
                            CASE
                                WHEN startNode(
                                    relationship
                                ) = entity
                                THEN "outgoing"
                                ELSE "incoming"
                            END,
                        neighbor_id: neighbor.id,
                        neighbor_name: coalesce(
                            neighbor.name,
                            neighbor.title
                        ),
                        neighbor_type:
                            head(labels(neighbor))
                    }
                END
            )[0..$connection_limit] AS connections

        RETURN
            entity.id AS entity_id,
            coalesce(
                entity.name,
                entity.title
            ) AS name,
            head(labels(entity)) AS entity_type,
            coalesce(
                entity.description,
                entity.role,
                entity.industry,
                entity.department,
                entity.category,
                entity.content
            ) AS description,
            score,
            source_chunks,
            connections

        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            records = self.client.execute_query(
                query,
                {
                    "index_name": self.entity_index_name,
                    "lucene_query": lucene_query,
                    "entity_labels": list(ENTITY_LABELS),
                    "context_limit": 3,
                    "connection_limit": 10,
                    "limit": limit,
                },
            )
        except Exception as error:
            logger.exception(
                "Entity full-text retrieval failed."
            )

            raise FullTextRetrievalError(
                "Unable to search the entity full-text index."
            ) from error

        results: list[EntitySearchResult] = []

        for record in records:
            entity_id = record.get("entity_id")
            name = record.get("name")
            entity_type = record.get("entity_type")

            if not entity_id or not name or not entity_type:
                continue

            results.append(
                EntitySearchResult(
                    entity_id=str(entity_id),
                    name=str(name),
                    entity_type=str(entity_type),
                    description=self._optional_string(
                        record.get("description")
                    ),
                    score=float(
                        record.get("score") or 0.0
                    ),
                    source_chunks=self._parse_source_chunks(
                        record.get("source_chunks")
                    ),
                    connections=self._parse_connections(
                        record.get("connections")
                    ),
                )
            )

        return results

    @staticmethod
    def _validate_question(question: str) -> str:
        """Validate and normalize a user question."""
        if not isinstance(question, str):
            raise TypeError(
                "The retrieval question must be a string."
            )

        normalized_question = " ".join(
            question.split()
        )

        if not normalized_question:
            raise ValueError(
                "The retrieval question cannot be empty."
            )

        return normalized_question

    def _resolve_limit(
        self,
        limit: int | None,
    ) -> int:
        """Resolve and validate the retrieval limit."""
        effective_limit = (
            self.default_top_k
            if limit is None
            else limit
        )

        if effective_limit <= 0:
            raise ValueError(
                "The retrieval limit must be greater than zero."
            )

        if effective_limit > 50:
            raise ValueError(
                "The retrieval limit cannot exceed 50."
            )

        return effective_limit

    @staticmethod
    def _build_lucene_query(question: str) -> str:
        """Convert a natural-language question into a safe query."""
        raw_tokens = re.findall(
            r"[A-Za-z0-9]+",
            question,
        )

        meaningful_tokens: list[str] = []
        seen_tokens: set[str] = set()

        for raw_token in raw_tokens:
            token = raw_token.casefold()

            if token in QUESTION_STOP_WORDS:
                continue

            if len(token) < 2:
                continue

            if token in seen_tokens:
                continue

            meaningful_tokens.append(token)
            seen_tokens.add(token)

        if not meaningful_tokens:
            meaningful_tokens = [
                token.casefold()
                for token in raw_tokens
                if len(token) >= 2
            ]

        if not meaningful_tokens:
            raise ValueError(
                "The question contains no searchable terms."
            )

        return " OR ".join(
            f'"{token}"'
            for token in meaningful_tokens
        )

    @staticmethod
    def _parse_entity_references(
        raw_entities: Any,
    ) -> list[EntityReference]:
        """Convert Neo4j entity maps into validated models."""
        if not isinstance(raw_entities, list):
            return []

        entities: list[EntityReference] = []

        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue

            entity_id = raw_entity.get("entity_id")
            name = raw_entity.get("name")
            entity_type = raw_entity.get("entity_type")

            if not entity_id or not name or not entity_type:
                continue

            entities.append(
                EntityReference(
                    entity_id=str(entity_id),
                    name=str(name),
                    entity_type=str(entity_type),
                )
            )

        return entities

    @staticmethod
    def _parse_source_chunks(
        raw_chunks: Any,
    ) -> list[SourceChunkReference]:
        """Convert Neo4j chunk maps into validated models."""
        if not isinstance(raw_chunks, list):
            return []

        chunks: list[SourceChunkReference] = []

        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                continue

            chunk_id = raw_chunk.get("chunk_id")
            text = raw_chunk.get("text")

            if not chunk_id or not text:
                continue

            chunks.append(
                SourceChunkReference(
                    chunk_id=str(chunk_id),
                    document_id=(
                        FullTextRetriever._optional_string(
                            raw_chunk.get("document_id")
                        )
                    ),
                    document_title=(
                        FullTextRetriever._optional_string(
                            raw_chunk.get("document_title")
                        )
                    ),
                    text=str(text),
                )
            )

        return chunks

    @staticmethod
    def _parse_connections(
        raw_connections: Any,
    ) -> list[GraphConnection]:
        """Convert Neo4j relationship maps into models."""
        if not isinstance(raw_connections, list):
            return []

        connections: list[GraphConnection] = []

        for raw_connection in raw_connections:
            if not isinstance(raw_connection, dict):
                continue

            relationship_type = raw_connection.get(
                "relationship_type"
            )
            direction = raw_connection.get("direction")
            neighbor_id = raw_connection.get("neighbor_id")
            neighbor_name = raw_connection.get(
                "neighbor_name"
            )
            neighbor_type = raw_connection.get(
                "neighbor_type"
            )

            if not all(
                (
                    relationship_type,
                    direction,
                    neighbor_id,
                    neighbor_name,
                    neighbor_type,
                )
            ):
                continue

            connections.append(
                GraphConnection(
                    relationship_type=str(
                        relationship_type
                    ),
                    direction=str(direction),
                    neighbor_id=str(neighbor_id),
                    neighbor_name=str(neighbor_name),
                    neighbor_type=str(neighbor_type),
                )
            )

        return connections

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        """Convert a value to a nonempty optional string."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        return normalized_value or None
"""Retrieve semantically similar chunks from the Neo4j vector index."""

import logging
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
    EmbeddingService,
)
from app.retrieval.models import (
    ChunkSearchResult,
    EntityReference,
)


logger = logging.getLogger(__name__)


class VectorRetrievalError(RuntimeError):
    """Raised when semantic vector retrieval fails."""


class SemanticVectorRetriever:
    """Search Neo4j chunks using semantic vector similarity."""

    def __init__(
        self,
        client: Neo4jClient,
        embedding_service: EmbeddingService,
        vector_index_name: str,
        default_top_k: int = 5,
    ) -> None:
        """Initialize the semantic vector retriever."""
        if not vector_index_name:
            raise ValueError(
                "vector_index_name cannot be empty."
            )

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than zero."
            )

        self.client = client
        self.embedding_service = embedding_service
        self.vector_index_name = vector_index_name
        self.default_top_k = default_top_k

    @classmethod
    def from_settings(
        cls,
        client: Neo4jClient,
    ) -> "SemanticVectorRetriever":
        """Create a retriever using application settings."""
        return cls(
            client=client,
            embedding_service=EmbeddingService.from_settings(),
            vector_index_name=settings.chunk_vector_index,
            default_top_k=settings.retrieval_top_k,
        )

    def search(
        self,
        question: str,
        limit: int | None = None,
        minimum_score: float = 0.0,
    ) -> list[ChunkSearchResult]:
        """Retrieve semantically relevant document chunks."""
        normalized_question = self._validate_question(
            question
        )

        effective_limit = self._resolve_limit(limit)

        self._validate_minimum_score(minimum_score)

        try:
            query_embedding = (
                self.embedding_service.embed_text(
                    normalized_question
                )
            )
        except EmbeddingGenerationError:
            raise
        except Exception as error:
            logger.exception(
                "Unexpected question-embedding failure."
            )

            raise VectorRetrievalError(
                "Unable to generate the question embedding."
            ) from error

        return self.search_by_embedding(
            query_embedding=query_embedding,
            limit=effective_limit,
            minimum_score=minimum_score,
        )

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int | None = None,
        minimum_score: float = 0.0,
    ) -> list[ChunkSearchResult]:
        """Retrieve chunks using a pre-generated embedding."""
        effective_limit = self._resolve_limit(limit)

        self._validate_minimum_score(minimum_score)

        if not query_embedding:
            raise ValueError(
                "query_embedding cannot be empty."
            )

        if (
            len(query_embedding)
            != self.embedding_service.dimensions
        ):
            raise ValueError(
                "Question embedding contains "
                f"{len(query_embedding)} dimensions; expected "
                f"{self.embedding_service.dimensions}."
            )

        numeric_embedding: list[float] = []

        for value in query_embedding:
            if not isinstance(value, (int, float)):
                raise TypeError(
                    "Every question-embedding value must "
                    "be numeric."
                )

            numeric_embedding.append(float(value))

        records = self._query_vector_index(
            query_embedding=numeric_embedding,
            limit=effective_limit,
            minimum_score=minimum_score,
        )

        results = self._parse_results(records)

        logger.info(
            "Vector retrieval returned %s chunks from '%s'.",
            len(results),
            self.vector_index_name,
        )

        return results

    def _query_vector_index(
        self,
        query_embedding: list[float],
        limit: int,
        minimum_score: float,
    ) -> list[dict[str, Any]]:
        """Query the Neo4j Chunk vector index."""
        query = """
        CALL db.index.vector.queryNodes(
            $index_name,
            $candidate_limit,
            $query_embedding
        )
        YIELD node, score

        WITH node AS chunk, score

        WHERE
            chunk:Chunk
            AND score >= $minimum_score

        OPTIONAL MATCH
            (document:Document)-[:HAS_CHUNK]->(chunk)

        OPTIONAL MATCH
            (chunk)-[:MENTIONS]->(entity)

        WITH
            chunk,
            score,
            document,
            [
                item IN collect(
                    DISTINCT CASE
                        WHEN entity IS NULL
                        THEN NULL
                        ELSE {
                            entity_id: entity.id,
                            name: coalesce(
                                entity.name,
                                entity.title
                            ),
                            entity_type:
                                head(labels(entity))
                        }
                    END
                )
                WHERE item IS NOT NULL
            ] AS mentioned_entities

        RETURN
            chunk.id AS chunk_id,
            coalesce(
                document.id,
                chunk.document_id
            ) AS document_id,
            document.title AS document_title,
            coalesce(
                chunk.chunk_index,
                0
            ) AS chunk_index,
            chunk.text AS text,
            score,
            mentioned_entities AS entities

        ORDER BY
            score DESC,
            chunk.id

        LIMIT $limit
        """

        candidate_limit = min(
            max(limit * 3, limit),
            100,
        )

        try:
            return self.client.execute_query(
                query,
                {
                    "index_name": self.vector_index_name,
                    "candidate_limit": candidate_limit,
                    "query_embedding": query_embedding,
                    "minimum_score": minimum_score,
                    "limit": limit,
                },
            )

        except Exception as error:
            logger.exception(
                "Neo4j vector-index retrieval failed."
            )

            raise VectorRetrievalError(
                "Unable to search the Neo4j chunk "
                "vector index."
            ) from error

    def _parse_results(
        self,
        records: list[dict[str, Any]],
    ) -> list[ChunkSearchResult]:
        """Convert Neo4j records into validated result models."""
        results: list[ChunkSearchResult] = []

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("text")

            if not chunk_id or not text:
                logger.warning(
                    "Skipped a vector-search record without "
                    "a chunk ID or text."
                )
                continue

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
                    entities=self._parse_entities(
                        record.get("entities")
                    ),
                )
            )

        return results

    @staticmethod
    def _parse_entities(
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

            if not all(
                (
                    entity_id,
                    name,
                    entity_type,
                )
            ):
                continue

            entities.append(
                EntityReference(
                    entity_id=str(entity_id),
                    name=str(name),
                    entity_type=str(entity_type),
                )
            )

        return entities

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
                "The retrieval limit must be greater "
                "than zero."
            )

        if effective_limit > 50:
            raise ValueError(
                "The retrieval limit cannot exceed 50."
            )

        return effective_limit

    @staticmethod
    def _validate_question(question: str) -> str:
        """Validate and normalize the search question."""
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

    @staticmethod
    def _validate_minimum_score(
        minimum_score: float,
    ) -> None:
        """Validate the vector similarity threshold."""
        if not isinstance(
            minimum_score,
            (int, float),
        ):
            raise TypeError(
                "minimum_score must be numeric."
            )

        if not 0.0 <= float(minimum_score) <= 1.0:
            raise ValueError(
                "minimum_score must be between 0.0 and 1.0."
            )

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """Convert a value into an optional string."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        return normalized_value or None
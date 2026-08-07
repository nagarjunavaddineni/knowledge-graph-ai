"""Expand hybrid retrieval results through Neo4j relationships."""

import logging
from typing import Any, Final

from app.database.neo4j_client import Neo4jClient
from app.retrieval.context_models import (
    ContextEntity,
    ExpandedChunkContext,
    GraphContextResult,
)
from app.retrieval.models import (
    GraphConnection,
    HybridChunkSearchResult,
    HybridSearchResult,
)


logger = logging.getLogger(__name__)


class GraphContextExpansionError(RuntimeError):
    """Raised when graph-context expansion fails."""


BUSINESS_RELATIONSHIPS: Final[tuple[str, ...]] = (
    "MANAGES",
    "WORKS_ON",
    "SERVES",
    "USES",
    "MEMBER_OF",
    "OWNS",
)


class GraphContextExpander:
    """Expand retrieved chunks using their mentioned entities."""

    def __init__(
        self,
        client: Neo4jClient,
        relationship_types: tuple[str, ...] = (
            BUSINESS_RELATIONSHIPS
        ),
    ) -> None:
        """Initialize the graph-context expander."""
        if not relationship_types:
            raise ValueError(
                "At least one relationship type is required."
            )

        normalized_relationships: list[str] = []

        for relationship_type in relationship_types:
            normalized = relationship_type.strip().upper()

            if not normalized:
                raise ValueError(
                    "Relationship types cannot be empty."
                )

            if not normalized.replace("_", "").isalnum():
                raise ValueError(
                    "Relationship types must contain only "
                    "letters, numbers, and underscores."
                )

            normalized_relationships.append(normalized)

        self.client = client
        self.relationship_types = tuple(
            dict.fromkeys(normalized_relationships)
        )

    def expand(
        self,
        retrieval_result: HybridSearchResult,
        max_connections_per_entity: int = 10,
    ) -> GraphContextResult:
        """Expand every hybrid result with connected graph facts."""
        if max_connections_per_entity <= 0:
            raise ValueError(
                "max_connections_per_entity must be "
                "greater than zero."
            )

        if max_connections_per_entity > 50:
            raise ValueError(
                "max_connections_per_entity cannot exceed 50."
            )

        if not retrieval_result.results:
            return GraphContextResult(
                question=retrieval_result.question,
                retrieved_chunk_count=0,
                expanded_chunk_count=0,
                chunks=[],
            )

        chunk_ids = [
            result.chunk_id
            for result in retrieval_result.results
        ]

        records = self._query_graph_context(
            chunk_ids=chunk_ids,
            max_connections_per_entity=(
                max_connections_per_entity
            ),
        )

        records_by_chunk_id = {
            str(record["chunk_id"]): record
            for record in records
            if record.get("chunk_id")
        }

        expanded_chunks: list[ExpandedChunkContext] = []

        for retrieval_item in retrieval_result.results:
            record = records_by_chunk_id.get(
                retrieval_item.chunk_id
            )

            expanded_chunks.append(
                self._build_expanded_chunk(
                    retrieval_item=retrieval_item,
                    record=record,
                )
            )

        logger.info(
            "Expanded %s of %s retrieved chunks with "
            "Neo4j graph context.",
            sum(
                1
                for chunk in expanded_chunks
                if chunk.entities
            ),
            len(expanded_chunks),
        )

        return GraphContextResult(
            question=retrieval_result.question,
            retrieved_chunk_count=len(
                retrieval_result.results
            ),
            expanded_chunk_count=len(expanded_chunks),
            chunks=expanded_chunks,
        )

    def _query_graph_context(
        self,
        chunk_ids: list[str],
        max_connections_per_entity: int,
    ) -> list[dict[str, Any]]:
        """Retrieve mentioned entities and their connections."""
        query = """
        UNWIND $chunk_ids AS requested_chunk_id

        MATCH (chunk:Chunk {id: requested_chunk_id})

        OPTIONAL MATCH
            (document:Document)-[:HAS_CHUNK]->(chunk)

        OPTIONAL MATCH
            (chunk)-[:MENTIONS]->(entity)

        OPTIONAL MATCH
            (entity)-[relationship]-(neighbor)

        WHERE
            relationship IS NULL
            OR type(relationship) IN $relationship_types

        WITH
            chunk,
            document,
            entity,
            [
                connection IN collect(
                    DISTINCT CASE
                        WHEN relationship IS NULL
                            OR neighbor IS NULL
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
                )
                WHERE connection IS NOT NULL
            ][0..$connection_limit] AS connections

        WITH
            chunk,
            document,
            [
                entity_context IN collect(
                    CASE
                        WHEN entity IS NULL
                        THEN NULL
                        ELSE {
                            entity_id: entity.id,
                            name: coalesce(
                                entity.name,
                                entity.title
                            ),
                            entity_type:
                                head(labels(entity)),
                            description: coalesce(
                                entity.description,
                                entity.role,
                                entity.industry,
                                entity.department,
                                entity.category
                            ),
                            connections: connections
                        }
                    END
                )
                WHERE entity_context IS NOT NULL
            ] AS entities

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
            entities

        ORDER BY
            chunk.document_id,
            chunk.chunk_index,
            chunk.id
        """

        try:
            return self.client.execute_query(
                query,
                {
                    "chunk_ids": chunk_ids,
                    "relationship_types": list(
                        self.relationship_types
                    ),
                    "connection_limit": (
                        max_connections_per_entity
                    ),
                },
            )

        except Exception as error:
            logger.exception(
                "Neo4j graph-context expansion failed."
            )

            raise GraphContextExpansionError(
                "Unable to expand retrieval results "
                "through the knowledge graph."
            ) from error

    def _build_expanded_chunk(
        self,
        retrieval_item: HybridChunkSearchResult,
        record: dict[str, Any] | None,
    ) -> ExpandedChunkContext:
        """Combine retrieval ranking with graph context."""
        if record is None:
            return ExpandedChunkContext(
                chunk_id=retrieval_item.chunk_id,
                document_id=retrieval_item.document_id,
                document_title=(
                    retrieval_item.document_title
                ),
                chunk_index=retrieval_item.chunk_index,
                text=retrieval_item.text,
                rrf_score=retrieval_item.rrf_score,
                fulltext_score=(
                    retrieval_item.fulltext_score
                ),
                vector_score=retrieval_item.vector_score,
                fulltext_rank=(
                    retrieval_item.fulltext_rank
                ),
                vector_rank=retrieval_item.vector_rank,
                matched_by=retrieval_item.matched_by,
                entities=[],
            )

        return ExpandedChunkContext(
            chunk_id=retrieval_item.chunk_id,
            document_id=(
                self._optional_string(
                    record.get("document_id")
                )
                or retrieval_item.document_id
            ),
            document_title=(
                self._optional_string(
                    record.get("document_title")
                )
                or retrieval_item.document_title
            ),
            chunk_index=int(
                record.get("chunk_index")
                if record.get("chunk_index") is not None
                else retrieval_item.chunk_index
            ),
            text=(
                self._optional_string(record.get("text"))
                or retrieval_item.text
            ),
            rrf_score=retrieval_item.rrf_score,
            fulltext_score=retrieval_item.fulltext_score,
            vector_score=retrieval_item.vector_score,
            fulltext_rank=retrieval_item.fulltext_rank,
            vector_rank=retrieval_item.vector_rank,
            matched_by=retrieval_item.matched_by,
            entities=self._parse_entities(
                record.get("entities")
            ),
        )

    def _parse_entities(
        self,
        raw_entities: Any,
    ) -> list[ContextEntity]:
        """Convert Neo4j entity maps into validated models."""
        if not isinstance(raw_entities, list):
            return []

        entities: list[ContextEntity] = []
        seen_entity_ids: set[str] = set()

        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue

            entity_id = self._optional_string(
                raw_entity.get("entity_id")
            )
            name = self._optional_string(
                raw_entity.get("name")
            )
            entity_type = self._optional_string(
                raw_entity.get("entity_type")
            )

            if not entity_id or not name or not entity_type:
                continue

            if entity_id in seen_entity_ids:
                continue

            entities.append(
                ContextEntity(
                    entity_id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    description=self._optional_string(
                        raw_entity.get("description")
                    ),
                    connections=self._parse_connections(
                        raw_entity.get("connections")
                    ),
                )
            )

            seen_entity_ids.add(entity_id)

        return entities

    @staticmethod
    def _parse_connections(
        raw_connections: Any,
    ) -> list[GraphConnection]:
        """Convert Neo4j relationship maps into models."""
        if not isinstance(raw_connections, list):
            return []

        connections: list[GraphConnection] = []
        seen_connections: set[
            tuple[str, str, str]
        ] = set()

        for raw_connection in raw_connections:
            if not isinstance(raw_connection, dict):
                continue

            relationship_type = (
                GraphContextExpander._optional_string(
                    raw_connection.get(
                        "relationship_type"
                    )
                )
            )
            direction = (
                GraphContextExpander._optional_string(
                    raw_connection.get("direction")
                )
            )
            neighbor_id = (
                GraphContextExpander._optional_string(
                    raw_connection.get("neighbor_id")
                )
            )
            neighbor_name = (
                GraphContextExpander._optional_string(
                    raw_connection.get("neighbor_name")
                )
            )
            neighbor_type = (
                GraphContextExpander._optional_string(
                    raw_connection.get("neighbor_type")
                )
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

            connection_key = (
                relationship_type,
                direction,
                neighbor_id,
            )

            if connection_key in seen_connections:
                continue

            connections.append(
                GraphConnection(
                    relationship_type=(
                        relationship_type
                    ),
                    direction=direction,
                    neighbor_id=neighbor_id,
                    neighbor_name=neighbor_name,
                    neighbor_type=neighbor_type,
                )
            )

            seen_connections.add(connection_key)

        return connections

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """Convert a value into a nonempty optional string."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        return normalized_value or None
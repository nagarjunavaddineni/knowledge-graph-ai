"""Load dashboard statistics from the Neo4j knowledge graph."""

import logging
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from app.database.neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


class DashboardServiceError(RuntimeError):
    """Raised when dashboard statistics cannot be loaded."""


class DashboardModel(BaseModel):
    """Base model for dashboard results."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class DashboardTotals(DashboardModel):
    """Top-level knowledge-graph totals."""

    total_nodes: int = Field(default=0, ge=0)
    total_relationships: int = Field(default=0, ge=0)
    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    embedded_chunks: int = Field(default=0, ge=0)
    entities: int = Field(default=0, ge=0)
    business_relationships: int = Field(
        default=0,
        ge=0,
    )

    @property
    def chunks_without_embeddings(self) -> int:
        """Return the number of chunks without embeddings."""
        return max(
            self.chunks - self.embedded_chunks,
            0,
        )

    @property
    def embedding_coverage(self) -> float:
        """Return embedding coverage between zero and one."""
        if self.chunks == 0:
            return 0.0

        return min(
            max(
                self.embedded_chunks / self.chunks,
                0.0,
            ),
            1.0,
        )


class DistributionItem(DashboardModel):
    """One grouped dashboard count."""

    name: str = Field(min_length=1)
    count: int = Field(ge=0)


class RecentDocument(DashboardModel):
    """A recently ingested Neo4j document."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    filename: str | None = None
    document_type: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    embedded_chunk_count: int = Field(
        default=0,
        ge=0,
    )
    updated_at: str | None = None

    @property
    def embedding_coverage(self) -> float:
        """Return this document's chunk embedding coverage."""
        if self.chunk_count == 0:
            return 0.0

        return min(
            max(
                (
                    self.embedded_chunk_count
                    / self.chunk_count
                ),
                0.0,
            ),
            1.0,
        )


class DashboardResult(DashboardModel):
    """Complete data required by the dashboard page."""

    totals: DashboardTotals
    entity_distribution: list[DistributionItem] = Field(
        default_factory=list
    )
    relationship_distribution: list[
        DistributionItem
    ] = Field(
        default_factory=list
    )
    recent_documents: list[RecentDocument] = Field(
        default_factory=list
    )


ENTITY_LABELS: Final[tuple[str, ...]] = (
    "Person",
    "Project",
    "Customer",
    "Team",
    "Technology",
)


BUSINESS_RELATIONSHIP_TYPES: Final[
    tuple[str, ...]
] = (
    "MANAGES",
    "WORKS_ON",
    "SERVES",
    "USES",
    "MEMBER_OF",
    "OWNS",
)


class DashboardService:
    """Read operational and graph statistics from Neo4j."""

    def __init__(
        self,
        client: Neo4jClient,
    ) -> None:
        """Initialize the dashboard service."""
        self.client = client

    def load_dashboard(
        self,
        recent_document_limit: int = 10,
    ) -> DashboardResult:
        """Load all dashboard data."""
        if recent_document_limit <= 0:
            raise ValueError(
                "recent_document_limit must be greater "
                "than zero."
            )

        if recent_document_limit > 100:
            raise ValueError(
                "recent_document_limit cannot exceed 100."
            )

        try:
            totals = self._load_totals()

            entity_distribution = (
                self._load_entity_distribution()
            )

            relationship_distribution = (
                self._load_relationship_distribution()
            )

            recent_documents = (
                self._load_recent_documents(
                    limit=recent_document_limit
                )
            )

        except DashboardServiceError:
            raise

        except Exception as error:
            logger.exception(
                "Unable to load Neo4j dashboard data."
            )

            raise DashboardServiceError(
                "Unable to load knowledge-graph "
                "dashboard statistics."
            ) from error

        return DashboardResult(
            totals=totals,
            entity_distribution=entity_distribution,
            relationship_distribution=(
                relationship_distribution
            ),
            recent_documents=recent_documents,
        )

    def _load_totals(self) -> DashboardTotals:
        """Load top-level graph totals."""
        query = """
        CALL {
            MATCH (node)
            RETURN count(node) AS total_nodes
        }

        CALL {
            MATCH ()-[relationship]->()
            RETURN count(relationship)
                AS total_relationships
        }

        CALL {
            MATCH (document:Document)
            RETURN count(document) AS documents
        }

        CALL {
            MATCH (chunk:Chunk)
            RETURN
                count(chunk) AS chunks,
                count(chunk.embedding)
                    AS embedded_chunks
        }

        CALL {
            MATCH (entity)
            WHERE any(
                label IN labels(entity)
                WHERE label IN $entity_labels
            )
            RETURN count(entity) AS entities
        }

        CALL {
            MATCH ()-[relationship]->()
            WHERE type(relationship)
                IN $business_relationship_types
            RETURN count(relationship)
                AS business_relationships
        }

        RETURN
            total_nodes,
            total_relationships,
            documents,
            chunks,
            embedded_chunks,
            entities,
            business_relationships
        """

        records = self._execute_query(
            query,
            {
                "entity_labels": list(
                    ENTITY_LABELS
                ),
                "business_relationship_types": list(
                    BUSINESS_RELATIONSHIP_TYPES
                ),
            },
            operation_name="graph totals",
        )

        if not records:
            return DashboardTotals()

        record = records[0]

        return DashboardTotals(
            total_nodes=self._read_integer(
                record,
                "total_nodes",
            ),
            total_relationships=self._read_integer(
                record,
                "total_relationships",
            ),
            documents=self._read_integer(
                record,
                "documents",
            ),
            chunks=self._read_integer(
                record,
                "chunks",
            ),
            embedded_chunks=self._read_integer(
                record,
                "embedded_chunks",
            ),
            entities=self._read_integer(
                record,
                "entities",
            ),
            business_relationships=self._read_integer(
                record,
                "business_relationships",
            ),
        )

    def _load_entity_distribution(
        self,
    ) -> list[DistributionItem]:
        """Load counts for business entity labels."""
        query = """
        MATCH (entity)

        UNWIND labels(entity) AS entity_type

        WITH entity_type
        WHERE entity_type IN $entity_labels

        RETURN
            entity_type AS name,
            count(*) AS count

        ORDER BY
            count DESC,
            name
        """

        records = self._execute_query(
            query,
            {
                "entity_labels": list(
                    ENTITY_LABELS
                )
            },
            operation_name="entity distribution",
        )

        return self._parse_distribution(records)

    def _load_relationship_distribution(
        self,
    ) -> list[DistributionItem]:
        """Load counts grouped by relationship type."""
        query = """
        MATCH ()-[relationship]->()

        RETURN
            type(relationship) AS name,
            count(*) AS count

        ORDER BY
            count DESC,
            name
        """

        records = self._execute_query(
            query,
            operation_name=(
                "relationship distribution"
            ),
        )

        return self._parse_distribution(records)

    def _load_recent_documents(
        self,
        limit: int,
    ) -> list[RecentDocument]:
        """Load recent documents and embedding coverage."""
        query = """
        MATCH (document:Document)

        OPTIONAL MATCH
            (document)-[:HAS_CHUNK]->(chunk:Chunk)

        WITH
            document,
            count(DISTINCT chunk) AS chunk_count,
            count(
                DISTINCT CASE
                    WHEN chunk.embedding IS NOT NULL
                    THEN chunk
                    ELSE NULL
                END
            ) AS embedded_chunk_count

        RETURN
            document.id AS document_id,
            coalesce(
                document.title,
                document.filename,
                document.name,
                document.id
            ) AS title,
            document.filename AS filename,
            coalesce(
                document.document_type,
                document.file_type,
                document.type
            ) AS document_type,
            chunk_count,
            embedded_chunk_count,
            coalesce(
                toString(document.updated_at),
                toString(document.created_at),
                toString(document.ingested_at)
            ) AS updated_at

        ORDER BY
            updated_at DESC,
            title

        LIMIT $limit
        """

        records = self._execute_query(
            query,
            {
                "limit": limit,
            },
            operation_name="recent documents",
        )

        documents: list[RecentDocument] = []

        for record in records:
            document_id = self._optional_string(
                record.get("document_id")
            )

            title = self._optional_string(
                record.get("title")
            )

            if not document_id or not title:
                continue

            documents.append(
                RecentDocument(
                    document_id=document_id,
                    title=title,
                    filename=self._optional_string(
                        record.get("filename")
                    ),
                    document_type=self._optional_string(
                        record.get("document_type")
                    ),
                    chunk_count=self._read_integer(
                        record,
                        "chunk_count",
                    ),
                    embedded_chunk_count=(
                        self._read_integer(
                            record,
                            "embedded_chunk_count",
                        )
                    ),
                    updated_at=self._optional_string(
                        record.get("updated_at")
                    ),
                )
            )

        return documents

    def _execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        operation_name: str = "dashboard query",
    ) -> list[dict[str, Any]]:
        """Execute a dashboard query with clear errors."""
        try:
            return self.client.execute_query(
                query,
                parameters,
            )

        except Exception as error:
            logger.exception(
                "Neo4j dashboard operation failed: %s.",
                operation_name,
            )

            raise DashboardServiceError(
                f"Unable to load {operation_name}."
            ) from error

    @staticmethod
    def _parse_distribution(
        records: list[dict[str, Any]],
    ) -> list[DistributionItem]:
        """Convert Neo4j grouped counts into models."""
        distribution: list[DistributionItem] = []

        for record in records:
            name = DashboardService._optional_string(
                record.get("name")
            )

            if not name:
                continue

            distribution.append(
                DistributionItem(
                    name=name,
                    count=DashboardService._read_integer(
                        record,
                        "count",
                    ),
                )
            )

        return distribution

    @staticmethod
    def _read_integer(
        record: dict[str, Any],
        key: str,
    ) -> int:
        """Read a nonnegative integer value."""
        value = record.get(key, 0)

        try:
            return max(
                int(value or 0),
                0,
            )

        except (TypeError, ValueError) as error:
            raise DashboardServiceError(
                f"Invalid numeric dashboard value: {key}."
            ) from error

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """Convert a value into a nonempty optional string."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        return normalized_value or None
"""Manage Neo4j indexes used by the GraphRAG retrieval pipeline."""

import logging
import time
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


class RetrievalIndexError(RuntimeError):
    """Raised when retrieval indexes cannot be created or verified."""


class RetrievalIndexManager:
    """Create and inspect vector and full-text retrieval indexes."""

    def __init__(
        self,
        client: Neo4jClient,
        vector_index_name: str,
        chunk_fulltext_index_name: str,
        entity_fulltext_index_name: str,
        embedding_dimensions: int,
    ) -> None:
        if not vector_index_name:
            raise ValueError(
                "vector_index_name cannot be empty."
            )

        if not chunk_fulltext_index_name:
            raise ValueError(
                "chunk_fulltext_index_name cannot be empty."
            )

        if not entity_fulltext_index_name:
            raise ValueError(
                "entity_fulltext_index_name cannot be empty."
            )

        if embedding_dimensions <= 0:
            raise ValueError(
                "embedding_dimensions must be greater than zero."
            )

        index_names = {
            vector_index_name,
            chunk_fulltext_index_name,
            entity_fulltext_index_name,
        }

        if len(index_names) != 3:
            raise ValueError(
                "Every retrieval index must have a unique name."
            )

        self.client = client
        self.vector_index_name = vector_index_name
        self.chunk_fulltext_index_name = (
            chunk_fulltext_index_name
        )
        self.entity_fulltext_index_name = (
            entity_fulltext_index_name
        )
        self.embedding_dimensions = embedding_dimensions

    @classmethod
    def from_settings(
        cls,
        client: Neo4jClient,
    ) -> "RetrievalIndexManager":
        """Create an index manager using application settings."""
        return cls(
            client=client,
            vector_index_name=settings.chunk_vector_index,
            chunk_fulltext_index_name=(
                settings.chunk_fulltext_index
            ),
            entity_fulltext_index_name=(
                settings.entity_fulltext_index
            ),
            embedding_dimensions=(
                settings.embedding_dimensions
            ),
        )

    @property
    def index_names(self) -> set[str]:
        """Return the expected retrieval index names."""
        return {
            self.vector_index_name,
            self.chunk_fulltext_index_name,
            self.entity_fulltext_index_name,
        }

    def create_indexes(self) -> None:
        """Create all GraphRAG retrieval indexes."""
        self.create_chunk_vector_index()
        self.create_chunk_fulltext_index()
        self.create_entity_fulltext_index()

        logger.info(
            "Retrieval-index creation commands completed."
        )

    def create_chunk_vector_index(self) -> None:
        """Create the vector index for Chunk embeddings."""
        query = """
        CREATE VECTOR INDEX $index_name IF NOT EXISTS
        FOR (chunk:Chunk)
        ON chunk.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`:
                    toInteger($dimensions),
                `vector.similarity_function`:
                    'cosine'
            }
        }
        """

        self.client.execute_query(
            query,
            {
                "index_name": self.vector_index_name,
                "dimensions": self.embedding_dimensions,
            },
        )

        logger.info(
            "Vector-index command completed for '%s'.",
            self.vector_index_name,
        )

    def create_chunk_fulltext_index(self) -> None:
        """Create the full-text index for document chunks."""
        query = """
        CREATE FULLTEXT INDEX $index_name IF NOT EXISTS
        FOR (chunk:Chunk)
        ON EACH [chunk.text]
        """

        self.client.execute_query(
            query,
            {
                "index_name": (
                    self.chunk_fulltext_index_name
                )
            },
        )

        logger.info(
            "Chunk full-text index command completed for '%s'.",
            self.chunk_fulltext_index_name,
        )

    def create_entity_fulltext_index(self) -> None:
        """Create a full-text index across graph entity labels."""
        query = """
        CREATE FULLTEXT INDEX $index_name IF NOT EXISTS
        FOR (
            entity:
                Person|
                Project|
                Customer|
                Team|
                Technology|
                Document
        )
        ON EACH [
            entity.name,
            entity.title,
            entity.description,
            entity.role,
            entity.industry,
            entity.department,
            entity.category
        ]
        """

        self.client.execute_query(
            query,
            {
                "index_name": (
                    self.entity_fulltext_index_name
                )
            },
        )

        logger.info(
            "Entity full-text index command completed for '%s'.",
            self.entity_fulltext_index_name,
        )

    def list_indexes(self) -> list[dict[str, Any]]:
        """Return the configured retrieval indexes."""
        query = """
        SHOW INDEXES
        YIELD
            name,
            type,
            entityType,
            labelsOrTypes,
            properties,
            state,
            populationPercent
        WHERE name IN $index_names
        RETURN
            name,
            type,
            entityType,
            labelsOrTypes,
            properties,
            state,
            populationPercent
        ORDER BY name
        """

        return self.client.execute_query(
            query,
            {
                "index_names": sorted(self.index_names)
            },
        )

    def wait_until_online(
        self,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Wait until all retrieval indexes become available."""
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero."
            )

        deadline = time.monotonic() + timeout_seconds

        while True:
            indexes = self.list_indexes()

            indexes_by_name = {
                str(index.get("name")): index
                for index in indexes
            }

            failed_indexes = [
                name
                for name, index in indexes_by_name.items()
                if str(
                    index.get("state", "")
                ).upper() == "FAILED"
            ]

            if failed_indexes:
                failed = ", ".join(
                    sorted(failed_indexes)
                )

                raise RetrievalIndexError(
                    f"Retrieval indexes failed: {failed}"
                )

            all_indexes_exist = (
                self.index_names
                <= set(indexes_by_name)
            )

            all_indexes_online = (
                all_indexes_exist
                and all(
                    str(
                        indexes_by_name[name].get(
                            "state",
                            "",
                        )
                    ).upper() == "ONLINE"
                    for name in self.index_names
                )
            )

            if all_indexes_online:
                logger.info(
                    "All retrieval indexes are ONLINE."
                )

                return indexes

            if time.monotonic() >= deadline:
                current_states = {
                    name: indexes_by_name.get(
                        name,
                        {},
                    ).get(
                        "state",
                        "MISSING",
                    )
                    for name in sorted(
                        self.index_names
                    )
                }

                raise RetrievalIndexError(
                    "Timed out waiting for retrieval "
                    f"indexes: {current_states}"
                )

            time.sleep(poll_interval_seconds)

    def verify_index_definitions(
        self,
        indexes: list[dict[str, Any]],
    ) -> None:
        """Verify index types, labels, and properties."""
        indexes_by_name = {
            str(index.get("name")): index
            for index in indexes
        }

        missing_indexes = (
            self.index_names
            - set(indexes_by_name)
        )

        if missing_indexes:
            missing = ", ".join(
                sorted(missing_indexes)
            )

            raise RetrievalIndexError(
                f"Missing retrieval indexes: {missing}"
            )

        expected_definitions = {
            self.vector_index_name: {
                "type": "VECTOR",
                "labels": {"Chunk"},
                "properties": {"embedding"},
            },
            self.chunk_fulltext_index_name: {
                "type": "FULLTEXT",
                "labels": {"Chunk"},
                "properties": {"text"},
            },
            self.entity_fulltext_index_name: {
                "type": "FULLTEXT",
                "labels": {
                    "Person",
                    "Project",
                    "Customer",
                    "Team",
                    "Technology",
                    "Document",
                },
                "properties": {
                    "name",
                    "title",
                    "description",
                    "role",
                    "industry",
                    "department",
                    "category",
                },
            },
        }

        for index_name, expected in (
            expected_definitions.items()
        ):
            actual = indexes_by_name[index_name]

            actual_type = str(
                actual.get("type", "")
            ).upper()

            actual_labels = {
                str(label)
                for label in (
                    actual.get("labelsOrTypes")
                    or []
                )
            }

            actual_properties = {
                str(property_name)
                for property_name in (
                    actual.get("properties")
                    or []
                )
            }

            if actual_type != expected["type"]:
                raise RetrievalIndexError(
                    f"Index '{index_name}' has type "
                    f"'{actual_type}', expected "
                    f"'{expected['type']}'."
                )

            if actual_labels != expected["labels"]:
                raise RetrievalIndexError(
                    f"Index '{index_name}' has labels "
                    f"{sorted(actual_labels)}, expected "
                    f"{sorted(expected['labels'])}."
                )

            if (
                actual_properties
                != expected["properties"]
            ):
                raise RetrievalIndexError(
                    f"Index '{index_name}' has properties "
                    f"{sorted(actual_properties)}, expected "
                    f"{sorted(expected['properties'])}."
                )
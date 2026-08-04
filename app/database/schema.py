"""Neo4j schema constraints for the Knowledge Graph AI project."""

import logging
from typing import Final

from app.database.neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


CONSTRAINT_QUERIES: Final[tuple[str, ...]] = (
    """
    CREATE CONSTRAINT person_id_unique IF NOT EXISTS
    FOR (person:Person)
    REQUIRE person.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT project_id_unique IF NOT EXISTS
    FOR (project:Project)
    REQUIRE project.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT customer_id_unique IF NOT EXISTS
    FOR (customer:Customer)
    REQUIRE customer.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT team_id_unique IF NOT EXISTS
    FOR (team:Team)
    REQUIRE team.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT technology_id_unique IF NOT EXISTS
    FOR (technology:Technology)
    REQUIRE technology.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (document:Document)
    REQUIRE document.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (chunk:Chunk)
    REQUIRE chunk.id IS UNIQUE
    """,
)


class GraphSchemaManager:
    """Create and inspect the Neo4j graph schema."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the schema manager."""
        self.client = client

    def create_constraints(self) -> None:
        """Create all application uniqueness constraints."""
        for query in CONSTRAINT_QUERIES:
            self.client.execute_query(query)

        logger.info(
            "Neo4j schema setup completed with %s constraints.",
            len(CONSTRAINT_QUERIES),
        )

    def list_constraints(self) -> list[dict[str, object]]:
        """Return the constraints currently configured in Neo4j."""
        query = """
        SHOW CONSTRAINTS
        YIELD
            name,
            type,
            entityType,
            labelsOrTypes,
            properties
        RETURN
            name,
            type,
            entityType,
            labelsOrTypes,
            properties
        ORDER BY name
        """

        return self.client.execute_query(query)
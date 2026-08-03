"""Reusable Neo4j database client."""

import logging
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from app.config import settings


logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manage Neo4j connections and execute Cypher queries."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.database = database
        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

    @classmethod
    def from_settings(cls) -> "Neo4jClient":
        """Create a client using application environment settings."""
        return cls(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        )

    def verify_connection(self) -> bool:
        """Verify that the Neo4j server is reachable."""
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully.")
            return True
        except Neo4jError:
            logger.exception("Neo4j connectivity verification failed.")
            return False

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Cypher query and return dictionaries."""
        try:
            records, summary, _ = self._driver.execute_query(
                query,
                parameters_=parameters or {},
                database_=self.database,
            )

            logger.debug(
                "Cypher query completed in %s ms and returned %s records.",
                summary.result_available_after,
                len(records),
            )

            return [record.data() for record in records]

        except Neo4jError:
            logger.exception("Neo4j query execution failed.")
            raise

    def close(self) -> None:
        """Close the Neo4j driver and release its resources."""
        self._driver.close()
        logger.info("Neo4j connection closed.")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
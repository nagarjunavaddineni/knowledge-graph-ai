from neo4j import GraphDatabase

from app.config import settings


class Neo4jClient:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str,
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        self.database = database

    @classmethod
    def from_settings(cls) -> "Neo4jClient":
        """Create a client using application settings."""
        return cls(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
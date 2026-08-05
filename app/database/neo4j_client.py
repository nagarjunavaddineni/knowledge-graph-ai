    @classmethod
    def from_settings(cls) -> "Neo4jClient":
        """Create a client using application settings."""
        return cls(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
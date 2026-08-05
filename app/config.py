"""Central application configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def read_positive_integer(
    variable_name: str,
    default_value: int,
) -> int:
    """Read and validate a positive integer environment variable."""
    raw_value = os.getenv(
        variable_name,
        str(default_value),
    ).strip()

    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be a valid integer."
        ) from error

    if parsed_value <= 0:
        raise ValueError(
            f"{variable_name} must be greater than zero."
        )

    return parsed_value


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    openai_api_key: str
    openai_model: str
    openai_embedding_model: str
    embedding_dimensions: int

    chunk_vector_index: str
    chunk_fulltext_index: str
    entity_fulltext_index: str
    retrieval_top_k: int

    app_env: str
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Create settings from environment variables."""
        settings = cls(
            neo4j_uri=os.getenv(
                "NEO4J_URI",
                "",
            ).strip(),
            neo4j_username=os.getenv(
                "NEO4J_USERNAME",
                "neo4j",
            ).strip(),
            neo4j_password=os.getenv(
                "NEO4J_PASSWORD",
                "",
            ).strip(),
            neo4j_database=os.getenv(
                "NEO4J_DATABASE",
                "neo4j",
            ).strip(),
            openai_api_key=os.getenv(
                "OPENAI_API_KEY",
                "",
            ).strip(),
            openai_model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4.1-mini",
            ).strip(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ).strip(),
            embedding_dimensions=read_positive_integer(
                "EMBEDDING_DIMENSIONS",
                1536,
            ),
            chunk_vector_index=os.getenv(
                "CHUNK_VECTOR_INDEX",
                "chunk_embedding_index",
            ).strip(),
            chunk_fulltext_index=os.getenv(
                "CHUNK_FULLTEXT_INDEX",
                "chunk_fulltext_index",
            ).strip(),
            entity_fulltext_index=os.getenv(
                "ENTITY_FULLTEXT_INDEX",
                "entity_fulltext_index",
            ).strip(),
            retrieval_top_k=read_positive_integer(
                "RETRIEVAL_TOP_K",
                5,
            ),
            app_env=os.getenv(
                "APP_ENV",
                "development",
            ).strip(),
            log_level=os.getenv(
                "LOG_LEVEL",
                "INFO",
            ).strip(),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate required configuration values."""
        required_values = {
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_USERNAME": self.neo4j_username,
            "NEO4J_PASSWORD": self.neo4j_password,
            "NEO4J_DATABASE": self.neo4j_database,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_MODEL": self.openai_model,
            "OPENAI_EMBEDDING_MODEL": (
                self.openai_embedding_model
            ),
            "CHUNK_VECTOR_INDEX": self.chunk_vector_index,
            "CHUNK_FULLTEXT_INDEX": (
                self.chunk_fulltext_index
            ),
            "ENTITY_FULLTEXT_INDEX": (
                self.entity_fulltext_index
            ),
        }

        missing_variables = [
            variable_name
            for variable_name, value in required_values.items()
            if not value
        ]

        if missing_variables:
            missing = ", ".join(missing_variables)

            raise ValueError(
                "Missing required environment variables: "
                f"{missing}"
            )


settings = Settings.from_environment()
"""Shared pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock

import pytest


# Provide safe placeholder values before application modules are imported.
# These are not real credentials and no database connection is made.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_MODEL", "gpt-4.1-mini")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    """Return a mocked Neo4j client for repository tests."""
    client = MagicMock()
    client.execute_query.return_value = []
    return client
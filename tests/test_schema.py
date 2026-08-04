"""Unit tests for the Neo4j graph schema manager."""

from unittest.mock import MagicMock

from app.database.schema import (
    CONSTRAINT_QUERIES,
    GraphSchemaManager,
)


EXPECTED_CONSTRAINT_NAMES = {
    "person_id_unique",
    "project_id_unique",
    "customer_id_unique",
    "team_id_unique",
    "technology_id_unique",
    "document_id_unique",
    "chunk_id_unique",
}


def test_create_all_constraints(
    mock_neo4j_client: MagicMock,
) -> None:
    """The schema manager should execute every constraint query."""
    manager = GraphSchemaManager(mock_neo4j_client)

    manager.create_constraints()

    assert len(CONSTRAINT_QUERIES) == 7

    assert (
        mock_neo4j_client.execute_query.call_count
        == len(CONSTRAINT_QUERIES)
    )

    executed_queries = [
        call.args[0]
        for call in (
            mock_neo4j_client
            .execute_query
            .call_args_list
        )
    ]

    for constraint_name in EXPECTED_CONSTRAINT_NAMES:
        assert any(
            constraint_name in query
            for query in executed_queries
        ), f"Constraint query was not executed: {constraint_name}"


def test_constraint_queries_use_if_not_exists() -> None:
    """Every constraint should be safe to execute repeatedly."""
    for query in CONSTRAINT_QUERIES:
        assert "IF NOT EXISTS" in query


def test_constraint_queries_require_unique_id() -> None:
    """Every application constraint should enforce a unique ID."""
    for query in CONSTRAINT_QUERIES:
        assert "REQUIRE" in query
        assert ".id IS UNIQUE" in query


def test_list_constraints(
    mock_neo4j_client: MagicMock,
) -> None:
    """The schema manager should return available constraints."""
    expected_constraints = [
        {
            "name": "person_id_unique",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["Person"],
            "properties": ["id"],
        },
        {
            "name": "chunk_id_unique",
            "type": "UNIQUENESS",
            "entityType": "NODE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["id"],
        },
    ]

    mock_neo4j_client.execute_query.return_value = (
        expected_constraints
    )

    manager = GraphSchemaManager(mock_neo4j_client)

    result = manager.list_constraints()

    assert result == expected_constraints

    mock_neo4j_client.execute_query.assert_called_once()

    query = (
        mock_neo4j_client
        .execute_query
        .call_args
        .args[0]
    )

    assert "SHOW CONSTRAINTS" in query
    assert "YIELD" in query
    assert "name" in query
    assert "type" in query
    assert "entityType" in query
    assert "labelsOrTypes" in query
    assert "properties" in query
    assert "ORDER BY name" in query
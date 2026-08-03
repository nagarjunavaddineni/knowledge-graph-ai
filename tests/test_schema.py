"""Unit tests for the Neo4j schema manager."""

from unittest.mock import MagicMock

from app.database.schema import (
    CONSTRAINT_QUERIES,
    GraphSchemaManager,
)


def test_create_all_constraints(
    mock_neo4j_client: MagicMock,
) -> None:
    """The schema manager should execute every constraint query."""
    manager = GraphSchemaManager(mock_neo4j_client)

    manager.create_constraints()

    assert mock_neo4j_client.execute_query.call_count == len(
        CONSTRAINT_QUERIES
    )

    executed_queries = [
        call.args[0]
        for call in mock_neo4j_client.execute_query.call_args_list
    ]

    assert any("person_id_unique" in query for query in executed_queries)
    assert any("project_id_unique" in query for query in executed_queries)
    assert any("customer_id_unique" in query for query in executed_queries)
    assert any("team_id_unique" in query for query in executed_queries)
    assert any(
        "technology_id_unique" in query
        for query in executed_queries
    )
    assert any(
        "document_id_unique" in query
        for query in executed_queries
    )


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
        }
    ]

    mock_neo4j_client.execute_query.return_value = (
        expected_constraints
    )

    manager = GraphSchemaManager(mock_neo4j_client)
    result = manager.list_constraints()

    assert result == expected_constraints

    query = mock_neo4j_client.execute_query.call_args.args[0]

    assert "SHOW CONSTRAINTS" in query
    assert "ORDER BY name" in query
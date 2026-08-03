"""Unit tests for the knowledge-graph repository."""

from unittest.mock import MagicMock

from app.database.repositories import KnowledgeGraphRepository


def test_upsert_person(mock_neo4j_client: MagicMock) -> None:
    """The repository should create or update a Person node."""
    expected_result = {
        "person": {
            "id": "person-001",
            "name": "Alex Morgan",
            "email": "alex@example.com",
            "role": "Project Manager",
        }
    }

    mock_neo4j_client.execute_query.return_value = [expected_result]

    repository = KnowledgeGraphRepository(mock_neo4j_client)

    result = repository.upsert_person(
        person_id="person-001",
        name="Alex Morgan",
        email="alex@example.com",
        role="Project Manager",
    )

    assert result == expected_result

    query, parameters = (
        mock_neo4j_client.execute_query.call_args.args
    )

    assert "MERGE (person:Person" in query
    assert parameters == {
        "person_id": "person-001",
        "name": "Alex Morgan",
        "email": "alex@example.com",
        "role": "Project Manager",
    }


def test_upsert_project(mock_neo4j_client: MagicMock) -> None:
    """The repository should create or update a Project node."""
    expected_result = {
        "project": {
            "id": "project-001",
            "name": "GraphRAG Assistant",
            "description": "Knowledge graph assistant",
            "status": "active",
        }
    }

    mock_neo4j_client.execute_query.return_value = [expected_result]

    repository = KnowledgeGraphRepository(mock_neo4j_client)

    result = repository.upsert_project(
        project_id="project-001",
        name="GraphRAG Assistant",
        description="Knowledge graph assistant",
    )

    assert result == expected_result

    query, parameters = (
        mock_neo4j_client.execute_query.call_args.args
    )

    assert "MERGE (project:Project" in query
    assert parameters["project_id"] == "project-001"
    assert parameters["status"] == "active"


def test_assign_project_manager(
    mock_neo4j_client: MagicMock,
) -> None:
    """The repository should create a MANAGES relationship."""
    expected_result = {
        "person": "Alex Morgan",
        "relationship": "MANAGES",
        "project": "GraphRAG Assistant",
    }

    mock_neo4j_client.execute_query.return_value = [expected_result]

    repository = KnowledgeGraphRepository(mock_neo4j_client)

    result = repository.assign_project_manager(
        person_id="person-001",
        project_id="project-001",
    )

    assert result == expected_result

    query, parameters = (
        mock_neo4j_client.execute_query.call_args.args
    )

    assert "MANAGES" in query
    assert parameters == {
        "person_id": "person-001",
        "project_id": "project-001",
    }


def test_get_project_context(
    mock_neo4j_client: MagicMock,
) -> None:
    """The repository should return project graph context."""
    expected_result = [
        {
            "project": {
                "id": "project-001",
                "name": "GraphRAG Assistant",
                "status": "active",
            },
            "people": [],
            "customers": [],
            "technologies": [],
        }
    ]

    mock_neo4j_client.execute_query.return_value = expected_result

    repository = KnowledgeGraphRepository(mock_neo4j_client)

    result = repository.get_project_context("project-001")

    assert result == expected_result

    query, parameters = (
        mock_neo4j_client.execute_query.call_args.args
    )

    assert "OPTIONAL MATCH" in query
    assert parameters == {"project_id": "project-001"}


def test_get_graph_statistics(
    mock_neo4j_client: MagicMock,
) -> None:
    """The repository should return graph node counts."""
    expected_result = {
        "people": 2,
        "projects": 1,
        "customers": 1,
        "teams": 1,
        "technologies": 2,
        "documents": 1,
    }

    mock_neo4j_client.execute_query.return_value = [expected_result]

    repository = KnowledgeGraphRepository(mock_neo4j_client)

    result = repository.get_graph_statistics()

    assert result == expected_result
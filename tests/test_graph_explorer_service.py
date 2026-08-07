"""Unit tests for the Neo4j graph-explorer service."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.services.graph_explorer_service import (
    GraphExplorerError,
    GraphExplorerService,
    GraphNode,
)


def create_service(
    mock_client: MagicMock | None = None,
) -> tuple[GraphExplorerService, MagicMock]:
    """Create a graph-explorer service with a mocked client."""
    client = mock_client or MagicMock()

    return GraphExplorerService(client), client


def create_graph_record(
    *,
    source_node_id: str = "node-source",
    source_labels: list[str] | None = None,
    source_properties: dict[str, object] | None = None,
    relationship_id: str | None = "relationship-001",
    relationship_type: str | None = "USES",
    relationship_properties: dict[str, object] | None = None,
    target_node_id: str = "node-target",
    target_labels: list[str] | None = None,
    target_properties: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create one Neo4j graph-explorer query record."""
    return {
        "source_node_id": source_node_id,
        "source_labels": (
            source_labels
            if source_labels is not None
            else ["Project"]
        ),
        "source_properties": (
            source_properties
            if source_properties is not None
            else {
                "id": "project-001",
                "name": (
                    "Enterprise GraphRAG Assistant"
                ),
                "description": (
                    "Enterprise knowledge assistant"
                ),
            }
        ),
        "relationship_id": relationship_id,
        "relationship_type": relationship_type,
        "relationship_properties": (
            relationship_properties
            if relationship_properties is not None
            else {}
        ),
        "target_node_id": target_node_id,
        "target_labels": (
            target_labels
            if target_labels is not None
            else ["Technology"]
        ),
        "target_properties": (
            target_properties
            if target_properties is not None
            else {
                "id": "technology-001",
                "name": "Neo4j",
                "category": "Graph database",
            }
        ),
    }


def test_get_available_node_labels() -> None:
    """Node labels should be normalized and sorted."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        {
            "node_label": "Technology",
        },
        {
            "node_label": "Project",
        },
        {
            "node_label": "Technology",
        },
        {
            "node_label": "",
        },
        {
            "node_label": None,
        },
        {
            "node_label": "Person",
        },
    ]

    result = service.get_available_node_labels()

    assert result == [
        "Person",
        "Project",
        "Technology",
    ]

    query = (
        mock_client.execute_query
        .call_args
        .args[0]
    )

    assert "UNWIND labels(node)" in query
    assert "RETURN DISTINCT node_label" in query


def test_get_available_relationship_types() -> None:
    """Relationship types should be normalized and sorted."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        {
            "relationship_type": "USES",
        },
        {
            "relationship_type": "MANAGES",
        },
        {
            "relationship_type": "USES",
        },
        {
            "relationship_type": None,
        },
        {
            "relationship_type": "",
        },
    ]

    result = (
        service.get_available_relationship_types()
    )

    assert result == [
        "MANAGES",
        "USES",
    ]

    query = (
        mock_client.execute_query
        .call_args
        .args[0]
    )

    assert "type(relationship)" in query
    assert "RETURN DISTINCT" in query


def test_node_label_failure_is_wrapped() -> None:
    """Neo4j label errors should become explorer errors."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    with pytest.raises(
        GraphExplorerError,
        match="available node labels",
    ):
        service.get_available_node_labels()


def test_relationship_type_failure_is_wrapped() -> None:
    """Relationship-type errors should be wrapped clearly."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    with pytest.raises(
        GraphExplorerError,
        match="relationship types",
    ):
        service.get_available_relationship_types()


def test_load_graph_parses_nodes_and_relationships() -> None:
    """Neo4j records should become validated graph models."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            relationship_properties={
                "confidence": 0.98,
            }
        )
    ]

    result = service.load_graph(
        node_labels=[
            "Project",
            "Technology",
        ],
        relationship_types=[
            "uses",
        ],
        node_limit=25,
        relationship_limit=50,
    )

    assert len(result.nodes) == 2
    assert len(result.edges) == 1

    assert result.selected_labels == [
        "Project",
        "Technology",
    ]

    assert result.selected_relationship_types == [
        "USES",
    ]

    assert result.node_limit == 25
    assert result.relationship_limit == 50

    nodes_by_id = {
        node.node_id: node
        for node in result.nodes
    }

    project = nodes_by_id["node-source"]
    technology = nodes_by_id["node-target"]

    assert project.business_id == "project-001"
    assert project.display_name == (
        "Enterprise GraphRAG Assistant"
    )
    assert project.node_type == "Project"
    assert project.labels == ["Project"]

    assert technology.business_id == (
        "technology-001"
    )
    assert technology.display_name == "Neo4j"
    assert technology.node_type == "Technology"

    edge = result.edges[0]

    assert edge.edge_id == "relationship-001"
    assert edge.source_id == "node-source"
    assert edge.target_id == "node-target"
    assert edge.relationship_type == "USES"
    assert edge.properties == {
        "confidence": 0.98,
    }

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "MATCH" in query
    assert "elementId(source)" in query
    assert "elementId(relationship)" in query
    assert "elementId(target)" in query

    assert parameters == {
        "node_labels": [
            "Project",
            "Technology",
        ],
        "relationship_types": [
            "USES",
        ],
        "relationship_limit": 50,
    }


def test_filter_values_are_normalized_and_deduplicated() -> None:
    """Filter collections should remove blanks and duplicates."""
    service, mock_client = create_service()
    mock_client.execute_query.return_value = []

    result = service.load_graph(
        node_labels=[
            "Project",
            "Project",
            " ",
            "Technology",
        ],
        relationship_types=[
            "uses",
            "USES",
            " ",
            "manages",
        ],
    )

    assert result.selected_labels == [
        "Project",
        "Technology",
    ]

    assert result.selected_relationship_types == [
        "USES",
        "MANAGES",
    ]

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["node_labels"] == [
        "Project",
        "Technology",
    ]

    assert parameters["relationship_types"] == [
        "USES",
        "MANAGES",
    ]


def test_empty_filters_search_all_graph_data() -> None:
    """Missing filters should be passed as empty lists."""
    service, mock_client = create_service()
    mock_client.execute_query.return_value = []

    result = service.load_graph(
        node_labels=None,
        relationship_types=None,
    )

    assert result.selected_labels == []
    assert result.selected_relationship_types == []

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["node_labels"] == []
    assert parameters["relationship_types"] == []


def test_duplicate_nodes_and_edges_are_removed() -> None:
    """Repeated graph records should not duplicate output."""
    service, mock_client = create_service()

    duplicate_record = create_graph_record()

    mock_client.execute_query.return_value = [
        duplicate_record,
        duplicate_record,
    ]

    result = service.load_graph()

    assert len(result.nodes) == 2
    assert len(result.edges) == 1


def test_shared_nodes_are_reused_across_relationships() -> None:
    """A node connected by multiple edges should appear once."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            relationship_id="relationship-001",
            relationship_type="USES",
            target_node_id="technology-001",
            target_properties={
                "id": "technology-001",
                "name": "Neo4j",
            },
        ),
        create_graph_record(
            relationship_id="relationship-002",
            relationship_type="USES",
            target_node_id="technology-002",
            target_properties={
                "id": "technology-002",
                "name": "Python",
            },
        ),
    ]

    result = service.load_graph()

    assert len(result.nodes) == 3
    assert len(result.edges) == 2

    node_ids = {
        node.node_id
        for node in result.nodes
    }

    assert node_ids == {
        "node-source",
        "technology-001",
        "technology-002",
    }


def test_node_limit_skips_records_that_exceed_limit() -> None:
    """Records that introduce too many nodes should be skipped."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            source_node_id="node-001",
            target_node_id="node-002",
            relationship_id="relationship-001",
        ),
        create_graph_record(
            source_node_id="node-003",
            target_node_id="node-004",
            relationship_id="relationship-002",
        ),
    ]

    result = service.load_graph(
        node_limit=2,
        relationship_limit=10,
    )

    assert len(result.nodes) == 2
    assert len(result.edges) == 1

    assert result.edges[0].edge_id == (
        "relationship-001"
    )


def test_relationship_limit_bounds_returned_edges() -> None:
    """Returned relationships should not exceed the limit."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            relationship_id="relationship-001",
            target_node_id="technology-001",
        ),
        create_graph_record(
            relationship_id="relationship-002",
            target_node_id="technology-002",
            target_properties={
                "id": "technology-002",
                "name": "Python",
            },
        ),
    ]

    result = service.load_graph(
        node_limit=10,
        relationship_limit=1,
    )

    assert len(result.edges) == 1

    assert result.edges[0].edge_id == (
        "relationship-001"
    )


def test_invalid_node_records_are_skipped() -> None:
    """Records without valid source or target IDs are ignored."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            source_node_id="",
        ),
        create_graph_record(
            target_node_id="",
        ),
        create_graph_record(
            source_node_id="valid-source",
            target_node_id="valid-target",
            relationship_id="valid-relationship",
        ),
    ]

    result = service.load_graph()

    assert len(result.nodes) == 2
    assert len(result.edges) == 1

    assert result.edges[0].edge_id == (
        "valid-relationship"
    )


def test_missing_relationship_type_skips_only_edge() -> None:
    """Valid nodes may remain when relationship type is missing."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            relationship_type=None,
        )
    ]

    result = service.load_graph()

    assert len(result.nodes) == 2
    assert result.edges == []


def test_missing_relationship_id_uses_fallback() -> None:
    """An edge identifier should be generated when absent."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        create_graph_record(
            relationship_id=None,
            relationship_type="USES",
        )
    ]

    result = service.load_graph()

    assert len(result.edges) == 1

    assert result.edges[0].edge_id == (
        "node-source:USES:node-target"
    )


def test_display_name_field_priority() -> None:
    """Node names should use the configured property priority."""
    name_node = GraphExplorerService._create_node(
        node_id="node-name",
        labels=["Project"],
        properties={
            "name": "Name value",
            "title": "Title value",
            "filename": "file.txt",
            "id": "project-001",
        },
    )

    assert name_node is not None
    assert name_node.display_name == "Name value"

    title_node = GraphExplorerService._create_node(
        node_id="node-title",
        labels=["Document"],
        properties={
            "title": "Title value",
            "filename": "file.txt",
            "id": "document-001",
        },
    )

    assert title_node is not None
    assert title_node.display_name == "Title value"

    filename_node = GraphExplorerService._create_node(
        node_id="node-file",
        labels=["Document"],
        properties={
            "filename": "file.txt",
        },
    )

    assert filename_node is not None
    assert filename_node.display_name == "file.txt"


def test_fallback_display_name_uses_node_type() -> None:
    """Nodes without identifying properties get a safe name."""
    node = GraphExplorerService._create_node(
        node_id="4:abc123456789",
        labels=["Technology"],
        properties={},
    )

    assert node is not None

    assert node.display_name == (
        "Technology 23456789"
    )


def test_primary_node_type_uses_priority_order() -> None:
    """Known business labels should take priority."""
    node = GraphExplorerService._create_node(
        node_id="node-001",
        labels=[
            "CustomLabel",
            "Technology",
            "Project",
        ],
        properties={
            "name": "Graph Project",
        },
    )

    assert node is not None
    assert node.node_type == "Project"

    assert node.labels == [
        "CustomLabel",
        "Project",
        "Technology",
    ]


def test_unknown_label_becomes_node_type() -> None:
    """Unknown labeled nodes should use their first label."""
    node = GraphExplorerService._create_node(
        node_id="node-001",
        labels=["CustomLabel"],
        properties={
            "name": "Custom Node",
        },
    )

    assert node is not None
    assert node.node_type == "CustomLabel"


def test_unlabeled_node_uses_generic_type() -> None:
    """Nodes without labels should use the generic Node type."""
    node = GraphExplorerService._create_node(
        node_id="node-001",
        labels=None,
        properties={
            "name": "Unlabeled Node",
        },
    )

    assert node is not None
    assert node.node_type == "Node"
    assert node.labels == []


def test_properties_are_normalized_recursively() -> None:
    """Neo4j property values should become JSON-compatible."""
    timestamp = datetime(
        2026,
        8,
        5,
        19,
        30,
    )

    node = GraphExplorerService._create_node(
        node_id="node-001",
        labels=["Project"],
        properties={
            "id": "project-001",
            "name": "GraphRAG Project",
            "empty_value": None,
            "created_at": timestamp,
            "metadata": {
                "owner": "AI Team",
                "created_at": timestamp,
            },
            "tags": [
                "neo4j",
                timestamp,
            ],
        },
    )

    assert node is not None

    assert "empty_value" not in node.properties

    assert node.properties["created_at"] == str(
        timestamp
    )

    assert node.properties["metadata"] == {
        "owner": "AI Team",
        "created_at": str(timestamp),
    }

    assert node.properties["tags"] == [
        "neo4j",
        str(timestamp),
    ]


def test_non_dictionary_properties_become_empty() -> None:
    """Unexpected property collections should be ignored."""
    node = GraphExplorerService._create_node(
        node_id="node-001",
        labels=["Project"],
        properties=[
            "invalid",
        ],
    )

    assert node is not None
    assert node.properties == {}


@pytest.mark.parametrize(
    "invalid_node_limit",
    [
        0,
        -1,
        501,
    ],
)
def test_invalid_node_limits_are_rejected(
    invalid_node_limit: int,
) -> None:
    """Node limits must remain between 1 and 500."""
    service, mock_client = create_service()

    with pytest.raises(ValueError):
        service.load_graph(
            node_limit=invalid_node_limit,
        )

    mock_client.execute_query.assert_not_called()


@pytest.mark.parametrize(
    "invalid_relationship_limit",
    [
        0,
        -1,
        1001,
    ],
)
def test_invalid_relationship_limits_are_rejected(
    invalid_relationship_limit: int,
) -> None:
    """Relationship limits must remain between 1 and 1000."""
    service, mock_client = create_service()

    with pytest.raises(ValueError):
        service.load_graph(
            relationship_limit=(
                invalid_relationship_limit
            ),
        )

    mock_client.execute_query.assert_not_called()


def test_load_graph_failure_is_wrapped() -> None:
    """Neo4j graph-loading errors should be wrapped."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = RuntimeError(
        "Query failed"
    )

    with pytest.raises(
        GraphExplorerError,
        match="Unable to load the selected graph",
    ):
        service.load_graph()
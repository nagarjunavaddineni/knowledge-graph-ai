"""Unit tests for Neo4j graph-context expansion."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.graph_context_expander import (
    BUSINESS_RELATIONSHIPS,
    GraphContextExpander,
    GraphContextExpansionError,
)
from app.retrieval.models import (
    HybridChunkSearchResult,
    HybridSearchResult,
)


def create_hybrid_chunk(
    chunk_id: str,
    rrf_score: float = 0.03,
    document_id: str = "document-001",
    document_title: str = "Project Overview",
    chunk_index: int = 0,
    text: str | None = None,
) -> HybridChunkSearchResult:
    """Create one hybrid retrieval result."""
    return HybridChunkSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        chunk_index=chunk_index,
        text=text or f"Text for {chunk_id}",
        rrf_score=rrf_score,
        fulltext_score=2.5,
        vector_score=0.92,
        fulltext_rank=1,
        vector_rank=1,
        matched_by=[
            "fulltext",
            "vector",
        ],
        entities=[],
    )


def create_hybrid_result(
    chunks: list[HybridChunkSearchResult],
    question: str = "Which technologies are used?",
) -> HybridSearchResult:
    """Create a complete hybrid-search result."""
    return HybridSearchResult(
        question=question,
        rrf_constant=60,
        candidate_limit=15,
        fulltext_candidate_count=len(chunks),
        vector_candidate_count=len(chunks),
        results=chunks,
    )


def test_expand_adds_entities_and_connections() -> None:
    """Retrieved chunks should be expanded with graph facts."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_title": "Project Overview",
            "chunk_index": 0,
            "text": (
                "The Enterprise GraphRAG Assistant "
                "uses Neo4j."
            ),
            "entities": [
                {
                    "entity_id": "project-001",
                    "name": (
                        "Enterprise GraphRAG Assistant"
                    ),
                    "entity_type": "Project",
                    "description": (
                        "Enterprise knowledge assistant"
                    ),
                    "connections": [
                        {
                            "relationship_type": "USES",
                            "direction": "outgoing",
                            "neighbor_id": "technology-001",
                            "neighbor_name": "Neo4j",
                            "neighbor_type": "Technology",
                        }
                    ],
                }
            ],
        }
    ]

    retrieval_result = create_hybrid_result(
        [
            create_hybrid_chunk(
                chunk_id="chunk-001",
                text=(
                    "The Enterprise GraphRAG Assistant "
                    "uses Neo4j."
                ),
            )
        ]
    )

    expander = GraphContextExpander(mock_client)

    context = expander.expand(
        retrieval_result,
        max_connections_per_entity=5,
    )

    assert context.question == (
        "Which technologies are used?"
    )
    assert context.retrieved_chunk_count == 1
    assert context.expanded_chunk_count == 1
    assert len(context.chunks) == 1

    chunk = context.chunks[0]

    assert chunk.chunk_id == "chunk-001"
    assert chunk.rrf_score == 0.03
    assert len(chunk.entities) == 1

    entity = chunk.entities[0]

    assert entity.entity_id == "project-001"
    assert entity.name == (
        "Enterprise GraphRAG Assistant"
    )
    assert entity.entity_type == "Project"
    assert entity.description == (
        "Enterprise knowledge assistant"
    )

    assert len(entity.connections) == 1

    connection = entity.connections[0]

    assert connection.relationship_type == "USES"
    assert connection.direction == "outgoing"
    assert connection.neighbor_name == "Neo4j"
    assert connection.neighbor_type == "Technology"

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "UNWIND $chunk_ids" in query
    assert "MENTIONS" in query
    assert "relationship_types" in query

    assert parameters == {
        "chunk_ids": [
            "chunk-001",
        ],
        "relationship_types": list(
            BUSINESS_RELATIONSHIPS
        ),
        "connection_limit": 5,
    }


def test_expand_preserves_retrieval_order() -> None:
    """Expanded chunks should retain hybrid ranking order."""
    mock_client = MagicMock()

    # Neo4j records intentionally returned in reverse order.
    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "document_title": "Project Overview",
            "chunk_index": 1,
            "text": "Second chunk",
            "entities": [],
        },
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_title": "Project Overview",
            "chunk_index": 0,
            "text": "First chunk",
            "entities": [],
        },
    ]

    retrieval_result = create_hybrid_result(
        [
            create_hybrid_chunk(
                chunk_id="chunk-001",
                rrf_score=0.04,
                chunk_index=0,
                text="First chunk",
            ),
            create_hybrid_chunk(
                chunk_id="chunk-002",
                rrf_score=0.03,
                chunk_index=1,
                text="Second chunk",
            ),
        ]
    )

    expander = GraphContextExpander(mock_client)

    context = expander.expand(retrieval_result)

    assert [
        chunk.chunk_id
        for chunk in context.chunks
    ] == [
        "chunk-001",
        "chunk-002",
    ]

    assert context.chunks[0].rrf_score == 0.04
    assert context.chunks[1].rrf_score == 0.03


def test_missing_database_record_uses_retrieval_data() -> None:
    """A missing graph record should not remove a result."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    retrieval_chunk = create_hybrid_chunk(
        chunk_id="chunk-001",
        document_id="document-fallback",
        document_title="Fallback Document",
        chunk_index=3,
        text="Fallback retrieval text",
    )

    expander = GraphContextExpander(mock_client)

    context = expander.expand(
        create_hybrid_result(
            [retrieval_chunk]
        )
    )

    assert len(context.chunks) == 1

    result = context.chunks[0]

    assert result.chunk_id == "chunk-001"
    assert result.document_id == "document-fallback"
    assert result.document_title == (
        "Fallback Document"
    )
    assert result.chunk_index == 3
    assert result.text == "Fallback retrieval text"
    assert result.entities == []


def test_empty_retrieval_result_skips_database_query() -> None:
    """No Neo4j query is needed when no chunks were retrieved."""
    mock_client = MagicMock()
    expander = GraphContextExpander(mock_client)

    retrieval_result = create_hybrid_result([])

    context = expander.expand(retrieval_result)

    assert context.retrieved_chunk_count == 0
    assert context.expanded_chunk_count == 0
    assert context.chunks == []

    mock_client.execute_query.assert_not_called()


def test_duplicate_entities_are_removed() -> None:
    """Repeated entity records should be returned once."""
    mock_client = MagicMock()

    duplicate_entity = {
        "entity_id": "technology-001",
        "name": "Neo4j",
        "entity_type": "Technology",
        "description": "Graph database",
        "connections": [],
    }

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "text": "The project uses Neo4j.",
            "entities": [
                duplicate_entity,
                duplicate_entity,
            ],
        }
    ]

    expander = GraphContextExpander(mock_client)

    context = expander.expand(
        create_hybrid_result(
            [
                create_hybrid_chunk(
                    "chunk-001"
                )
            ]
        )
    )

    assert len(context.chunks[0].entities) == 1
    assert (
        context.chunks[0].entities[0].entity_id
        == "technology-001"
    )


def test_duplicate_connections_are_removed() -> None:
    """Repeated graph relationships should appear once."""
    mock_client = MagicMock()

    duplicate_connection = {
        "relationship_type": "USES",
        "direction": "outgoing",
        "neighbor_id": "technology-001",
        "neighbor_name": "Neo4j",
        "neighbor_type": "Technology",
    }

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "text": "The project uses Neo4j.",
            "entities": [
                {
                    "entity_id": "project-001",
                    "name": (
                        "Enterprise GraphRAG Assistant"
                    ),
                    "entity_type": "Project",
                    "connections": [
                        duplicate_connection,
                        duplicate_connection,
                    ],
                }
            ],
        }
    ]

    expander = GraphContextExpander(mock_client)

    context = expander.expand(
        create_hybrid_result(
            [
                create_hybrid_chunk(
                    "chunk-001"
                )
            ]
        )
    )

    connections = (
        context.chunks[0]
        .entities[0]
        .connections
    )

    assert len(connections) == 1
    assert connections[0].neighbor_name == "Neo4j"


def test_invalid_entities_and_connections_are_skipped() -> None:
    """Incomplete graph records should not break expansion."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "text": "Knowledge graph text",
            "entities": [
                None,
                {
                    "entity_id": None,
                    "name": "Missing ID",
                    "entity_type": "Technology",
                },
                {
                    "entity_id": "project-001",
                    "name": "GraphRAG Assistant",
                    "entity_type": "Project",
                    "connections": [
                        None,
                        {
                            "relationship_type": "USES",
                            "direction": "outgoing",
                            "neighbor_id": None,
                            "neighbor_name": "Neo4j",
                            "neighbor_type": "Technology",
                        },
                        {
                            "relationship_type": "USES",
                            "direction": "outgoing",
                            "neighbor_id": "technology-001",
                            "neighbor_name": "Neo4j",
                            "neighbor_type": "Technology",
                        },
                    ],
                },
            ],
        }
    ]

    expander = GraphContextExpander(mock_client)

    context = expander.expand(
        create_hybrid_result(
            [
                create_hybrid_chunk(
                    "chunk-001"
                )
            ]
        )
    )

    assert len(context.chunks[0].entities) == 1
    assert (
        context.chunks[0].entities[0].name
        == "GraphRAG Assistant"
    )

    assert len(
        context.chunks[0]
        .entities[0]
        .connections
    ) == 1


def test_relationship_names_are_normalized() -> None:
    """Relationship names should be uppercase and deduplicated."""
    expander = GraphContextExpander(
        client=MagicMock(),
        relationship_types=(
            "uses",
            "USES",
            "works_on",
        ),
    )

    assert expander.relationship_types == (
        "USES",
        "WORKS_ON",
    )


def test_database_failure_is_wrapped() -> None:
    """Neo4j errors should become context-expansion errors."""
    mock_client = MagicMock()

    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    expander = GraphContextExpander(mock_client)

    with pytest.raises(
        GraphContextExpansionError,
        match="Unable to expand retrieval results",
    ):
        expander.expand(
            create_hybrid_result(
                [
                    create_hybrid_chunk(
                        "chunk-001"
                    )
                ]
            )
        )


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 51],
)
def test_invalid_connection_limits_are_rejected(
    invalid_limit: int,
) -> None:
    """Connection limits must remain between one and fifty."""
    expander = GraphContextExpander(MagicMock())

    with pytest.raises(ValueError):
        expander.expand(
            create_hybrid_result(
                [
                    create_hybrid_chunk(
                        "chunk-001"
                    )
                ]
            ),
            max_connections_per_entity=invalid_limit,
        )


def test_empty_relationship_collection_is_rejected() -> None:
    """At least one graph relationship type is required."""
    with pytest.raises(
        ValueError,
        match="At least one relationship type",
    ):
        GraphContextExpander(
            client=MagicMock(),
            relationship_types=(),
        )


def test_blank_relationship_name_is_rejected() -> None:
    """Relationship names cannot be blank."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        GraphContextExpander(
            client=MagicMock(),
            relationship_types=(
                "USES",
                "   ",
            ),
        )


def test_invalid_relationship_characters_are_rejected() -> None:
    """Relationship names should use safe identifier characters."""
    with pytest.raises(
        ValueError,
        match="letters, numbers, and underscores",
    ):
        GraphContextExpander(
            client=MagicMock(),
            relationship_types=(
                "WORKS-ON",
            ),
        )
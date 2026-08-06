"""Unit tests for Neo4j full-text retrieval."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.fulltext_retriever import (
    FullTextRetrievalError,
    FullTextRetriever,
)


def create_retriever(
    mock_client: MagicMock,
    default_top_k: int = 5,
) -> FullTextRetriever:
    """Create a full-text retriever for testing."""
    return FullTextRetriever(
        client=mock_client,
        chunk_index_name="test_chunk_fulltext_index",
        entity_index_name="test_entity_fulltext_index",
        default_top_k=default_top_k,
    )


def test_build_lucene_query_removes_stop_words() -> None:
    """Common question words should not dominate retrieval."""
    query = FullTextRetriever._build_lucene_query(
        "Which project uses Neo4j?"
    )

    assert query == '"project" OR "uses" OR "neo4j"'


def test_build_lucene_query_removes_duplicate_terms() -> None:
    """Duplicate terms should appear only once."""
    query = FullTextRetriever._build_lucene_query(
        "Neo4j project Neo4j project"
    )

    assert query == '"neo4j" OR "project"'


def test_build_lucene_query_falls_back_for_stop_words() -> None:
    """A question containing only stop words should remain searchable."""
    query = FullTextRetriever._build_lucene_query(
        "what is it"
    )

    assert query == '"what" OR "is" OR "it"'


def test_search_chunks_parses_neo4j_records() -> None:
    """Chunk search records should become validated models."""
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
            "score": 2.75,
            "entities": [
                {
                    "entity_id": "project-001",
                    "name": "Enterprise GraphRAG Assistant",
                    "entity_type": "Project",
                },
                {
                    "entity_id": "technology-001",
                    "name": "Neo4j",
                    "entity_type": "Technology",
                },
                None,
            ],
        }
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search_chunks(
        "Which project uses Neo4j?",
        limit=3,
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-001"
    assert result.document_id == "document-001"
    assert result.document_title == "Project Overview"
    assert result.chunk_index == 0
    assert result.score == 2.75
    assert len(result.entities) == 2
    assert result.entities[1].name == "Neo4j"

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "db.index.fulltext.queryNodes" in query
    assert "HAS_CHUNK" in query
    assert "MENTIONS" in query

    assert parameters == {
        "index_name": "test_chunk_fulltext_index",
        "lucene_query": (
            '"project" OR "uses" OR "neo4j"'
        ),
        "limit": 3,
    }


def test_search_chunks_skips_invalid_records() -> None:
    """Records without a chunk ID or text should be skipped."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": None,
            "text": "Missing ID",
            "score": 1.0,
        },
        {
            "chunk_id": "chunk-002",
            "text": None,
            "score": 1.0,
        },
        {
            "chunk_id": "chunk-003",
            "text": "Valid text",
            "score": 0.8,
            "entities": [],
        },
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search_chunks("Neo4j")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-003"


def test_search_entities_parses_graph_context() -> None:
    """Entity search should parse source chunks and connections."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "entity_id": "project-001",
            "name": "Enterprise GraphRAG Assistant",
            "entity_type": "Project",
            "description": "Enterprise AI assistant",
            "score": 3.25,
            "source_chunks": [
                {
                    "chunk_id": "chunk-001",
                    "document_id": "document-001",
                    "document_title": "Project Overview",
                    "text": (
                        "The project uses Neo4j and Python."
                    ),
                },
                None,
            ],
            "connections": [
                {
                    "relationship_type": "USES",
                    "direction": "outgoing",
                    "neighbor_id": "technology-001",
                    "neighbor_name": "Neo4j",
                    "neighbor_type": "Technology",
                },
                None,
            ],
        }
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search_entities(
        "Enterprise GraphRAG Assistant",
        limit=4,
    )

    assert len(results) == 1

    result = results[0]

    assert result.entity_id == "project-001"
    assert result.entity_type == "Project"
    assert result.description == "Enterprise AI assistant"
    assert result.score == 3.25

    assert len(result.source_chunks) == 1
    assert result.source_chunks[0].chunk_id == "chunk-001"

    assert len(result.connections) == 1
    assert result.connections[0].relationship_type == "USES"
    assert result.connections[0].neighbor_name == "Neo4j"

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "db.index.fulltext.queryNodes" in query
    assert "source_chunks" in query
    assert "connections" in query

    assert parameters["index_name"] == (
        "test_entity_fulltext_index"
    )
    assert parameters["limit"] == 4
    assert parameters["context_limit"] == 3
    assert parameters["connection_limit"] == 10


def test_search_combines_chunk_and_entity_results() -> None:
    """Combined search should query both indexes."""
    mock_client = MagicMock()

    mock_client.execute_query.side_effect = [
        [
            {
                "chunk_id": "chunk-001",
                "text": "The project uses Neo4j.",
                "score": 1.8,
                "entities": [],
            }
        ],
        [
            {
                "entity_id": "technology-001",
                "name": "Neo4j",
                "entity_type": "Technology",
                "score": 2.1,
                "source_chunks": [],
                "connections": [],
            }
        ],
    ]

    retriever = create_retriever(mock_client)

    result = retriever.search(
        "Which project uses Neo4j?",
        limit=5,
    )

    assert result.question == "Which project uses Neo4j?"
    assert result.lucene_query == (
        '"project" OR "uses" OR "neo4j"'
    )
    assert len(result.chunks) == 1
    assert len(result.entities) == 1

    assert mock_client.execute_query.call_count == 2


def test_chunk_query_failure_is_wrapped() -> None:
    """Neo4j chunk-search failures should produce a clear error."""
    mock_client = MagicMock()
    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    retriever = create_retriever(mock_client)

    with pytest.raises(
        FullTextRetrievalError,
        match="chunk full-text index",
    ):
        retriever.search_chunks("Neo4j")


def test_entity_query_failure_is_wrapped() -> None:
    """Neo4j entity-search failures should produce a clear error."""
    mock_client = MagicMock()
    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    retriever = create_retriever(mock_client)

    with pytest.raises(
        FullTextRetrievalError,
        match="entity full-text index",
    ):
        retriever.search_entities("Neo4j")


def test_empty_question_is_rejected() -> None:
    """A blank question cannot be searched."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retriever.search("   ")


def test_non_string_question_is_rejected() -> None:
    """The question must be a string."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        retriever.search(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 51],
)
def test_invalid_limits_are_rejected(
    invalid_limit: int,
) -> None:
    """Search limits must be between one and fifty."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(ValueError):
        retriever.search(
            "Neo4j",
            limit=invalid_limit,
        )


def test_empty_index_names_are_rejected() -> None:
    """Both full-text index names are required."""
    with pytest.raises(
        ValueError,
        match="chunk_index_name cannot be empty",
    ):
        FullTextRetriever(
            client=MagicMock(),
            chunk_index_name="",
            entity_index_name="entity_index",
        )

    with pytest.raises(
        ValueError,
        match="entity_index_name cannot be empty",
    ):
        FullTextRetriever(
            client=MagicMock(),
            chunk_index_name="chunk_index",
            entity_index_name="",
        )


def test_invalid_default_top_k_is_rejected() -> None:
    """The default result limit must be positive."""
    with pytest.raises(
        ValueError,
        match="default_top_k must be greater than zero",
    ):
        FullTextRetriever(
            client=MagicMock(),
            chunk_index_name="chunk_index",
            entity_index_name="entity_index",
            default_top_k=0,
        )
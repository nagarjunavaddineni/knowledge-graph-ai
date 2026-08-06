"""Unit tests for semantic Neo4j vector retrieval."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
)
from app.retrieval.vector_retriever import (
    SemanticVectorRetriever,
    VectorRetrievalError,
)


def create_embedding_service(
    dimensions: int = 3,
) -> MagicMock:
    """Create a mocked embedding service."""
    service = MagicMock()
    service.dimensions = dimensions
    service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    return service


def create_retriever(
    mock_client: MagicMock,
    embedding_service: MagicMock | None = None,
    default_top_k: int = 5,
) -> SemanticVectorRetriever:
    """Create a semantic vector retriever for testing."""
    return SemanticVectorRetriever(
        client=mock_client,
        embedding_service=(
            embedding_service
            or create_embedding_service()
        ),
        vector_index_name="test_chunk_vector_index",
        default_top_k=default_top_k,
    )


def test_search_generates_question_embedding() -> None:
    """A natural-language question should be embedded once."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    embedding_service = create_embedding_service()

    retriever = create_retriever(
        mock_client,
        embedding_service,
    )

    result = retriever.search(
        "Which project uses Neo4j?",
        limit=3,
        minimum_score=0.5,
    )

    assert result == []

    embedding_service.embed_text.assert_called_once_with(
        "Which project uses Neo4j?"
    )

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "db.index.vector.queryNodes" in query
    assert parameters["index_name"] == (
        "test_chunk_vector_index"
    )
    assert parameters["candidate_limit"] == 9
    assert parameters["query_embedding"] == [
        0.1,
        0.2,
        0.3,
    ]
    assert parameters["minimum_score"] == 0.5
    assert parameters["limit"] == 3


def test_search_parses_vector_results() -> None:
    """Neo4j vector results should become validated models."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_title": "Project Overview",
            "chunk_index": 1,
            "text": (
                "The Enterprise GraphRAG Assistant "
                "uses Neo4j."
            ),
            "score": 0.91,
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
            ],
        }
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search(
        "What graph database does the assistant use?"
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-001"
    assert result.document_id == "document-001"
    assert result.document_title == "Project Overview"
    assert result.chunk_index == 1
    assert result.score == 0.91

    assert len(result.entities) == 2
    assert result.entities[1].name == "Neo4j"


def test_invalid_vector_records_are_skipped() -> None:
    """Records without a Chunk ID or text should be ignored."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": None,
            "text": "Missing ID",
            "score": 0.8,
        },
        {
            "chunk_id": "chunk-002",
            "text": None,
            "score": 0.7,
        },
        {
            "chunk_id": "chunk-003",
            "text": "Valid text",
            "score": 0.6,
            "entities": [],
        },
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search("Neo4j")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-003"


def test_invalid_entity_references_are_skipped() -> None:
    """Incomplete entity maps should not enter the results."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "text": "The project uses Neo4j.",
            "score": 0.9,
            "entities": [
                {
                    "entity_id": "technology-001",
                    "name": "Neo4j",
                    "entity_type": "Technology",
                },
                {
                    "entity_id": None,
                    "name": "Python",
                    "entity_type": "Technology",
                },
                None,
            ],
        }
    ]

    retriever = create_retriever(mock_client)

    results = retriever.search("Technology")

    assert len(results[0].entities) == 1
    assert results[0].entities[0].name == "Neo4j"


def test_search_by_embedding_accepts_integer_values() -> None:
    """Numeric vectors should be normalized to floats."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    retriever = create_retriever(mock_client)

    result = retriever.search_by_embedding(
        query_embedding=[1, 2, 3],
        limit=2,
    )

    assert result == []

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["query_embedding"] == [
        1.0,
        2.0,
        3.0,
    ]


def test_candidate_limit_is_capped_at_one_hundred() -> None:
    """The Neo4j candidate request should remain bounded."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    retriever = create_retriever(mock_client)

    retriever.search_by_embedding(
        query_embedding=[0.1, 0.2, 0.3],
        limit=50,
    )

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["candidate_limit"] == 100
    assert parameters["limit"] == 50


def test_empty_question_is_rejected() -> None:
    """A blank question cannot be embedded."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retriever.search("   ")


def test_non_string_question_is_rejected() -> None:
    """The semantic-search question must be a string."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        retriever.search(123)  # type: ignore[arg-type]


def test_empty_query_embedding_is_rejected() -> None:
    """A semantic search requires a query vector."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        ValueError,
        match="query_embedding cannot be empty",
    ):
        retriever.search_by_embedding([])


def test_incorrect_query_dimensions_are_rejected() -> None:
    """The query vector must match configured dimensions."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        ValueError,
        match="2 dimensions; expected 3",
    ):
        retriever.search_by_embedding(
            [0.1, 0.2]
        )


def test_non_numeric_query_value_is_rejected() -> None:
    """Every query-vector value must be numeric."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        retriever.search_by_embedding(
            [
                0.1,
                "invalid",
                0.3,
            ]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "invalid_score",
    [-0.01, 1.01],
)
def test_invalid_minimum_scores_are_rejected(
    invalid_score: float,
) -> None:
    """Similarity thresholds must remain between zero and one."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        retriever.search_by_embedding(
            [0.1, 0.2, 0.3],
            minimum_score=invalid_score,
        )


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 51],
)
def test_invalid_limits_are_rejected(
    invalid_limit: int,
) -> None:
    """Vector result limits must be between one and fifty."""
    retriever = create_retriever(MagicMock())

    with pytest.raises(ValueError):
        retriever.search_by_embedding(
            [0.1, 0.2, 0.3],
            limit=invalid_limit,
        )


def test_embedding_generation_error_is_preserved() -> None:
    """Known embedding failures should remain identifiable."""
    mock_client = MagicMock()

    embedding_service = create_embedding_service()
    embedding_service.embed_text.side_effect = (
        EmbeddingGenerationError(
            "Embedding request failed"
        )
    )

    retriever = create_retriever(
        mock_client,
        embedding_service,
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="Embedding request failed",
    ):
        retriever.search("Neo4j")


def test_unexpected_embedding_failure_is_wrapped() -> None:
    """Unexpected embedding errors should be wrapped."""
    embedding_service = create_embedding_service()
    embedding_service.embed_text.side_effect = RuntimeError(
        "Unexpected failure"
    )

    retriever = create_retriever(
        MagicMock(),
        embedding_service,
    )

    with pytest.raises(
        VectorRetrievalError,
        match="question embedding",
    ):
        retriever.search("Neo4j")


def test_neo4j_vector_query_failure_is_wrapped() -> None:
    """Neo4j vector-index errors should be wrapped clearly."""
    mock_client = MagicMock()
    mock_client.execute_query.side_effect = RuntimeError(
        "Index unavailable"
    )

    retriever = create_retriever(mock_client)

    with pytest.raises(
        VectorRetrievalError,
        match="vector index",
    ):
        retriever.search_by_embedding(
            [0.1, 0.2, 0.3]
        )


def test_empty_vector_index_name_is_rejected() -> None:
    """A vector-index name is required."""
    with pytest.raises(
        ValueError,
        match="vector_index_name cannot be empty",
    ):
        SemanticVectorRetriever(
            client=MagicMock(),
            embedding_service=create_embedding_service(),
            vector_index_name="",
        )


def test_invalid_default_top_k_is_rejected() -> None:
    """The default result count must be positive."""
    with pytest.raises(
        ValueError,
        match="default_top_k must be greater than zero",
    ):
        SemanticVectorRetriever(
            client=MagicMock(),
            embedding_service=create_embedding_service(),
            vector_index_name="vector_index",
            default_top_k=0,
        )
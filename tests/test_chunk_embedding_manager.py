"""Unit tests for Neo4j Chunk embedding management."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.chunk_embedding_manager import (
    ChunkEmbeddingError,
    ChunkEmbeddingManager,
)


def create_manager(
    mock_client: MagicMock,
    dimensions: int = 3,
) -> ChunkEmbeddingManager:
    """Create a ChunkEmbeddingManager for testing."""
    return ChunkEmbeddingManager(
        client=mock_client,
        model="test-embedding-model",
        dimensions=dimensions,
    )


def test_get_chunks_requiring_embeddings() -> None:
    """The manager should parse eligible Neo4j Chunk records."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "chunk_index": 0,
            "text": "Neo4j stores connected information.",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "chunk_index": 1,
            "text": "Python powers the GraphRAG application.",
        },
    ]

    manager = create_manager(mock_client)

    chunks = manager.get_chunks_requiring_embeddings()

    assert len(chunks) == 2

    assert chunks[0].chunk_id == "chunk-001"
    assert chunks[0].document_id == "document-001"
    assert chunks[0].chunk_index == 0
    assert (
        chunks[0].text
        == "Neo4j stores connected information."
    )

    assert chunks[1].chunk_id == "chunk-002"
    assert chunks[1].chunk_index == 1

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "MATCH (chunk:Chunk)" in query
    assert "chunk.embedding IS NULL" in query

    assert parameters == {
        "force": False,
        "embedding_model": "test-embedding-model",
        "embedding_dimensions": 3,
        "limit": 2_147_483_647,
    }


def test_get_chunks_supports_force_regeneration() -> None:
    """The force option should be sent to Neo4j."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    manager = create_manager(mock_client)

    result = manager.get_chunks_requiring_embeddings(
        force=True,
        limit=25,
    )

    assert result == []

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["force"] is True
    assert parameters["limit"] == 25


def test_get_chunks_skips_invalid_records() -> None:
    """Records without a chunk ID or text should be ignored."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "chunk_id": None,
            "document_id": "document-001",
            "chunk_index": 0,
            "text": "Missing chunk ID",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "chunk_index": 1,
            "text": None,
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "document-001",
            "chunk_index": 2,
            "text": "Valid chunk text",
        },
    ]

    manager = create_manager(mock_client)

    chunks = manager.get_chunks_requiring_embeddings()

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-003"


def test_invalid_chunk_limit_is_rejected() -> None:
    """The optional limit must be greater than zero."""
    manager = create_manager(MagicMock())

    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        manager.get_chunks_requiring_embeddings(
            limit=0
        )


def test_store_embeddings_successfully() -> None:
    """Valid embeddings should be written to Neo4j."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "updated_count": 2,
        }
    ]

    manager = create_manager(mock_client)

    updated_count = manager.store_embeddings(
        [
            {
                "chunk_id": "chunk-001",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "chunk_id": "chunk-002",
                "embedding": [0.4, 0.5, 0.6],
            },
        ]
    )

    assert updated_count == 2

    query, parameters = (
        mock_client.execute_query.call_args.args
    )

    assert "UNWIND $items AS item" in query
    assert "db.create.setNodeVectorProperty" in query
    assert "chunk.embedding_model" in query

    assert parameters == {
        "items": [
            {
                "chunk_id": "chunk-001",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "chunk_id": "chunk-002",
                "embedding": [0.4, 0.5, 0.6],
            },
        ],
        "embedding_model": "test-embedding-model",
        "embedding_dimensions": 3,
    }


def test_store_embeddings_converts_integers_to_floats() -> None:
    """Integer vector values should be normalized to floats."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "updated_count": 1,
        }
    ]

    manager = create_manager(mock_client)

    result = manager.store_embeddings(
        [
            {
                "chunk_id": "chunk-001",
                "embedding": [1, 2, 3],
            }
        ]
    )

    assert result == 1

    parameters = (
        mock_client.execute_query
        .call_args
        .args[1]
    )

    assert parameters["items"] == [
        {
            "chunk_id": "chunk-001",
            "embedding": [1.0, 2.0, 3.0],
        }
    ]


def test_empty_embedding_collection_returns_zero() -> None:
    """No database query is needed for an empty collection."""
    mock_client = MagicMock()
    manager = create_manager(mock_client)

    result = manager.store_embeddings([])

    assert result == 0
    mock_client.execute_query.assert_not_called()


def test_missing_chunk_id_is_rejected() -> None:
    """Every embedding must include a chunk identifier."""
    manager = create_manager(MagicMock())

    with pytest.raises(
        ChunkEmbeddingError,
        match="must include a chunk_id",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )


def test_duplicate_chunk_ids_are_rejected() -> None:
    """A batch cannot contain duplicate Chunk identifiers."""
    manager = create_manager(MagicMock())

    with pytest.raises(
        ChunkEmbeddingError,
        match="Duplicate chunk embedding",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "chunk_id": "chunk-001",
                    "embedding": [0.4, 0.5, 0.6],
                },
            ]
        )


def test_embedding_must_be_a_list() -> None:
    """The vector must be provided as a list."""
    manager = create_manager(MagicMock())

    with pytest.raises(
        ChunkEmbeddingError,
        match="must be a list",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": (
                        0.1,
                        0.2,
                        0.3,
                    ),
                }
            ]
        )


def test_incorrect_vector_dimensions_are_rejected() -> None:
    """Every vector must match the configured dimensions."""
    manager = create_manager(
        MagicMock(),
        dimensions=3,
    )

    with pytest.raises(
        ChunkEmbeddingError,
        match="contains 2 dimensions; expected 3",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": [0.1, 0.2],
                }
            ]
        )


def test_non_numeric_vector_value_is_rejected() -> None:
    """Every embedding value must be numeric."""
    manager = create_manager(MagicMock())

    with pytest.raises(
        ChunkEmbeddingError,
        match="contains a non-numeric value",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": [
                        0.1,
                        "invalid",
                        0.3,
                    ],
                }
            ]
        )


def test_missing_storage_result_is_rejected() -> None:
    """Neo4j must return a result after storing vectors."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    manager = create_manager(mock_client)

    with pytest.raises(
        ChunkEmbeddingError,
        match="returned no result",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )


def test_updated_count_mismatch_is_rejected() -> None:
    """Neo4j must update every supplied Chunk node."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "updated_count": 1,
        }
    ]

    manager = create_manager(mock_client)

    with pytest.raises(
        ChunkEmbeddingError,
        match="updated 1 chunks, but 2 embeddings",
    ):
        manager.store_embeddings(
            [
                {
                    "chunk_id": "chunk-001",
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "chunk_id": "chunk-002",
                    "embedding": [0.4, 0.5, 0.6],
                },
            ]
        )


def test_get_statistics() -> None:
    """The manager should return embedding statistics."""
    mock_client = MagicMock()

    mock_client.execute_query.return_value = [
        {
            "total_chunks": 10,
            "embedded_chunks": 8,
            "chunks_without_embeddings": 2,
        }
    ]

    manager = create_manager(mock_client)

    result = manager.get_statistics()

    assert result == {
        "total_chunks": 10,
        "embedded_chunks": 8,
        "chunks_without_embeddings": 2,
    }

    query = (
        mock_client.execute_query
        .call_args
        .args[0]
    )

    assert "count(chunk) AS total_chunks" in query
    assert (
        "count(chunk.embedding) AS embedded_chunks"
        in query
    )


def test_get_statistics_returns_zero_defaults() -> None:
    """Missing Neo4j results should return zero counts."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []

    manager = create_manager(mock_client)

    result = manager.get_statistics()

    assert result == {
        "total_chunks": 0,
        "embedded_chunks": 0,
        "chunks_without_embeddings": 0,
    }


def test_empty_embedding_model_is_rejected() -> None:
    """The embedding model name cannot be empty."""
    with pytest.raises(
        ValueError,
        match="embedding model cannot be empty",
    ):
        ChunkEmbeddingManager(
            client=MagicMock(),
            model="",
            dimensions=3,
        )


def test_invalid_dimensions_are_rejected() -> None:
    """Embedding dimensions must be positive."""
    with pytest.raises(
        ValueError,
        match="dimensions must be greater than zero",
    ):
        ChunkEmbeddingManager(
            client=MagicMock(),
            model="test-model",
            dimensions=0,
        )
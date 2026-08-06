"""Unit tests for the OpenAI embedding service."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
    EmbeddingService,
)


def create_embedding_item(
    index: int,
    values: list[float],
) -> MagicMock:
    """Create one mocked OpenAI embedding item."""
    item = MagicMock()
    item.index = index
    item.embedding = values

    return item


def create_embedding_response(
    items: list[MagicMock],
) -> MagicMock:
    """Create a mocked OpenAI embeddings response."""
    response = MagicMock()
    response.data = items

    return response


def test_embed_single_text_returns_vector() -> None:
    """The service should generate one embedding vector."""
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = (
        create_embedding_response(
            [
                create_embedding_item(
                    index=0,
                    values=[0.1, 0.2, 0.3],
                )
            ]
        )
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    result = service.embed_text(
        "Enterprise knowledge graph"
    )

    assert result == [0.1, 0.2, 0.3]

    mock_client.embeddings.create.assert_called_once_with(
        model="test-embedding-model",
        input=["Enterprise knowledge graph"],
        dimensions=3,
        encoding_format="float",
    )


def test_embed_multiple_texts_returns_ordered_vectors() -> None:
    """Embeddings should be returned in input-index order."""
    mock_client = MagicMock()

    # Intentionally return the items in reverse order.
    mock_client.embeddings.create.return_value = (
        create_embedding_response(
            [
                create_embedding_item(
                    index=1,
                    values=[0.4, 0.5, 0.6],
                ),
                create_embedding_item(
                    index=0,
                    values=[0.1, 0.2, 0.3],
                ),
            ]
        )
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    result = service.embed_texts(
        [
            "Neo4j graph database",
            "Python application",
        ]
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_integer_vector_values_are_converted_to_float() -> None:
    """Numeric embedding values should be converted to floats."""
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = (
        create_embedding_response(
            [
                create_embedding_item(
                    index=0,
                    values=[1, 2, 3],
                )
            ]
        )
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    result = service.embed_text("Neo4j")

    assert result == [1.0, 2.0, 3.0]

    assert all(
        isinstance(value, float)
        for value in result
    )


def test_empty_text_collection_is_rejected() -> None:
    """At least one text value must be supplied."""
    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="At least one text value is required",
    ):
        service.embed_texts([])


def test_blank_text_is_rejected() -> None:
    """Whitespace-only text should not be embedded."""
    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        service.embed_texts(
            [
                "Valid text",
                "   ",
            ]
        )


def test_non_string_text_is_rejected() -> None:
    """Embedding inputs must be strings."""
    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=MagicMock(),
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        service.embed_texts(
            [
                "Valid text",
                123,  # type: ignore[list-item]
            ]
        )


def test_empty_openai_response_is_rejected() -> None:
    """An OpenAI response without data should fail clearly."""
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = (
        create_embedding_response([])
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="returned no embedding data",
    ):
        service.embed_text("Neo4j")


def test_embedding_count_mismatch_is_rejected() -> None:
    """The response count must match the input count."""
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = (
        create_embedding_response(
            [
                create_embedding_item(
                    index=0,
                    values=[0.1, 0.2, 0.3],
                )
            ]
        )
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="number of returned embeddings",
    ):
        service.embed_texts(
            [
                "First text",
                "Second text",
            ]
        )


def test_incorrect_embedding_dimensions_are_rejected() -> None:
    """Every returned vector must use configured dimensions."""
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = (
        create_embedding_response(
            [
                create_embedding_item(
                    index=0,
                    values=[0.1, 0.2],
                )
            ]
        )
    )

    service = EmbeddingService(
        api_key="test-api-key",
        model="test-embedding-model",
        dimensions=3,
        client=mock_client,
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="2 dimensions; expected 3",
    ):
        service.embed_text("Neo4j")


def test_missing_api_key_is_rejected() -> None:
    """The embedding service requires an API key."""
    with pytest.raises(
        ValueError,
        match="API key is required",
    ):
        EmbeddingService(
            api_key="",
            model="test-embedding-model",
            dimensions=3,
        )


def test_missing_model_is_rejected() -> None:
    """The embedding model cannot be empty."""
    with pytest.raises(
        ValueError,
        match="embedding model is required",
    ):
        EmbeddingService(
            api_key="test-api-key",
            model="",
            dimensions=3,
        )


def test_invalid_dimensions_are_rejected() -> None:
    """Embedding dimensions must be positive."""
    with pytest.raises(
        ValueError,
        match="dimensions must be greater than zero",
    ):
        EmbeddingService(
            api_key="test-api-key",
            model="test-embedding-model",
            dimensions=0,
        )


def test_negative_retry_count_is_rejected() -> None:
    """The OpenAI retry count cannot be negative."""
    with pytest.raises(
        ValueError,
        match="max_retries cannot be negative",
    ):
        EmbeddingService(
            api_key="test-api-key",
            model="test-embedding-model",
            dimensions=3,
            max_retries=-1,
        )
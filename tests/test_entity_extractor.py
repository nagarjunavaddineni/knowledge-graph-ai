"""Unit tests for AI entity and relationship extraction."""

from unittest.mock import MagicMock

import pytest

from app.ingestion.entity_extractor import (
    EntityExtractionError,
    EntityExtractor,
)
from app.ingestion.models import (
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
    TextChunk,
)


def create_chunk(
    chunk_id: str = "chunk-test-001",
    text: str = (
        "Alex Morgan manages the Enterprise GraphRAG Assistant. "
        "The project uses Neo4j."
    ),
) -> TextChunk:
    """Create a validated text chunk for tests."""
    return TextChunk(
        chunk_id=chunk_id,
        document_id="document-test-001",
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def create_mock_completion(
    parsed_result: ExtractionResult,
) -> MagicMock:
    """Create a mocked OpenAI structured-output response."""
    message = MagicMock()
    message.refusal = None
    message.parsed = parsed_result

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]

    return completion


def test_extract_chunk_returns_validated_result() -> None:
    """The extractor should return parsed entities and relationships."""
    chunk = create_chunk()

    parsed_result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=[
            GraphEntity(
                name="Alex Morgan",
                entity_type=EntityType.PERSON,
                description="Project manager",
                confidence=0.98,
            ),
            GraphEntity(
                name="Enterprise GraphRAG Assistant",
                entity_type=EntityType.PROJECT,
                confidence=0.99,
            ),
            GraphEntity(
                name="Neo4j",
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.99,
            ),
        ],
        relationships=[
            GraphRelationship(
                source_name="Alex Morgan",
                source_type=EntityType.PERSON,
                relationship_type=RelationshipType.MANAGES,
                target_name="Enterprise GraphRAG Assistant",
                target_type=EntityType.PROJECT,
                evidence=(
                    "Alex Morgan manages the "
                    "Enterprise GraphRAG Assistant."
                ),
                confidence=0.98,
            ),
            GraphRelationship(
                source_name="Enterprise GraphRAG Assistant",
                source_type=EntityType.PROJECT,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence="The project uses Neo4j.",
                confidence=0.99,
            ),
        ],
        summary=(
            "Alex Morgan manages a GraphRAG project "
            "that uses Neo4j."
        ),
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        create_mock_completion(parsed_result)
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    result = extractor.extract_chunk(chunk)

    assert result.chunk_id == chunk.chunk_id
    assert len(result.entities) == 3
    assert len(result.relationships) == 2

    assert result.entities[0].name == "Alex Morgan"

    mock_client.chat.completions.parse.assert_called_once()

    call_arguments = (
        mock_client.chat.completions.parse.call_args.kwargs
    )

    assert call_arguments["model"] == "test-model"
    assert call_arguments["response_format"] is ExtractionResult
    assert len(call_arguments["messages"]) == 2

    user_message = call_arguments["messages"][1]["content"]

    assert chunk.chunk_id in user_message
    assert chunk.document_id in user_message
    assert chunk.text in user_message


def test_duplicate_entities_keep_highest_confidence() -> None:
    """Duplicate entities should be consolidated."""
    chunk = create_chunk()

    parsed_result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=[
            GraphEntity(
                name="Neo4j",
                entity_type=EntityType.TECHNOLOGY,
                description="Graph database",
                confidence=0.70,
            ),
            GraphEntity(
                name=" neo4j ",
                entity_type=EntityType.TECHNOLOGY,
                description="Enterprise graph database",
                confidence=0.97,
            ),
        ],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        create_mock_completion(parsed_result)
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    result = extractor.extract_chunk(chunk)

    assert len(result.entities) == 1
    assert result.entities[0].confidence == 0.97
    assert result.entities[0].name == "neo4j"


def test_duplicate_relationships_keep_highest_confidence() -> None:
    """Duplicate relationships should be consolidated."""
    chunk = create_chunk()

    entities = [
        GraphEntity(
            name="Enterprise GraphRAG Assistant",
            entity_type=EntityType.PROJECT,
        ),
        GraphEntity(
            name="Neo4j",
            entity_type=EntityType.TECHNOLOGY,
        ),
    ]

    parsed_result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=entities,
        relationships=[
            GraphRelationship(
                source_name="Enterprise GraphRAG Assistant",
                source_type=EntityType.PROJECT,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence="The project uses Neo4j.",
                confidence=0.75,
            ),
            GraphRelationship(
                source_name="Enterprise GraphRAG Assistant",
                source_type=EntityType.PROJECT,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence="The project uses Neo4j.",
                confidence=0.96,
            ),
        ],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        create_mock_completion(parsed_result)
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    result = extractor.extract_chunk(chunk)

    assert len(result.relationships) == 1
    assert result.relationships[0].confidence == 0.96


def test_relationship_with_missing_endpoint_is_removed() -> None:
    """Relationships must reference entities in the entity list."""
    chunk = create_chunk()

    parsed_result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=[
            GraphEntity(
                name="Enterprise GraphRAG Assistant",
                entity_type=EntityType.PROJECT,
            )
        ],
        relationships=[
            GraphRelationship(
                source_name="Enterprise GraphRAG Assistant",
                source_type=EntityType.PROJECT,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence="The project uses Neo4j.",
            )
        ],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        create_mock_completion(parsed_result)
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    result = extractor.extract_chunk(chunk)

    assert len(result.entities) == 1
    assert result.relationships == []


def test_result_chunk_id_is_normalized() -> None:
    """The output must use the actual input chunk identifier."""
    chunk = create_chunk(chunk_id="chunk-correct-id")

    parsed_result = ExtractionResult(
        chunk_id="chunk-wrong-model-id",
        entities=[],
        relationships=[],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        create_mock_completion(parsed_result)
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    result = extractor.extract_chunk(chunk)

    assert result.chunk_id == "chunk-correct-id"


def test_empty_completion_choices_raise_error() -> None:
    """An empty model response should produce a clear error."""
    chunk = create_chunk()

    empty_completion = MagicMock()
    empty_completion.choices = []

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        empty_completion
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    with pytest.raises(
        EntityExtractionError,
        match="no completion choices",
    ):
        extractor.extract_chunk(chunk)


def test_model_refusal_raises_error() -> None:
    """A refusal should be converted into an extraction error."""
    chunk = create_chunk()

    message = MagicMock()
    message.refusal = "Unable to process this content."
    message.parsed = None

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        completion
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    with pytest.raises(
        EntityExtractionError,
        match="request was refused",
    ):
        extractor.extract_chunk(chunk)


def test_missing_parsed_result_raises_error() -> None:
    """A response without parsed structured data should fail."""
    chunk = create_chunk()

    message = MagicMock()
    message.refusal = None
    message.parsed = None

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = (
        completion
    )

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    with pytest.raises(
        EntityExtractionError,
        match="no parsed extraction result",
    ):
        extractor.extract_chunk(chunk)


def test_extract_multiple_chunks() -> None:
    """The extractor should process multiple chunks in order."""
    first_chunk = create_chunk(
        chunk_id="chunk-one",
        text="Alex Morgan manages the project.",
    )
    second_chunk = create_chunk(
        chunk_id="chunk-two",
        text="The project uses Neo4j.",
    )

    first_result = ExtractionResult(
        chunk_id=first_chunk.chunk_id,
        entities=[],
        relationships=[],
    )
    second_result = ExtractionResult(
        chunk_id=second_chunk.chunk_id,
        entities=[],
        relationships=[],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        create_mock_completion(first_result),
        create_mock_completion(second_result),
    ]

    extractor = EntityExtractor(
        api_key="test-api-key",
        model="test-model",
        client=mock_client,
    )

    results = extractor.extract_chunks(
        [first_chunk, second_chunk]
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-one"
    assert results[1].chunk_id == "chunk-two"

    assert (
        mock_client.chat.completions.parse.call_count
        == 2
    )


def test_missing_api_key_is_rejected() -> None:
    """An API key is required when creating an extractor."""
    with pytest.raises(
        ValueError,
        match="API key is required",
    ):
        EntityExtractor(
            api_key="",
            model="test-model",
        )


def test_invalid_retry_count_is_rejected() -> None:
    """Retry count cannot be negative."""
    with pytest.raises(
        ValueError,
        match="max_retries cannot be negative",
    ):
        EntityExtractor(
            api_key="test-api-key",
            model="test-model",
            max_retries=-1,
        )
"""Unit tests for building extracted knowledge in Neo4j."""

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.graph_builder import (
    GraphBuildError,
    KnowledgeGraphBuilder,
)
from app.ingestion.models import (
    DocumentType,
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelationship,
    LoadedDocument,
    RelationshipType,
    TextChunk,
)


def create_document() -> LoadedDocument:
    """Create a source document for graph-builder tests."""
    return LoadedDocument(
        document_id="document-test-001",
        filename="project_overview.txt",
        file_type=DocumentType.TXT,
        source_path="data/sample_documents/project_overview.txt",
        content=(
            "The Enterprise GraphRAG Assistant uses Neo4j."
        ),
    )


def create_chunk(
    document_id: str = "document-test-001",
    chunk_id: str = "chunk-test-001",
) -> TextChunk:
    """Create a document chunk for graph-builder tests."""
    text = "The Enterprise GraphRAG Assistant uses Neo4j."

    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def create_extraction(
    chunk_id: str = "chunk-test-001",
) -> ExtractionResult:
    """Create a valid structured extraction result."""
    return ExtractionResult(
        chunk_id=chunk_id,
        entities=[
            GraphEntity(
                name="Enterprise GraphRAG Assistant",
                entity_type=EntityType.PROJECT,
                description="Enterprise knowledge graph assistant",
                confidence=0.98,
            ),
            GraphEntity(
                name="Neo4j",
                entity_type=EntityType.TECHNOLOGY,
                description="Graph database",
                confidence=0.99,
            ),
        ],
        relationships=[
            GraphRelationship(
                source_name="Enterprise GraphRAG Assistant",
                source_type=EntityType.PROJECT,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence=(
                    "The Enterprise GraphRAG Assistant uses Neo4j."
                ),
                confidence=0.99,
            )
        ],
        summary="The project uses Neo4j.",
    )


def configure_repository_mock(
    repository: MagicMock,
) -> None:
    """Configure successful repository responses."""
    repository.upsert_document.return_value = {
        "document": {
            "id": "document-test-001",
            "title": "project_overview.txt",
        }
    }

    repository.upsert_project.return_value = {
        "project": {
            "id": "project-test",
            "name": "Enterprise GraphRAG Assistant",
        }
    }

    repository.upsert_technology.return_value = {
        "technology": {
            "id": "technology-test",
            "name": "Neo4j",
        }
    }

    repository.upsert_person.return_value = {
        "person": {
            "id": "person-test",
            "name": "Alex Morgan",
        }
    }


def test_entity_id_is_stable() -> None:
    """Equivalent entity names should produce the same stable ID."""
    first_entity = GraphEntity(
        name="Neo4j",
        entity_type=EntityType.TECHNOLOGY,
    )

    second_entity = GraphEntity(
        name="neo4j",
        entity_type=EntityType.TECHNOLOGY,
    )

    first_id = KnowledgeGraphBuilder._create_entity_id(
        first_entity
    )
    second_id = KnowledgeGraphBuilder._create_entity_id(
        second_entity
    )

    assert first_id == second_id
    assert first_id.startswith("technology-neo4j-")


def test_build_rejects_empty_chunk_list() -> None:
    """At least one chunk must be supplied."""
    document = create_document()
    mock_client = MagicMock()

    builder = KnowledgeGraphBuilder(mock_client)

    with pytest.raises(
        GraphBuildError,
        match="At least one text chunk is required",
    ):
        builder.build_document_graph(
            document=document,
            chunks=[],
            extraction_results=[],
        )


def test_build_rejects_chunk_from_another_document() -> None:
    """Every chunk must belong to the supplied document."""
    document = create_document()

    invalid_chunk = create_chunk(
        document_id="different-document"
    )

    mock_client = MagicMock()
    builder = KnowledgeGraphBuilder(mock_client)

    with pytest.raises(
        GraphBuildError,
        match="belongs to a different document",
    ):
        builder.build_document_graph(
            document=document,
            chunks=[invalid_chunk],
            extraction_results=[],
        )


def test_build_rejects_duplicate_chunk_ids() -> None:
    """Duplicate chunk identifiers should be rejected."""
    document = create_document()
    chunk = create_chunk()

    mock_client = MagicMock()
    builder = KnowledgeGraphBuilder(mock_client)

    with pytest.raises(
        GraphBuildError,
        match="Duplicate chunk ID",
    ):
        builder.build_document_graph(
            document=document,
            chunks=[chunk, chunk],
            extraction_results=[],
        )


def test_build_rejects_unknown_extraction_chunk() -> None:
    """Extraction results must reference an existing chunk."""
    document = create_document()
    chunk = create_chunk()

    invalid_extraction = ExtractionResult(
        chunk_id="unknown-chunk",
        entities=[],
        relationships=[],
    )

    mock_client = MagicMock()
    builder = KnowledgeGraphBuilder(mock_client)

    with pytest.raises(
        GraphBuildError,
        match="unknown chunk",
    ):
        builder.build_document_graph(
            document=document,
            chunks=[chunk],
            extraction_results=[invalid_extraction],
        )


@patch(
    "app.ingestion.graph_builder.KnowledgeGraphRepository"
)
def test_build_document_graph_success(
    mock_repository_class: MagicMock,
) -> None:
    """A valid extraction should create graph data successfully."""
    document = create_document()
    chunk = create_chunk()
    extraction = create_extraction()

    mock_client = MagicMock()

    # Direct Cypher operations must return at least one record.
    mock_client.execute_query.return_value = [
        {"result": "success"}
    ]

    mock_repository = mock_repository_class.return_value
    configure_repository_mock(mock_repository)

    builder = KnowledgeGraphBuilder(mock_client)

    summary = builder.build_document_graph(
        document=document,
        chunks=[chunk],
        extraction_results=[extraction],
    )

    assert summary == {
        "document_id": "document-test-001",
        "filename": "project_overview.txt",
        "chunks_written": 1,
        "entities_written": 2,
        "mentions_written": 2,
        "relationships_written": 1,
    }

    mock_repository.upsert_document.assert_called_once_with(
        document_id=document.document_id,
        title=document.filename,
        source=document.source_path,
        content=document.content,
    )

    mock_repository.upsert_project.assert_called_once()
    mock_repository.upsert_technology.assert_called_once()

    executed_queries = [
        call.args[0]
        for call in mock_client.execute_query.call_args_list
    ]

    assert any(
        "HAS_CHUNK" in query
        for query in executed_queries
    )

    assert any(
        "MENTIONS" in query
        for query in executed_queries
    )

    assert any(
        ":USES" in query
        for query in executed_queries
    )


@patch(
    "app.ingestion.graph_builder.KnowledgeGraphRepository"
)
def test_invalid_relationship_direction_is_skipped(
    mock_repository_class: MagicMock,
) -> None:
    """Relationships with unsupported directions should be skipped."""
    document = create_document()
    chunk = create_chunk()

    extraction = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=[
            GraphEntity(
                name="Alex Morgan",
                entity_type=EntityType.PERSON,
            ),
            GraphEntity(
                name="Neo4j",
                entity_type=EntityType.TECHNOLOGY,
            ),
        ],
        relationships=[
            GraphRelationship(
                source_name="Alex Morgan",
                source_type=EntityType.PERSON,
                relationship_type=RelationshipType.USES,
                target_name="Neo4j",
                target_type=EntityType.TECHNOLOGY,
                evidence="Alex Morgan uses Neo4j.",
                confidence=0.90,
            )
        ],
    )

    mock_client = MagicMock()
    mock_client.execute_query.return_value = [
        {"result": "success"}
    ]

    mock_repository = mock_repository_class.return_value
    configure_repository_mock(mock_repository)

    builder = KnowledgeGraphBuilder(mock_client)

    summary = builder.build_document_graph(
        document=document,
        chunks=[chunk],
        extraction_results=[extraction],
    )

    assert summary["entities_written"] == 2
    assert summary["mentions_written"] == 2
    assert summary["relationships_written"] == 0

    executed_queries = [
        call.args[0]
        for call in mock_client.execute_query.call_args_list
    ]

    assert not any(
        ":USES" in query
        for query in executed_queries
    )
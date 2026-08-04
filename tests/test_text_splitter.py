"""Unit tests for the overlapping text splitter."""

import pytest

from app.ingestion.models import (
    DocumentType,
    LoadedDocument,
)
from app.ingestion.text_splitter import TextSplitter


def create_document(content: str) -> LoadedDocument:
    """Create a validated document for splitter tests."""
    return LoadedDocument(
        document_id="document-test",
        filename="test.txt",
        file_type=DocumentType.TXT,
        source_path="data/test.txt",
        content=content,
    )


def test_short_document_creates_one_chunk() -> None:
    """A document below the size limit should remain intact."""
    document = create_document(
        "Neo4j powers the enterprise knowledge graph."
    )

    splitter = TextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    chunks = splitter.split_document(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == document.content
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_long_document_creates_multiple_chunks() -> None:
    """A large document should be divided into multiple chunks."""
    content = " ".join(
        [
            (
                "The Enterprise GraphRAG Assistant uses Neo4j "
                "and Python to retrieve connected data."
            )
            for _ in range(20)
        ]
    )

    document = create_document(content)

    splitter = TextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        minimum_chunk_size=40,
    )

    chunks = splitter.split_document(document)

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.document_id == document.document_id
        assert chunk.text
        assert chunk.end_char > chunk.start_char


def test_chunks_have_stable_ids() -> None:
    """Splitting the same document should produce identical IDs."""
    document = create_document(
        " ".join(
            ["Knowledge graph data"] * 100
        )
    )

    splitter = TextSplitter(
        chunk_size=150,
        chunk_overlap=25,
    )

    first_result = splitter.split_document(document)
    second_result = splitter.split_document(document)

    first_ids = [
        chunk.chunk_id
        for chunk in first_result
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_result
    ]

    assert first_ids == second_ids


def test_invalid_overlap_is_rejected() -> None:
    """Overlap cannot equal or exceed the chunk size."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller",
    ):
        TextSplitter(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_multiple_documents_are_supported() -> None:
    """The splitter should process multiple documents."""
    documents = [
        create_document(
            "Alex Morgan manages the GraphRAG project."
        ),
        LoadedDocument(
            document_id="document-test-two",
            filename="second.txt",
            file_type=DocumentType.TXT,
            source_path="data/second.txt",
            content=(
                "The project uses Neo4j and Python."
            ),
        ),
    ]

    splitter = TextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    chunks = splitter.split_documents(documents)

    assert len(chunks) == 2

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    assert document_ids == {
        "document-test",
        "document-test-two",
    }
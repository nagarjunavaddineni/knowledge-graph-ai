"""Unit tests for Streamlit document-upload utilities."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.retrieval.chunk_embedding_manager import (
    ChunkForEmbedding,
)
from app.services.document_ingestion_service import (
    DocumentIngestionResult,
    DocumentPreview,
)
from app.ui import document_page
from app.ui.constants import (
    SESSION_UPLOADED_DOCUMENTS_KEY,
)
from app.ui.document_page import (
    DocumentPageError,
    batched,
    embed_document_chunks,
    format_file_size,
    save_uploaded_file,
    store_ingestion_history,
)


def create_preview(
    *,
    document_id: str = "document-001",
    filename: str = "project_overview.txt",
) -> DocumentPreview:
    """Create a document preview for testing."""
    return DocumentPreview(
        document_id=document_id,
        filename=filename,
        document_type="txt",
        file_size_bytes=2048,
        character_count=1500,
        chunk_count=3,
    )


def create_ingestion_result(
    *,
    document_id: str = "document-001",
    filename: str = "project_overview.txt",
) -> DocumentIngestionResult:
    """Create a completed ingestion result."""
    return DocumentIngestionResult(
        preview=create_preview(
            document_id=document_id,
            filename=filename,
        ),
        chunks_written=3,
        entities_written=5,
        mentions_written=7,
        relationships_written=4,
        extracted_entity_count=5,
        extracted_relationship_count=4,
    )


def test_save_uploaded_file_preserves_content() -> None:
    """Uploaded bytes should be written without modification."""
    content = b"Enterprise GraphRAG Assistant uses Neo4j."

    temporary_directory, temporary_path = (
        save_uploaded_file(
            filename="project_overview.txt",
            content=content,
        )
    )

    try:
        assert temporary_directory.exists()
        assert temporary_directory.is_dir()

        assert temporary_path.exists()
        assert temporary_path.is_file()
        assert temporary_path.name == (
            "project_overview.txt"
        )

        assert temporary_path.read_bytes() == content

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


def test_save_uploaded_file_removes_parent_path() -> None:
    """Directory traversal components should be removed."""
    temporary_directory, temporary_path = (
        save_uploaded_file(
            filename="../unsafe_document.txt",
            content=b"Safe content",
        )
    )

    try:
        assert temporary_path.parent == (
            temporary_directory
        )

        assert temporary_path.name == (
            "unsafe_document.txt"
        )

        assert ".." not in temporary_path.parts

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


def test_save_uploaded_file_accepts_supported_formats() -> None:
    """Every configured document format should be accepted."""
    filenames = [
        "document.txt",
        "document.pdf",
        "document.csv",
        "document.json",
    ]

    created_directories: list[Path] = []

    try:
        for filename in filenames:
            directory, path = save_uploaded_file(
                filename=filename,
                content=b"test",
            )

            created_directories.append(directory)

            assert path.exists()
            assert path.name == filename

    finally:
        for directory in created_directories:
            shutil.rmtree(
                directory,
                ignore_errors=True,
            )


def test_file_extension_check_is_case_insensitive() -> None:
    """Uppercase supported extensions should be accepted."""
    temporary_directory, temporary_path = (
        save_uploaded_file(
            filename="PROJECT_OVERVIEW.TXT",
            content=b"test",
        )
    )

    try:
        assert temporary_path.exists()
        assert temporary_path.suffix == ".TXT"

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


def test_unsupported_extension_is_rejected() -> None:
    """Unsupported uploaded file types should fail clearly."""
    with pytest.raises(
        DocumentPageError,
        match="Unsupported file extension",
    ):
        save_uploaded_file(
            filename="malware.exe",
            content=b"invalid",
        )


def test_missing_filename_is_rejected() -> None:
    """An uploaded file must have a usable filename."""
    with pytest.raises(
        DocumentPageError,
        match="does not have a valid name",
    ):
        save_uploaded_file(
            filename="   ",
            content=b"content",
        )


def test_batched_splits_items_into_fixed_sizes() -> None:
    """The batching helper should preserve item order."""
    result = list(
        batched(
            items=[1, 2, 3, 4, 5],
            batch_size=2,
        )
    )

    assert result == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_batched_handles_exact_batch_size() -> None:
    """Exact multiples should not create an empty batch."""
    result = list(
        batched(
            items=["a", "b", "c", "d"],
            batch_size=2,
        )
    )

    assert result == [
        ["a", "b"],
        ["c", "d"],
    ]


def test_batched_handles_empty_collection() -> None:
    """An empty collection should produce no batches."""
    result = list(
        batched(
            items=[],
            batch_size=10,
        )
    )

    assert result == []


@pytest.mark.parametrize(
    "invalid_batch_size",
    [0, -1],
)
def test_invalid_batch_size_is_rejected(
    invalid_batch_size: int,
) -> None:
    """Batch sizes must be positive."""
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        list(
            batched(
                items=[1, 2],
                batch_size=invalid_batch_size,
            )
        )


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 bytes"),
        (500, "500 bytes"),
        (1023, "1023 bytes"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 * 1024, "1.00 MB"),
        (2 * 1024 * 1024, "2.00 MB"),
    ],
)
def test_format_file_size(
    size_bytes: int,
    expected: str,
) -> None:
    """Byte counts should be displayed in readable units."""
    assert format_file_size(size_bytes) == expected


def test_store_ingestion_history_adds_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed ingestion should be stored in session state."""
    session_state: dict[str, object] = {}

    monkeypatch.setattr(
        document_page.st,
        "session_state",
        session_state,
    )

    ingestion_result = create_ingestion_result()

    store_ingestion_history(
        ingestion_result=ingestion_result,
        embeddings_written=3,
    )

    history = session_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ]

    assert isinstance(history, list)
    assert len(history) == 1

    record = history[0]

    assert record["document_id"] == "document-001"
    assert record["filename"] == (
        "project_overview.txt"
    )
    assert record["chunks_written"] == 3
    assert record["entities_written"] == 5
    assert record["relationships_written"] == 4
    assert record["embeddings_written"] == 3


def test_store_ingestion_history_resets_invalid_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid session history should be replaced safely."""
    session_state: dict[str, object] = {
        SESSION_UPLOADED_DOCUMENTS_KEY: (
            "invalid-history"
        )
    }

    monkeypatch.setattr(
        document_page.st,
        "session_state",
        session_state,
    )

    store_ingestion_history(
        ingestion_result=create_ingestion_result(),
        embeddings_written=2,
    )

    history = session_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ]

    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["embeddings_written"] == 2


def test_store_ingestion_history_keeps_latest_twenty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session history should retain only recent ingestions."""
    existing_history = [
        {
            "document_id": f"document-{index:03d}",
            "filename": f"document-{index:03d}.txt",
        }
        for index in range(20)
    ]

    session_state: dict[str, object] = {
        SESSION_UPLOADED_DOCUMENTS_KEY: (
            existing_history
        )
    }

    monkeypatch.setattr(
        document_page.st,
        "session_state",
        session_state,
    )

    store_ingestion_history(
        ingestion_result=create_ingestion_result(
            document_id="document-020",
            filename="document-020.txt",
        ),
        embeddings_written=1,
    )

    history = session_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ]

    assert isinstance(history, list)
    assert len(history) == 20

    assert history[0]["document_id"] == (
        "document-001"
    )

    assert history[-1]["document_id"] == (
        "document-020"
    )


def test_embed_document_chunks_returns_zero_when_not_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No OpenAI request is needed when no chunks qualify."""
    manager = MagicMock()

    manager.get_chunks_requiring_embeddings.return_value = (
        []
    )

    manager_factory = MagicMock()
    manager_factory.from_settings.return_value = manager

    embedding_factory = MagicMock()

    monkeypatch.setattr(
        document_page,
        "ChunkEmbeddingManager",
        manager_factory,
    )

    monkeypatch.setattr(
        document_page,
        "EmbeddingService",
        embedding_factory,
    )

    status = MagicMock()
    client = MagicMock()

    result = embed_document_chunks(
        client=client,
        document_id="document-001",
        status=status,
    )

    assert result == 0

    manager_factory.from_settings.assert_called_once_with(
        client
    )

    manager.get_chunks_requiring_embeddings.assert_called_once_with(
        force=False
    )

    embedding_factory.from_settings.assert_not_called()
    manager.store_embeddings.assert_not_called()

    status.write.assert_called_once_with(
        "No new chunk embeddings were required."
    )


def test_embed_document_chunks_filters_by_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only chunks belonging to the selected document are embedded."""
    manager = MagicMock()

    manager.get_chunks_requiring_embeddings.return_value = [
        ChunkForEmbedding(
            chunk_id="chunk-001",
            document_id="document-001",
            chunk_index=0,
            text="Neo4j stores connected information.",
        ),
        ChunkForEmbedding(
            chunk_id="chunk-002",
            document_id="document-002",
            chunk_index=0,
            text="This belongs to another document.",
        ),
    ]

    manager.store_embeddings.return_value = 1

    embedding_service = MagicMock()

    embedding_service.embed_texts.return_value = [
        [0.1, 0.2, 0.3],
    ]

    manager_factory = MagicMock()
    manager_factory.from_settings.return_value = manager

    embedding_factory = MagicMock()

    embedding_factory.from_settings.return_value = (
        embedding_service
    )

    monkeypatch.setattr(
        document_page,
        "ChunkEmbeddingManager",
        manager_factory,
    )

    monkeypatch.setattr(
        document_page,
        "EmbeddingService",
        embedding_factory,
    )

    status = MagicMock()

    result = embed_document_chunks(
        client=MagicMock(),
        document_id="document-001",
        status=status,
    )

    assert result == 1

    embedding_service.embed_texts.assert_called_once_with(
        [
            "Neo4j stores connected information.",
        ]
    )

    manager.store_embeddings.assert_called_once_with(
        [
            {
                "chunk_id": "chunk-001",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    )


def test_embed_document_chunks_processes_multiple_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large document embeddings should be written in batches."""
    chunks = [
        ChunkForEmbedding(
            chunk_id=f"chunk-{index:03d}",
            document_id="document-001",
            chunk_index=index,
            text=f"Chunk text {index}",
        )
        for index in range(5)
    ]

    manager = MagicMock()

    manager.get_chunks_requiring_embeddings.return_value = (
        chunks
    )

    manager.store_embeddings.side_effect = [
        2,
        2,
        1,
    ]

    embedding_service = MagicMock()

    embedding_service.embed_texts.side_effect = [
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        [
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        [
            [1.3, 1.4, 1.5],
        ],
    ]

    manager_factory = MagicMock()
    manager_factory.from_settings.return_value = manager

    embedding_factory = MagicMock()

    embedding_factory.from_settings.return_value = (
        embedding_service
    )

    monkeypatch.setattr(
        document_page,
        "ChunkEmbeddingManager",
        manager_factory,
    )

    monkeypatch.setattr(
        document_page,
        "EmbeddingService",
        embedding_factory,
    )

    monkeypatch.setattr(
        document_page,
        "EMBEDDING_BATCH_SIZE",
        2,
    )

    status = MagicMock()

    result = embed_document_chunks(
        client=MagicMock(),
        document_id="document-001",
        status=status,
    )

    assert result == 5

    assert embedding_service.embed_texts.call_count == 3
    assert manager.store_embeddings.call_count == 3

    first_text_batch = (
        embedding_service.embed_texts
        .call_args_list[0]
        .args[0]
    )

    second_text_batch = (
        embedding_service.embed_texts
        .call_args_list[1]
        .args[0]
    )

    third_text_batch = (
        embedding_service.embed_texts
        .call_args_list[2]
        .args[0]
    )

    assert first_text_batch == [
        "Chunk text 0",
        "Chunk text 1",
    ]

    assert second_text_batch == [
        "Chunk text 2",
        "Chunk text 3",
    ]

    assert third_text_batch == [
        "Chunk text 4",
    ]

    status_messages = [
        call.args[0]
        for call in status.write.call_args_list
    ]

    assert status_messages == [
        "Embedded batch 1/3: 2 chunks.",
        "Embedded batch 2/3: 2 chunks.",
        "Embedded batch 3/3: 1 chunks.",
    ]
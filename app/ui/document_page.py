"""Streamlit page for document preview and graph ingestion."""

import logging
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import streamlit as st

from app.database.neo4j_client import Neo4jClient
from app.database.retrieval_indexes import (
    RetrievalIndexError,
    RetrievalIndexManager,
)
from app.retrieval.chunk_embedding_manager import (
    ChunkEmbeddingError,
    ChunkEmbeddingManager,
    ChunkForEmbedding,
)
from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
    EmbeddingService,
)
from app.services.document_ingestion_service import (
    DocumentIngestionResult,
    DocumentIngestionService,
    DocumentIngestionServiceError,
    DocumentPreview,
)
from app.ui.constants import (
    MAX_UPLOAD_SIZE_MB,
    SESSION_UPLOADED_DOCUMENTS_KEY,
    SUPPORTED_UPLOAD_TYPES,
)


logger = logging.getLogger(__name__)

ItemType = TypeVar("ItemType")

EMBEDDING_BATCH_SIZE = 50


class DocumentPageError(RuntimeError):
    """Raised when UI document ingestion cannot be completed."""


def render_document_page() -> None:
    """Render document upload, preview, and ingestion controls."""
    st.subheader("Upload and ingest a document")

    st.caption(
        "Preview a TXT, PDF, CSV, or JSON document before "
        "extracting entities and relationships into Neo4j."
    )

    uploaded_file = st.file_uploader(
        label="Select a document",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=False,
        key="knowledge_graph_document_upload",
        max_upload_size=MAX_UPLOAD_SIZE_MB,
        help=(
            f"Supported formats: "
            f"{', '.join(SUPPORTED_UPLOAD_TYPES).upper()}. "
            f"Maximum size: {MAX_UPLOAD_SIZE_MB} MB."
        ),
    )

    if uploaded_file is None:
        render_recent_ingestions()
        return

    uploaded_bytes = uploaded_file.getvalue()
    uploaded_size = len(uploaded_bytes)

    if uploaded_size == 0:
        st.error("The uploaded file is empty.")
        render_recent_ingestions()
        return

    maximum_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if uploaded_size > maximum_bytes:
        st.error(
            f"The file exceeds the {MAX_UPLOAD_SIZE_MB} MB "
            "application limit."
        )
        render_recent_ingestions()
        return

    temporary_directory: Path | None = None

    try:
        temporary_directory, temporary_path = (
            save_uploaded_file(
                filename=uploaded_file.name,
                content=uploaded_bytes,
            )
        )

        preview_service = (
            DocumentIngestionService.for_preview()
        )

        preview = preview_service.preview_document(
            temporary_path
        )

        render_document_preview(preview)

        ingest_clicked = st.button(
            "Ingest into Neo4j",
            type="primary",
            use_container_width=True,
            key="ingest_uploaded_document",
        )

        if ingest_clicked:
            process_uploaded_document(
                temporary_path=temporary_path,
            )

    except DocumentIngestionServiceError as error:
        logger.exception(
            "Unable to preview uploaded document."
        )

        st.error(
            "The uploaded document could not be loaded "
            "or split into text chunks."
        )

        st.caption(str(error))

    except OSError as error:
        logger.exception(
            "Unable to create a temporary upload file."
        )

        st.error(
            "The uploaded file could not be prepared "
            "for processing."
        )

        st.caption(str(error))

    finally:
        if temporary_directory is not None:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

    render_recent_ingestions()


def save_uploaded_file(
    filename: str,
    content: bytes,
) -> tuple[Path, Path]:
    """Save uploaded bytes to a temporary directory."""
    safe_filename = Path(filename).name.strip()

    if not safe_filename:
        raise DocumentPageError(
            "The uploaded file does not have a valid name."
        )

    suffix = Path(safe_filename).suffix.lower()

    supported_suffixes = {
        f".{file_type}"
        for file_type in SUPPORTED_UPLOAD_TYPES
    }

    if suffix not in supported_suffixes:
        raise DocumentPageError(
            f"Unsupported file extension: {suffix}"
        )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="knowledge_graph_upload_"
        )
    )

    temporary_path = (
        temporary_directory / safe_filename
    )

    temporary_path.write_bytes(content)

    return temporary_directory, temporary_path


def render_document_preview(
    preview: DocumentPreview,
) -> None:
    """Display document metadata before ingestion."""
    st.success(
        "The document was loaded and split successfully."
    )

    st.markdown("#### Document preview")

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="File",
            value=preview.filename,
        )

    with second:
        st.metric(
            label="Type",
            value=preview.document_type.upper(),
        )

    with third:
        st.metric(
            label="Text chunks",
            value=preview.chunk_count,
        )

    with fourth:
        st.metric(
            label="Characters",
            value=f"{preview.character_count:,}",
        )

    with st.expander(
        "Document details",
        expanded=False,
        icon="📄",
    ):
        st.write(
            f"**Document ID:** `{preview.document_id}`"
        )

        st.write(
            f"**File size:** "
            f"{format_file_size(preview.file_size_bytes)}"
        )

        st.write(
            f"**Characters:** "
            f"{preview.character_count:,}"
        )

        st.write(
            f"**Generated chunks:** "
            f"{preview.chunk_count:,}"
        )

    st.info(
        "Ingestion sends each text chunk to the entity "
        "extractor, writes the resulting graph to Neo4j, "
        "and generates vector embeddings."
    )


def process_uploaded_document(
    temporary_path: Path,
) -> None:
    """Ingest one uploaded document and embed its chunks."""
    with st.status(
        "Preparing document ingestion...",
        expanded=True,
    ) as status:
        try:
            status.write(
                "Connecting to Neo4j."
            )

            with Neo4jClient.from_settings() as client:
                if not client.verify_connection():
                    raise DocumentPageError(
                        "Neo4j connection verification failed."
                    )

                status.write(
                    "Extracting document entities "
                    "and relationships."
                )

                ingestion_service = (
                    DocumentIngestionService.from_settings()
                )

                ingestion_result = (
                    ingestion_service.ingest_document(
                        file_path=temporary_path,
                        client=client,
                    )
                )

                status.write(
                    "Creating and verifying retrieval indexes."
                )

                index_manager = (
                    RetrievalIndexManager.from_settings(
                        client
                    )
                )

                index_manager.create_indexes()

                indexes = index_manager.wait_until_online(
                    timeout_seconds=180,
                    poll_interval_seconds=1,
                )

                index_manager.verify_index_definitions(
                    indexes
                )

                status.write(
                    "Generating embeddings for "
                    "the new document chunks."
                )

                embeddings_written = (
                    embed_document_chunks(
                        client=client,
                        document_id=(
                            ingestion_result
                            .preview
                            .document_id
                        ),
                        status=status,
                    )
                )

            status.update(
                label="Document ingestion completed",
                state="complete",
                expanded=False,
            )

            store_ingestion_history(
                ingestion_result=ingestion_result,
                embeddings_written=embeddings_written,
            )

            render_ingestion_result(
                ingestion_result=ingestion_result,
                embeddings_written=embeddings_written,
            )

            st.balloons()

        except (
            DocumentIngestionServiceError,
            RetrievalIndexError,
            EmbeddingGenerationError,
            ChunkEmbeddingError,
            DocumentPageError,
        ) as error:
            logger.exception(
                "Document ingestion failed."
            )

            status.update(
                label="Document ingestion failed",
                state="error",
                expanded=True,
            )

            st.error(
                "The document could not be completely "
                "ingested into the knowledge graph."
            )

            st.caption(str(error))

        except ValueError as error:
            logger.exception(
                "Document ingestion configuration failed."
            )

            status.update(
                label="Configuration error",
                state="error",
                expanded=True,
            )

            st.error(
                "Review the Neo4j and OpenAI settings "
                "in your local `.env` file."
            )

            st.caption(str(error))

        except Exception as error:
            logger.exception(
                "Unexpected document-ingestion failure."
            )

            status.update(
                label="Unexpected ingestion failure",
                state="error",
                expanded=True,
            )

            st.error(
                "An unexpected error occurred during "
                "document ingestion."
            )

            st.caption(
                f"{type(error).__name__}: {error}"
            )


def embed_document_chunks(
    client: Neo4jClient,
    document_id: str,
    status: Any,
) -> int:
    """Generate embeddings for missing chunks in one document."""
    manager = ChunkEmbeddingManager.from_settings(
        client
    )

    candidates = (
        manager.get_chunks_requiring_embeddings(
            force=False
        )
    )

    document_chunks = [
        chunk
        for chunk in candidates
        if chunk.document_id == document_id
    ]

    if not document_chunks:
        status.write(
            "No new chunk embeddings were required."
        )
        return 0

    embedding_service = (
        EmbeddingService.from_settings()
    )

    total_updated = 0

    chunk_batches = list(
        batched(
            document_chunks,
            EMBEDDING_BATCH_SIZE,
        )
    )

    for batch_number, chunk_batch in enumerate(
        chunk_batches,
        start=1,
    ):
        texts = [
            chunk.text
            for chunk in chunk_batch
        ]

        vectors = embedding_service.embed_texts(
            texts
        )

        embedding_items = [
            {
                "chunk_id": chunk.chunk_id,
                "embedding": vector,
            }
            for chunk, vector in zip(
                chunk_batch,
                vectors,
                strict=True,
            )
        ]

        updated_count = manager.store_embeddings(
            embedding_items
        )

        total_updated += updated_count

        status.write(
            f"Embedded batch {batch_number}/"
            f"{len(chunk_batches)}: "
            f"{updated_count} chunks."
        )

    return total_updated


def batched(
    items: Sequence[ItemType],
    batch_size: int,
) -> Iterator[Sequence[ItemType]]:
    """Yield fixed-size sequences."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    for start_index in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start_index:
            start_index + batch_size
        ]


def render_ingestion_result(
    ingestion_result: DocumentIngestionResult,
    embeddings_written: int,
) -> None:
    """Display the completed graph-ingestion summary."""
    st.success(
        f"`{ingestion_result.preview.filename}` "
        "was added to the knowledge graph."
    )

    st.markdown("#### Ingestion results")

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="Chunks written",
            value=ingestion_result.chunks_written,
        )

    with second:
        st.metric(
            label="Entities written",
            value=ingestion_result.entities_written,
        )

    with third:
        st.metric(
            label="Relationships",
            value=(
                ingestion_result.relationships_written
            ),
        )

    with fourth:
        st.metric(
            label="Embeddings",
            value=embeddings_written,
        )

    with st.expander(
        "Complete ingestion details",
        expanded=False,
        icon="✅",
    ):
        st.json(
            {
                **ingestion_result.as_dict(),
                "embeddings_written": (
                    embeddings_written
                ),
            }
        )


def store_ingestion_history(
    ingestion_result: DocumentIngestionResult,
    embeddings_written: int,
) -> None:
    """Store an ingestion result in Streamlit session state."""
    history = st.session_state.get(
        SESSION_UPLOADED_DOCUMENTS_KEY,
        [],
    )

    if not isinstance(history, list):
        history = []

    record = {
        **ingestion_result.as_dict(),
        "embeddings_written": embeddings_written,
    }

    history.append(record)

    st.session_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ] = history[-20:]


def render_recent_ingestions() -> None:
    """Display recent ingestions from the current UI session."""
    history = st.session_state.get(
        SESSION_UPLOADED_DOCUMENTS_KEY,
        [],
    )

    if not isinstance(history, list) or not history:
        return

    st.divider()
    st.markdown("#### Recent ingestions")

    for record in reversed(history[-5:]):
        if not isinstance(record, dict):
            continue

        filename = str(
            record.get("filename")
            or "Unknown document"
        )

        chunk_count = int(
            record.get("chunks_written")
            or record.get("chunk_count")
            or 0
        )

        entity_count = int(
            record.get("entities_written")
            or 0
        )

        relationship_count = int(
            record.get("relationships_written")
            or 0
        )

        embedding_count = int(
            record.get("embeddings_written")
            or 0
        )

        with st.expander(
            filename,
            expanded=False,
            icon="📄",
        ):
            first, second = st.columns(2)

            with first:
                st.write(
                    f"**Chunks:** {chunk_count}"
                )

                st.write(
                    f"**Entities:** {entity_count}"
                )

            with second:
                st.write(
                    "**Relationships:** "
                    f"{relationship_count}"
                )

                st.write(
                    f"**Embeddings:** "
                    f"{embedding_count}"
                )

            document_id = record.get(
                "document_id"
            )

            if document_id:
                st.caption(
                    f"Document ID: {document_id}"
                )


def format_file_size(
    size_bytes: int,
) -> str:
    """Convert a byte count into a readable value."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"

    size_kb = size_bytes / 1024

    if size_kb < 1024:
        return f"{size_kb:.1f} KB"

    size_mb = size_kb / 1024

    return f"{size_mb:.2f} MB"
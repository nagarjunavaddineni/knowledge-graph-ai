"""Reusable document-ingestion service for the Streamlit UI."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.database.neo4j_client import Neo4jClient
from app.database.schema import GraphSchemaManager
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.entity_extractor import EntityExtractor
from app.ingestion.graph_builder import KnowledgeGraphBuilder
from app.ingestion.models import (
    ExtractionResult,
    LoadedDocument,
    TextChunk,
)
from app.ingestion.text_splitter import TextSplitter


logger = logging.getLogger(__name__)


class DocumentIngestionServiceError(RuntimeError):
    """Raised when document ingestion cannot be completed."""


@dataclass(frozen=True)
class DocumentPreview:
    """Basic information about a loaded and split document."""

    document_id: str
    filename: str
    document_type: str
    file_size_bytes: int
    character_count: int
    chunk_count: int


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Summary returned after writing a document graph."""

    preview: DocumentPreview
    chunks_written: int
    entities_written: int
    mentions_written: int
    relationships_written: int
    extracted_entity_count: int
    extracted_relationship_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary."""
        return {
            "document_id": self.preview.document_id,
            "filename": self.preview.filename,
            "document_type": self.preview.document_type,
            "file_size_bytes": self.preview.file_size_bytes,
            "character_count": self.preview.character_count,
            "chunk_count": self.preview.chunk_count,
            "chunks_written": self.chunks_written,
            "entities_written": self.entities_written,
            "mentions_written": self.mentions_written,
            "relationships_written": (
                self.relationships_written
            ),
            "extracted_entity_count": (
                self.extracted_entity_count
            ),
            "extracted_relationship_count": (
                self.extracted_relationship_count
            ),
        }


class DocumentIngestionService:
    """Coordinate loading, splitting, extraction, and graph writing."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        text_splitter: TextSplitter,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        """Initialize the ingestion service."""
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.entity_extractor = entity_extractor

    @classmethod
    def for_preview(cls) -> "DocumentIngestionService":
        """Create a service that only loads and splits documents."""
        return cls(
            document_loader=DocumentLoader(),
            text_splitter=TextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                minimum_chunk_size=80,
            ),
            entity_extractor=None,
        )

    @classmethod
    def from_settings(cls) -> "DocumentIngestionService":
        """Create the full ingestion service."""
        return cls(
            document_loader=DocumentLoader(),
            text_splitter=TextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                minimum_chunk_size=80,
            ),
            entity_extractor=(
                EntityExtractor.from_settings()
            ),
        )

    def preview_document(
        self,
        file_path: str | Path,
    ) -> DocumentPreview:
        """Load and split a document without calling OpenAI."""
        path = self._validate_file_path(file_path)

        try:
            document = self.document_loader.load(path)

            chunks = self.text_splitter.split_document(
                document
            )

            return self._create_preview(
                path=path,
                document=document,
                chunks=chunks,
            )

        except DocumentIngestionServiceError:
            raise

        except Exception as error:
            logger.exception(
                "Document preview failed for '%s'.",
                path,
            )

            raise DocumentIngestionServiceError(
                "Unable to load and preview the document."
            ) from error

    def ingest_document(
        self,
        file_path: str | Path,
        client: Neo4jClient,
    ) -> DocumentIngestionResult:
        """Extract a document and write it to Neo4j."""
        path = self._validate_file_path(file_path)

        if self.entity_extractor is None:
            raise DocumentIngestionServiceError(
                "The service was created for preview only. "
                "Use DocumentIngestionService.from_settings() "
                "for full ingestion."
            )

        try:
            document = self.document_loader.load(path)

            chunks = self.text_splitter.split_document(
                document
            )

            if not chunks:
                raise DocumentIngestionServiceError(
                    "The document did not produce any text chunks."
                )

            preview = self._create_preview(
                path=path,
                document=document,
                chunks=chunks,
            )

            extraction_results = (
                self._extract_all_chunks(chunks)
            )

            GraphSchemaManager(
                client
            ).create_constraints()

            graph_builder = KnowledgeGraphBuilder(
                client
            )

            graph_summary = (
                graph_builder.build_document_graph(
                    document=document,
                    chunks=chunks,
                    extraction_results=(
                        extraction_results
                    ),
                )
            )

            if not isinstance(graph_summary, dict):
                raise DocumentIngestionServiceError(
                    "The graph builder returned an "
                    "invalid ingestion summary."
                )

            extracted_entity_count = sum(
                len(result.entities)
                for result in extraction_results
            )

            extracted_relationship_count = sum(
                len(result.relationships)
                for result in extraction_results
            )

            result = DocumentIngestionResult(
                preview=preview,
                chunks_written=self._read_integer(
                    graph_summary,
                    "chunks_written",
                ),
                entities_written=self._read_integer(
                    graph_summary,
                    "entities_written",
                ),
                mentions_written=self._read_integer(
                    graph_summary,
                    "mentions_written",
                ),
                relationships_written=(
                    self._read_integer(
                        graph_summary,
                        "relationships_written",
                    )
                ),
                extracted_entity_count=(
                    extracted_entity_count
                ),
                extracted_relationship_count=(
                    extracted_relationship_count
                ),
            )

            logger.info(
                "Ingested '%s': %s chunks, %s entities, "
                "%s relationships.",
                result.preview.filename,
                result.chunks_written,
                result.entities_written,
                result.relationships_written,
            )

            return result

        except DocumentIngestionServiceError:
            raise

        except Exception as error:
            logger.exception(
                "Document ingestion failed for '%s'.",
                path,
            )

            raise DocumentIngestionServiceError(
                "Unable to extract and write the document "
                "to the knowledge graph."
            ) from error

    def _extract_all_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[ExtractionResult]:
        """Extract entities and relationships from every chunk."""
        if self.entity_extractor is None:
            raise DocumentIngestionServiceError(
                "The entity extractor is not configured."
            )

        extraction_results: list[
            ExtractionResult
        ] = []

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):
            logger.info(
                "Extracting chunk %s of %s.",
                chunk_number,
                len(chunks),
            )

            extraction = (
                self.entity_extractor.extract_chunk(
                    chunk
                )
            )

            extraction_results.append(extraction)

        return extraction_results

    @staticmethod
    def _create_preview(
        path: Path,
        document: LoadedDocument,
        chunks: list[TextChunk],
    ) -> DocumentPreview:
        """Create a document-preview summary."""
        document_id = (
            DocumentIngestionService._read_object_value(
                document,
                "document_id",
                "id",
            )
        )

        filename = (
            DocumentIngestionService._read_object_value(
                document,
                "filename",
                "file_name",
                "name",
            )
            or path.name
        )

        document_type_value = (
            DocumentIngestionService._read_raw_value(
                document,
                "document_type",
                "file_type",
                "type",
            )
        )

        if hasattr(document_type_value, "value"):
            document_type = str(
                document_type_value.value
            )
        elif document_type_value is not None:
            document_type = str(
                document_type_value
            )
        else:
            document_type = (
                path.suffix.lower().lstrip(".")
            )

        content = (
            DocumentIngestionService._read_object_value(
                document,
                "content",
                "text",
            )
        )

        return DocumentPreview(
            document_id=(
                document_id
                or path.stem
            ),
            filename=filename,
            document_type=document_type,
            file_size_bytes=path.stat().st_size,
            character_count=len(content),
            chunk_count=len(chunks),
        )

    @staticmethod
    def _validate_file_path(
        file_path: str | Path,
    ) -> Path:
        """Validate the source document path."""
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise DocumentIngestionServiceError(
                f"Document does not exist: {path}"
            )

        if not path.is_file():
            raise DocumentIngestionServiceError(
                f"Document path is not a file: {path}"
            )

        return path

    @staticmethod
    def _read_raw_value(
        item: Any,
        *field_names: str,
    ) -> Any:
        """Read the first available value from a model."""
        for field_name in field_names:
            if hasattr(item, field_name):
                value = getattr(
                    item,
                    field_name,
                )

                if value is not None:
                    return value

        if hasattr(item, "model_dump"):
            dumped_item = item.model_dump()

            for field_name in field_names:
                value = dumped_item.get(field_name)

                if value is not None:
                    return value

        return None

    @staticmethod
    def _read_object_value(
        item: Any,
        *field_names: str,
    ) -> str:
        """Read a model value and convert it to text."""
        value = (
            DocumentIngestionService._read_raw_value(
                item,
                *field_names,
            )
        )

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def _read_integer(
        values: dict[str, Any],
        key: str,
    ) -> int:
        """Read an integer from a graph summary."""
        value = values.get(key, 0)

        try:
            return int(value or 0)

        except (TypeError, ValueError) as error:
            raise DocumentIngestionServiceError(
                f"Invalid graph-summary value for '{key}'."
            ) from error
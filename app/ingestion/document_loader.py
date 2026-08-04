"""Load supported documents into validated ingestion models."""

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import (
    DocumentType,
    LoadedDocument,
    SourceProperty,
)


class UnsupportedDocumentTypeError(ValueError):
    """Raised when the supplied file type is not supported."""


class DocumentLoadError(RuntimeError):
    """Raised when document content cannot be loaded."""


class DocumentLoader:
    """Load TXT, PDF, CSV, and JSON documents."""

    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024

    SUPPORTED_TYPES = {
        ".txt": DocumentType.TXT,
        ".pdf": DocumentType.PDF,
        ".csv": DocumentType.CSV,
        ".json": DocumentType.JSON,
    }

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Load and validate a document from the local filesystem."""
        path = Path(file_path).expanduser().resolve()

        self._validate_path(path)

        document_type = self.SUPPORTED_TYPES.get(
            path.suffix.casefold()
        )

        if document_type is None:
            supported = ", ".join(
                sorted(self.SUPPORTED_TYPES)
            )
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {path.suffix}. "
                f"Supported types: {supported}"
            )

        try:
            content, format_metadata = self._load_by_type(
                path,
                document_type,
            )
        except (
            OSError,
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
        ) as error:
            raise DocumentLoadError(
                f"Unable to load document '{path.name}': {error}"
            ) from error

        normalized_content = self._normalize_text(content)

        if not normalized_content:
            raise DocumentLoadError(
                f"Document '{path.name}' contains no readable text."
            )

        file_statistics = path.stat()
        document_id = self._create_document_id(
            normalized_content
        )

        metadata = [
            SourceProperty(
                key="file_size_bytes",
                value=str(file_statistics.st_size),
            ),
            SourceProperty(
                key="last_modified_utc",
                value=datetime.fromtimestamp(
                    file_statistics.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            ),
            *format_metadata,
        ]

        return LoadedDocument(
            document_id=document_id,
            filename=path.name,
            file_type=document_type,
            source_path=str(path),
            content=normalized_content,
            metadata=metadata,
        )

    def _validate_path(self, path: Path) -> None:
        """Validate that the source path is a usable file."""
        if not path.exists():
            raise FileNotFoundError(
                f"Document was not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Document path is not a file: {path}"
            )

        file_size = path.stat().st_size

        if file_size == 0:
            raise DocumentLoadError(
                f"Document is empty: {path.name}"
            )

        if file_size > self.MAX_FILE_SIZE_BYTES:
            maximum_megabytes = (
                self.MAX_FILE_SIZE_BYTES // 1024 // 1024
            )
            raise DocumentLoadError(
                f"Document '{path.name}' exceeds the "
                f"{maximum_megabytes} MB size limit."
            )

    def _load_by_type(
        self,
        path: Path,
        document_type: DocumentType,
    ) -> tuple[str, list[SourceProperty]]:
        """Select the appropriate loader for the document type."""
        if document_type == DocumentType.TXT:
            return self._load_txt(path)

        if document_type == DocumentType.PDF:
            return self._load_pdf(path)

        if document_type == DocumentType.CSV:
            return self._load_csv(path)

        if document_type == DocumentType.JSON:
            return self._load_json(path)

        raise UnsupportedDocumentTypeError(
            f"No loader exists for {document_type.value}."
        )

    def _load_txt(
        self,
        path: Path,
    ) -> tuple[str, list[SourceProperty]]:
        """Load a plain-text document."""
        content, encoding = self._read_text_file(path)

        return content, [
            SourceProperty(
                key="encoding",
                value=encoding,
            )
        ]

    def _load_pdf(
        self,
        path: Path,
    ) -> tuple[str, list[SourceProperty]]:
        """Extract text from a text-based PDF."""
        try:
            reader = PdfReader(str(path))
        except Exception as error:
            raise DocumentLoadError(
                f"Unable to open PDF '{path.name}': {error}"
            ) from error

        if reader.is_encrypted:
            raise DocumentLoadError(
                f"PDF '{path.name}' is encrypted."
            )

        extracted_pages: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                page_text = page.extract_text() or ""
            except Exception as error:
                raise DocumentLoadError(
                    f"Unable to extract page {page_number} "
                    f"from '{path.name}': {error}"
                ) from error

            if page_text.strip():
                extracted_pages.append(
                    f"--- Page {page_number} ---\n"
                    f"{page_text.strip()}"
                )

        if not extracted_pages:
            raise DocumentLoadError(
                f"PDF '{path.name}' contains no extractable text. "
                "It may be an image-only scanned PDF."
            )

        metadata = [
            SourceProperty(
                key="page_count",
                value=str(len(reader.pages)),
            ),
            SourceProperty(
                key="pages_with_text",
                value=str(len(extracted_pages)),
            ),
        ]

        return "\n\n".join(extracted_pages), metadata

    def _load_csv(
        self,
        path: Path,
    ) -> tuple[str, list[SourceProperty]]:
        """Convert CSV rows into readable text."""
        csv_text, encoding = self._read_text_file(path)
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            raise DocumentLoadError(
                f"CSV '{path.name}' contains no rows."
            )

        raw_headers = rows[0]
        headers = [
            header.strip() or f"column_{index}"
            for index, header in enumerate(
                raw_headers,
                start=1,
            )
        ]

        output_lines = [
            "Columns: " + ", ".join(headers)
        ]

        for row_number, row in enumerate(
            rows[1:],
            start=1,
        ):
            values = []

            for column_index, value in enumerate(row):
                if column_index < len(headers):
                    column_name = headers[column_index]
                else:
                    column_name = (
                        f"column_{column_index + 1}"
                    )

                values.append(
                    f"{column_name}={value.strip()}"
                )

            output_lines.append(
                f"Row {row_number}: " + " | ".join(values)
            )

        metadata = [
            SourceProperty(
                key="encoding",
                value=encoding,
            ),
            SourceProperty(
                key="column_count",
                value=str(len(headers)),
            ),
            SourceProperty(
                key="data_row_count",
                value=str(max(len(rows) - 1, 0)),
            ),
        ]

        return "\n".join(output_lines), metadata

    def _load_json(
        self,
        path: Path,
    ) -> tuple[str, list[SourceProperty]]:
        """Load and format a JSON document."""
        json_text, encoding = self._read_text_file(path)
        parsed_data = json.loads(json_text)

        formatted_content = json.dumps(
            parsed_data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

        metadata = [
            SourceProperty(
                key="encoding",
                value=encoding,
            ),
            SourceProperty(
                key="json_root_type",
                value=type(parsed_data).__name__,
            ),
        ]

        if isinstance(parsed_data, dict):
            metadata.append(
                SourceProperty(
                    key="top_level_key_count",
                    value=str(len(parsed_data)),
                )
            )

        if isinstance(parsed_data, list):
            metadata.append(
                SourceProperty(
                    key="top_level_item_count",
                    value=str(len(parsed_data)),
                )
            )

        return formatted_content, metadata

    @staticmethod
    def _read_text_file(
        path: Path,
    ) -> tuple[str, str]:
        """Read text using common encoding fallbacks."""
        encodings = (
            "utf-8",
            "utf-8-sig",
            "cp1252",
        )

        last_error: UnicodeDecodeError | None = None

        for encoding in encodings:
            try:
                return path.read_text(
                    encoding=encoding
                ), encoding
            except UnicodeDecodeError as error:
                last_error = error

        raise UnicodeError(
            f"Unable to decode '{path.name}' using "
            f"supported encodings."
        ) from last_error

    @staticmethod
    def _normalize_text(content: str) -> str:
        """Normalize line endings and excessive blank lines."""
        normalized = content.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        normalized = "\n".join(
            line.rstrip()
            for line in normalized.splitlines()
        )

        normalized = re.sub(
            r"\n{3,}",
            "\n\n",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _create_document_id(content: str) -> str:
        """Create a stable ID from normalized document content."""
        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        return f"document-{content_hash[:24]}"
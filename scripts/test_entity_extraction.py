"""Run one controlled entity-extraction request."""

import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.ingestion.document_loader import (
    DocumentLoadError,
    DocumentLoader,
    UnsupportedDocumentTypeError,
)
from app.ingestion.entity_extractor import (
    EntityExtractionError,
    EntityExtractor,
)
from app.ingestion.text_splitter import TextSplitter
from app.logging_config import configure_logging


DEFAULT_DOCUMENT = Path(
    "data/sample_documents/project_overview.txt"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a document and test one OpenAI "
            "knowledge-graph extraction."
        )
    )

    parser.add_argument(
        "document",
        nargs="?",
        default=str(DEFAULT_DOCUMENT),
        help="Path to the document being tested.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Send the first chunk to OpenAI. "
            "Without this option, the script performs a dry run."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run a controlled extraction against one document chunk."""
    arguments = parse_arguments()
    configure_logging(settings.log_level)

    try:
        document_path = Path(arguments.document)

        loader = DocumentLoader()
        document = loader.load(document_path)

        splitter = TextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            minimum_chunk_size=80,
        )

        chunks = splitter.split_document(document)

        if not chunks:
            print("No text chunks were created.")
            return 1

        first_chunk = chunks[0]

        print("\nControlled extraction test")
        print("-" * 70)
        print(f"Document       : {document.filename}")
        print(f"Document ID    : {document.document_id}")
        print(f"Document type  : {document.file_type.value}")
        print(f"Total chunks   : {len(chunks)}")
        print(f"Testing chunk  : {first_chunk.chunk_id}")
        print(f"Chunk index    : {first_chunk.chunk_index}")
        print(f"Chunk length   : {len(first_chunk.text)} characters")
        print(f"OpenAI model   : {settings.openai_model}")

        print("\nChunk preview")
        print("-" * 70)
        print(first_chunk.text[:500])

        if not arguments.execute:
            print("\nDry run completed successfully.")
            print(
                "No OpenAI request was made. "
                "Run again with --execute to perform extraction."
            )
            return 0

        print("\nSending one chunk to OpenAI...")

        extractor = EntityExtractor.from_settings()
        result = extractor.extract_chunk(first_chunk)

        print("\nStructured extraction result")
        print("-" * 70)
        print(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nExtraction summary")
        print("-" * 70)
        print(f"Entities      : {len(result.entities)}")
        print(f"Relationships : {len(result.relationships)}")

        if not result.entities:
            print(
                "Warning: no supported entities were extracted."
            )

        print("\nControlled extraction completed successfully.")
        return 0

    except FileNotFoundError as error:
        print(f"Document not found: {error}")
        return 1

    except (
        DocumentLoadError,
        UnsupportedDocumentTypeError,
    ) as error:
        print(f"Document loading failed: {error}")
        return 1

    except EntityExtractionError as error:
        print(f"Entity extraction failed: {error}")
        return 1

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(
            f"Unexpected extraction-test failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""Ingest a document into the Neo4j knowledge graph."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.database.schema import GraphSchemaManager
from app.ingestion.document_loader import (
    DocumentLoadError,
    DocumentLoader,
    UnsupportedDocumentTypeError,
)
from app.ingestion.entity_extractor import (
    EntityExtractionError,
    EntityExtractor,
)
from app.ingestion.graph_builder import (
    GraphBuildError,
    KnowledgeGraphBuilder,
)
from app.ingestion.models import ExtractionResult
from app.ingestion.text_splitter import TextSplitter
from app.logging_config import configure_logging


DEFAULT_DOCUMENT_PATH = Path(
    "data/sample_documents/project_overview.txt"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a document, extract knowledge-graph facts, "
            "and write them to Neo4j."
        )
    )

    parser.add_argument(
        "document",
        nargs="?",
        default=str(DEFAULT_DOCUMENT_PATH),
        help=(
            "Path to a TXT, PDF, CSV, or JSON document. "
            "Defaults to the sample project document."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Perform OpenAI extraction and write the result "
            "to Neo4j. Without this option, only a dry run occurs."
        ),
    )

    parser.add_argument(
        "--show-results",
        action="store_true",
        help=(
            "Display the complete extracted entities and "
            "relationships."
        ),
    )

    return parser.parse_args()


def display_document_information(
    document_path: Path,
    document_id: str,
    file_type: str,
    content_length: int,
    chunk_count: int,
) -> None:
    """Display information about the loaded document."""
    print("\nDocument ingestion")
    print("-" * 72)
    print(f"Document path     : {document_path}")
    print(f"Document ID       : {document_id}")
    print(f"Document type     : {file_type}")
    print(f"Content length    : {content_length} characters")
    print(f"Generated chunks  : {chunk_count}")
    print(f"OpenAI model      : {settings.openai_model}")


def display_extraction_progress(
    chunk_number: int,
    total_chunks: int,
    result: ExtractionResult,
) -> None:
    """Display the result count for one extracted chunk."""
    print(
        f"Chunk {chunk_number}/{total_chunks}: "
        f"{len(result.entities)} entities, "
        f"{len(result.relationships)} relationships"
    )


def display_complete_results(
    results: list[ExtractionResult],
) -> None:
    """Display all structured extraction results."""
    serializable_results = [
        result.model_dump(mode="json")
        for result in results
    ]

    print("\nComplete extraction results")
    print("-" * 72)
    print(
        json.dumps(
            serializable_results,
            indent=2,
            ensure_ascii=False,
        )
    )


def display_graph_summary(
    summary: dict[str, Any],
) -> None:
    """Display the completed Neo4j graph-build summary."""
    print("\nNeo4j graph-build summary")
    print("-" * 72)
    print(f"Document ID          : {summary['document_id']}")
    print(f"Filename             : {summary['filename']}")
    print(f"Chunks written       : {summary['chunks_written']}")
    print(f"Entities written     : {summary['entities_written']}")
    print(f"Mentions written     : {summary['mentions_written']}")
    print(
        "Relationships written: "
        f"{summary['relationships_written']}"
    )


def main() -> int:
    """Run the complete document-ingestion workflow."""
    arguments = parse_arguments()
    configure_logging(settings.log_level)

    document_path = Path(
        arguments.document
    ).expanduser()

    try:
        loader = DocumentLoader()
        document = loader.load(document_path)

        splitter = TextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            minimum_chunk_size=80,
        )

        chunks = splitter.split_document(document)

        if not chunks:
            print("No chunks were generated from the document.")
            return 1

        display_document_information(
            document_path=document_path,
            document_id=document.document_id,
            file_type=document.file_type.value,
            content_length=len(document.content),
            chunk_count=len(chunks),
        )

        print("\nFirst chunk preview")
        print("-" * 72)
        print(chunks[0].text[:600])

        if not arguments.execute:
            print("\nDry run completed successfully.")
            print("No OpenAI request was made.")
            print("No Neo4j data was created or updated.")
            print(
                f"Executing this document will make "
                f"{len(chunks)} extraction request(s)."
            )
            print(
                "\nRun with --execute after reviewing "
                "the document information."
            )
            return 0

        print("\nVerifying Neo4j connection...")

        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                print("Neo4j connection verification failed.")
                return 1

            print("Neo4j connection verified.")

            print("\nEnsuring database constraints exist...")

            schema_manager = GraphSchemaManager(client)
            schema_manager.create_constraints()

            print("Database constraints are ready.")

            extractor = EntityExtractor.from_settings()
            extraction_results: list[ExtractionResult] = []

            print("\nExtracting graph facts...")

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):
                result = extractor.extract_chunk(chunk)
                extraction_results.append(result)

                display_extraction_progress(
                    chunk_number=index,
                    total_chunks=len(chunks),
                    result=result,
                )

            total_entities = sum(
                len(result.entities)
                for result in extraction_results
            )

            total_relationships = sum(
                len(result.relationships)
                for result in extraction_results
            )

            print("\nExtraction completed.")
            print(f"Extracted entities      : {total_entities}")
            print(
                "Extracted relationships : "
                f"{total_relationships}"
            )

            if arguments.show_results:
                display_complete_results(
                    extraction_results
                )

            print("\nWriting extracted knowledge to Neo4j...")

            graph_builder = KnowledgeGraphBuilder(client)

            graph_summary = graph_builder.build_document_graph(
                document=document,
                chunks=chunks,
                extraction_results=extraction_results,
            )

            display_graph_summary(graph_summary)

            print(
                "\nDocument ingestion completed successfully."
            )

            return 0

    except FileNotFoundError as error:
        print(f"Document not found: {error}")
        return 1

    except UnsupportedDocumentTypeError as error:
        print(f"Unsupported document: {error}")
        return 1

    except DocumentLoadError as error:
        print(f"Document loading failed: {error}")
        return 1

    except EntityExtractionError as error:
        print(f"Entity extraction failed: {error}")
        return 1

    except GraphBuildError as error:
        print(f"Neo4j graph build failed: {error}")
        return 1

    except ValueError as error:
        print(f"Configuration or validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Document ingestion failed unexpectedly: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""Test Neo4j chunk and entity full-text retrieval."""

import argparse
import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.retrieval.fulltext_retriever import (
    FullTextRetrievalError,
    FullTextRetriever,
)
from app.retrieval.models import FullTextSearchResult


DEFAULT_QUESTION = "Which project uses Neo4j?"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Search Neo4j chunks and entities using "
            "full-text indexes."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Natural-language search question.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum results from each full-text index.",
    )

    return parser.parse_args()


def display_results(
    result: FullTextSearchResult,
) -> None:
    """Display full-text retrieval results."""
    print("\nFull-text retrieval")
    print("=" * 78)
    print(f"Question     : {result.question}")
    print(f"Lucene query: {result.lucene_query}")

    print("\nRelevant chunks")
    print("-" * 78)

    if not result.chunks:
        print("No matching chunks were found.")

    for position, chunk in enumerate(
        result.chunks,
        start=1,
    ):
        preview = " ".join(
            chunk.text[:350].split()
        )

        print(
            f"\n{position}. Score: {chunk.score:.4f}"
        )
        print(f"   Chunk ID : {chunk.chunk_id}")
        print(
            "   Document : "
            f"{chunk.document_title or chunk.document_id}"
        )
        print(f"   Text     : {preview}")

        if chunk.entities:
            entity_text = ", ".join(
                f"{entity.name} ({entity.entity_type})"
                for entity in chunk.entities
            )

            print(f"   Entities : {entity_text}")

    print("\nMatching entities")
    print("-" * 78)

    if not result.entities:
        print("No matching entities were found.")

    for position, entity in enumerate(
        result.entities,
        start=1,
    ):
        print(
            f"\n{position}. {entity.name} "
            f"({entity.entity_type})"
        )
        print(f"   Score       : {entity.score:.4f}")
        print(f"   Entity ID   : {entity.entity_id}")

        if entity.description:
            print(
                f"   Description : {entity.description}"
            )

        if entity.connections:
            print("   Connections :")

            for connection in entity.connections:
                arrow = (
                    "→"
                    if connection.direction == "outgoing"
                    else "←"
                )

                print(
                    "     "
                    f"{arrow} "
                    f"{connection.relationship_type} "
                    f"{connection.neighbor_name} "
                    f"({connection.neighbor_type})"
                )

        if entity.source_chunks:
            print(
                "   Source chunks: "
                f"{len(entity.source_chunks)}"
            )


def main() -> int:
    """Run one full-text retrieval test."""
    arguments = parse_arguments()
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as client:
            print("Verifying Neo4j connection...")

            if not client.verify_connection():
                print(
                    "Neo4j connection verification failed."
                )
                return 1

            print("Neo4j connection verified.")

            retriever = FullTextRetriever.from_settings(
                client
            )

            result = retriever.search(
                question=arguments.question,
                limit=arguments.limit,
            )

            display_results(result)

            print(
                "\nFull-text retrieval completed successfully."
            )

            return 0

    except FullTextRetrievalError as error:
        print(f"Full-text retrieval failed: {error}")
        return 1

    except ValueError as error:
        print(f"Validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected retrieval failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
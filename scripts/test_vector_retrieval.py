"""Test semantic vector retrieval from Neo4j."""

import argparse
import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
)
from app.retrieval.vector_retriever import (
    SemanticVectorRetriever,
    VectorRetrievalError,
)


DEFAULT_QUESTION = (
    "Which graph database is used by the "
    "Enterprise GraphRAG Assistant?"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve semantically relevant Neo4j chunks "
            "using the vector index."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Natural-language semantic-search question.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to return.",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.0,
        help=(
            "Minimum vector similarity score between "
            "0.0 and 1.0."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run one semantic vector retrieval."""
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

            retriever = SemanticVectorRetriever.from_settings(
                client
            )

            print("\nSemantic vector retrieval")
            print("=" * 78)
            print(f"Question      : {arguments.question}")
            print(
                "Embedding model: "
                f"{settings.openai_embedding_model}"
            )
            print(
                "Vector index   : "
                f"{settings.chunk_vector_index}"
            )
            print(
                "Minimum score  : "
                f"{arguments.minimum_score}"
            )

            results = retriever.search(
                question=arguments.question,
                limit=arguments.limit,
                minimum_score=arguments.minimum_score,
            )

            print("\nRelevant chunks")
            print("-" * 78)

            if not results:
                print(
                    "No vector-search results met the "
                    "minimum score."
                )

            for position, result in enumerate(
                results,
                start=1,
            ):
                preview = " ".join(
                    result.text[:500].split()
                )

                print(
                    f"\n{position}. Similarity score: "
                    f"{result.score:.4f}"
                )
                print(
                    f"   Chunk ID : {result.chunk_id}"
                )
                print(
                    "   Document : "
                    f"{result.document_title "
                    f"or result.document_id "
                    f"or 'Unknown'}"
                )
                print(
                    f"   Index    : "
                    f"{result.chunk_index}"
                )
                print(
                    f"   Text     : {preview}"
                )

                if result.entities:
                    entity_text = ", ".join(
                        (
                            f"{entity.name} "
                            f"({entity.entity_type})"
                        )
                        for entity in result.entities
                    )

                    print(
                        f"   Entities : {entity_text}"
                    )

            print(
                "\nSemantic vector retrieval "
                "completed successfully."
            )

            return 0

    except EmbeddingGenerationError as error:
        print(
            f"Question embedding failed: {error}"
        )
        return 1

    except VectorRetrievalError as error:
        print(f"Vector retrieval failed: {error}")
        return 1

    except ValueError as error:
        print(f"Validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected vector-retrieval failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
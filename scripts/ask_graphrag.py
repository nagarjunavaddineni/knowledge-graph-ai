"""Ask a grounded question against the Neo4j knowledge graph."""

import argparse
import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.services.answer_models import GroundedAnswer
from app.services.graphrag_service import (
    GraphRAGAnswerError,
    GraphRAGAnswerService,
)


DEFAULT_QUESTION = (
    "Which technologies are used by the "
    "Enterprise GraphRAG Assistant?"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask a grounded question using hybrid Neo4j "
            "GraphRAG retrieval."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Natural-language question.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of retrieved chunks.",
    )

    parser.add_argument(
        "--minimum-vector-score",
        type=float,
        default=0.0,
        help=(
            "Minimum vector similarity score between "
            "0.0 and 1.0."
        ),
    )

    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help=(
            "Maximum graph relationships retrieved "
            "for each entity."
        ),
    )

    parser.add_argument(
        "--show-graph-facts",
        action="store_true",
        help="Display graph facts used as answer context.",
    )

    return parser.parse_args()


def display_answer(
    result: GroundedAnswer,
    show_graph_facts: bool,
) -> None:
    """Display the grounded answer and its sources."""
    print("\nGrounded GraphRAG answer")
    print("=" * 80)
    print(f"Question   : {result.question}")
    print(
        "Answerable : "
        f"{'Yes' if result.answerable else 'No'}"
    )
    print(f"Confidence : {result.confidence:.2f}")

    print("\nAnswer")
    print("-" * 80)
    print(result.answer)

    if result.insufficient_evidence_reason:
        print("\nReason")
        print("-" * 80)
        print(result.insufficient_evidence_reason)

    print("\nSources")
    print("-" * 80)

    if not result.sources:
        print("No supporting sources were available.")

    for source in result.sources:
        print(
            f"\n[{source.source_id}] "
            f"{source.document_title "
            or source.document_id "
            or 'Unknown document'}"
        )
        print(f"Chunk ID : {source.chunk_id}")
        print(f"Index    : {source.chunk_index}")
        print(
            f"RRF score: "
            f"{source.retrieval_score:.6f}"
        )
        print(f"Excerpt  : {source.excerpt}")

    if show_graph_facts:
        print("\nGraph facts")
        print("-" * 80)

        if not result.graph_facts:
            print("No graph facts supported this answer.")

        for fact in result.graph_facts:
            print(f"- {fact}")


def main() -> int:
    """Run one grounded GraphRAG question."""
    arguments = parse_arguments()
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as neo4j_client:
            print("Verifying Neo4j connection...")

            if not neo4j_client.verify_connection():
                print(
                    "Neo4j connection verification failed."
                )
                return 1

            print("Neo4j connection verified.")
            print(
                f"Answer model: {settings.openai_model}"
            )

            service = GraphRAGAnswerService.from_settings(
                neo4j_client
            )

            print("\nRetrieving and generating answer...")

            result = service.answer_question(
                question=arguments.question,
                limit=arguments.limit,
                minimum_vector_score=(
                    arguments.minimum_vector_score
                ),
                max_connections_per_entity=(
                    arguments.max_connections
                ),
            )

            display_answer(
                result=result,
                show_graph_facts=(
                    arguments.show_graph_facts
                ),
            )

            print(
                "\nGraphRAG question completed successfully."
            )

            return 0

    except GraphRAGAnswerError as error:
        print(f"GraphRAG answer failed: {error}")
        return 1

    except ValueError as error:
        print(f"Validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected GraphRAG failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
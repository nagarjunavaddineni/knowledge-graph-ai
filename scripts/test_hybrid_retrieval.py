"""Test hybrid full-text and vector retrieval."""

import argparse
import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.retrieval.hybrid_retriever import (
    HybridRetrievalError,
    RankFusionHybridRetriever,
)
from app.retrieval.models import HybridSearchResult


DEFAULT_QUESTION = (
    "Which technologies are used by the "
    "Enterprise GraphRAG Assistant?"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine Neo4j full-text and vector "
            "retrieval using reciprocal rank fusion."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Natural-language retrieval question.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of fused results.",
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

    return parser.parse_args()


def display_results(
    result: HybridSearchResult,
) -> None:
    """Display fused retrieval results."""
    print("\nHybrid GraphRAG retrieval")
    print("=" * 80)
    print(f"Question                 : {result.question}")
    print(f"RRF constant             : {result.rrf_constant}")
    print(
        "Candidate limit          : "
        f"{result.candidate_limit}"
    )
    print(
        "Full-text candidates     : "
        f"{result.fulltext_candidate_count}"
    )
    print(
        "Vector candidates        : "
        f"{result.vector_candidate_count}"
    )

    print("\nFused results")
    print("-" * 80)

    if not result.results:
        print("No hybrid results were found.")
        return

    for position, item in enumerate(
        result.results,
        start=1,
    ):
        preview = " ".join(
            item.text[:500].split()
        )

        print(f"\n{position}. RRF score: {item.rrf_score:.6f}")
        print(f"   Chunk ID : {item.chunk_id}")
        print(
            "   Document : "
            f"{item.document_title "
            f"or item.document_id "
            f"or 'Unknown'}"
        )
        print(
            "   Matched by: "
            f"{', '.join(item.matched_by)}"
        )

        if item.fulltext_rank is not None:
            print(
                "   Full-text: "
                f"rank={item.fulltext_rank}, "
                f"score={item.fulltext_score:.4f}"
            )

        if item.vector_rank is not None:
            print(
                "   Vector   : "
                f"rank={item.vector_rank}, "
                f"score={item.vector_score:.4f}"
            )

        print(f"   Text     : {preview}")

        if item.entities:
            entity_text = ", ".join(
                (
                    f"{entity.name} "
                    f"({entity.entity_type})"
                )
                for entity in item.entities
            )

            print(f"   Entities : {entity_text}")


def main() -> int:
    """Run one hybrid retrieval test."""
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

            retriever = (
                RankFusionHybridRetriever.from_settings(
                    client
                )
            )

            result = retriever.search(
                question=arguments.question,
                limit=arguments.limit,
                minimum_vector_score=(
                    arguments.minimum_vector_score
                ),
            )

            display_results(result)

            print(
                "\nHybrid retrieval completed successfully."
            )

            return 0

    except HybridRetrievalError as error:
        print(f"Hybrid retrieval failed: {error}")
        return 1

    except ValueError as error:
        print(f"Validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected hybrid-retrieval failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
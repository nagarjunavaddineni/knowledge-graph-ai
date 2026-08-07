"""Test hybrid retrieval with Neo4j graph-context expansion."""

import argparse
import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.retrieval.context_models import GraphContextResult
from app.retrieval.graph_context_expander import (
    GraphContextExpander,
    GraphContextExpansionError,
)
from app.retrieval.hybrid_retriever import (
    HybridRetrievalError,
    RankFusionHybridRetriever,
)


DEFAULT_QUESTION = (
    "Which technologies are used by the "
    "Enterprise GraphRAG Assistant?"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid retrieval and expand results "
            "through Neo4j graph relationships."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Natural-language GraphRAG question.",
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
            "Minimum semantic similarity score between "
            "0.0 and 1.0."
        ),
    )

    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help=(
            "Maximum graph connections displayed "
            "for each entity."
        ),
    )

    return parser.parse_args()


def display_context(
    context: GraphContextResult,
) -> None:
    """Display expanded retrieval context."""
    print("\nExpanded GraphRAG context")
    print("=" * 80)
    print(f"Question              : {context.question}")
    print(
        "Retrieved chunks      : "
        f"{context.retrieved_chunk_count}"
    )
    print(
        "Expanded chunks       : "
        f"{context.expanded_chunk_count}"
    )

    if not context.chunks:
        print("\nNo graph context was found.")
        return

    for position, chunk in enumerate(
        context.chunks,
        start=1,
    ):
        preview = " ".join(
            chunk.text[:500].split()
        )

        print("\n" + "-" * 80)
        print(
            f"{position}. "
            f"{chunk.document_title "
            f"or chunk.document_id "
            f"or 'Unknown document'}"
        )
        print(f"Chunk ID   : {chunk.chunk_id}")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"RRF score  : {chunk.rrf_score:.6f}")
        print(
            "Matched by : "
            f"{', '.join(chunk.matched_by)}"
        )
        print(f"Text       : {preview}")

        if not chunk.entities:
            print("Entities   : No expanded entities")
            continue

        print("Entities and relationships:")

        for entity in chunk.entities:
            print(
                f"\n  • {entity.name} "
                f"({entity.entity_type})"
            )

            if entity.description:
                print(
                    f"    Description: "
                    f"{entity.description}"
                )

            if not entity.connections:
                print(
                    "    Connections: "
                    "No business relationships found"
                )
                continue

            for connection in entity.connections:
                if connection.direction == "outgoing":
                    print(
                        "    "
                        f"{entity.name} "
                        f"—[{connection.relationship_type}]→ "
                        f"{connection.neighbor_name} "
                        f"({connection.neighbor_type})"
                    )
                else:
                    print(
                        "    "
                        f"{connection.neighbor_name} "
                        f"({connection.neighbor_type}) "
                        f"—[{connection.relationship_type}]→ "
                        f"{entity.name}"
                    )


def main() -> int:
    """Run hybrid retrieval and graph expansion."""
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

            hybrid_retriever = (
                RankFusionHybridRetriever.from_settings(
                    client
                )
            )

            print("\nRunning hybrid retrieval...")

            retrieval_result = hybrid_retriever.search(
                question=arguments.question,
                limit=arguments.limit,
                minimum_vector_score=(
                    arguments.minimum_vector_score
                ),
            )

            print(
                f"Hybrid retrieval returned "
                f"{len(retrieval_result.results)} chunks."
            )

            expander = GraphContextExpander(client)

            print("Expanding results through the graph...")

            context = expander.expand(
                retrieval_result=retrieval_result,
                max_connections_per_entity=(
                    arguments.max_connections
                ),
            )

            display_context(context)

            print(
                "\nGraph-context expansion "
                "completed successfully."
            )

            return 0

    except HybridRetrievalError as error:
        print(f"Hybrid retrieval failed: {error}")
        return 1

    except GraphContextExpansionError as error:
        print(f"Graph expansion failed: {error}")
        return 1

    except ValueError as error:
        print(f"Validation error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected graph-context failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
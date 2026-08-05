"""Create and verify Neo4j GraphRAG retrieval indexes."""

import sys
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.database.retrieval_indexes import (
    RetrievalIndexError,
    RetrievalIndexManager,
)
from app.logging_config import configure_logging


def display_indexes(
    indexes: list[dict[str, Any]],
) -> None:
    """Display retrieval-index information."""
    print("\nGraphRAG retrieval indexes")
    print("-" * 80)

    for index in indexes:
        print(
            f"Name        : "
            f"{index.get('name', 'Unknown')}"
        )
        print(
            f"Type        : "
            f"{index.get('type', 'Unknown')}"
        )
        print(
            f"State       : "
            f"{index.get('state', 'Unknown')}"
        )
        print(
            f"Population  : "
            f"{index.get('populationPercent', 0)}%"
        )
        print(
            f"Labels      : "
            f"{index.get('labelsOrTypes', [])}"
        )
        print(
            f"Properties  : "
            f"{index.get('properties', [])}"
        )
        print("-" * 80)


def main() -> int:
    """Create and verify all retrieval indexes."""
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

            manager = RetrievalIndexManager.from_settings(
                client
            )

            print("\nCreating retrieval indexes...")
            manager.create_indexes()

            print(
                "Waiting for retrieval indexes "
                "to become ONLINE..."
            )

            indexes = manager.wait_until_online(
                timeout_seconds=180,
                poll_interval_seconds=1,
            )

            manager.verify_index_definitions(indexes)

            display_indexes(indexes)

            print(
                "\nRetrieval indexes created and "
                "verified successfully."
            )
            print(
                f"Verified {len(manager.index_names)} "
                "GraphRAG indexes."
            )

            return 0

    except RetrievalIndexError as error:
        print(f"Retrieval-index setup failed: {error}")
        return 1

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected retrieval-index setup failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""Verify the Neo4j configuration and database connection."""

import sys

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging


def main() -> int:
    """Connect to Neo4j and run a simple test query."""
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                print("Neo4j connection verification failed.")
                return 1

            records = client.execute_query(
                "RETURN 1 AS connection_test"
            )

            if not records:
                print("Neo4j returned no records.")
                return 1

            value = records[0].get("connection_test")

            if value != 1:
                print(f"Unexpected Neo4j response: {records}")
                return 1

            print("Neo4j connection test passed.")
            print(f"Database response: {value}")
            return 0

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(f"Neo4j connection test failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
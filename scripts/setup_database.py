"""Create and verify the Neo4j schema."""

import sys
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.database.schema import GraphSchemaManager
from app.logging_config import configure_logging


def display_constraints(
    constraints: list[dict[str, Any]],
) -> None:
    """Print Neo4j constraints in a readable format."""
    if not constraints:
        print("No constraints were found.")
        return

    print("\nNeo4j constraints")
    print("-" * 80)

    for constraint in constraints:
        name = constraint.get("name", "Unknown")
        constraint_type = constraint.get("type", "Unknown")
        labels = constraint.get("labelsOrTypes", [])
        properties = constraint.get("properties", [])

        print(f"Name       : {name}")
        print(f"Type       : {constraint_type}")
        print(f"Label      : {labels}")
        print(f"Properties : {properties}")
        print("-" * 80)


def main() -> int:
    """Create the Neo4j constraints and display the result."""
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                print("Neo4j connection verification failed.")
                return 1

            schema_manager = GraphSchemaManager(client)

            print("Creating Neo4j constraints...")
            schema_manager.create_constraints()

            constraints = schema_manager.list_constraints()
            display_constraints(constraints)

            expected_constraints = {
                "person_id_unique",
                "project_id_unique",
                "customer_id_unique",
                "team_id_unique",
                "technology_id_unique",
                "document_id_unique",
            }

            created_names = {
                str(constraint.get("name"))
                for constraint in constraints
            }

            missing_constraints = expected_constraints - created_names

            if missing_constraints:
                print(
                    "\nDatabase setup incomplete. Missing constraints:"
                )

                for constraint_name in sorted(missing_constraints):
                    print(f"- {constraint_name}")

                return 1

            print("\nDatabase schema setup completed successfully.")
            print(
                f"Verified {len(expected_constraints)} "
                "application constraints."
            )
            return 0

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(f"Database setup failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
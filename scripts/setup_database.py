"""Create and verify the Neo4j database schema."""

import sys
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.database.schema import GraphSchemaManager
from app.logging_config import configure_logging


EXPECTED_CONSTRAINTS = {
    "person_id_unique",
    "project_id_unique",
    "customer_id_unique",
    "team_id_unique",
    "technology_id_unique",
    "document_id_unique",
    "chunk_id_unique",
}


def display_constraints(
    constraints: list[dict[str, Any]],
) -> None:
    """Display Neo4j constraints in a readable format."""
    if not constraints:
        print("\nNo constraints were found.")
        return

    print("\nNeo4j constraints")
    print("-" * 80)

    for constraint in constraints:
        name = constraint.get("name", "Unknown")
        constraint_type = constraint.get("type", "Unknown")
        entity_type = constraint.get("entityType", "Unknown")
        labels = constraint.get("labelsOrTypes", [])
        properties = constraint.get("properties", [])

        print(f"Name        : {name}")
        print(f"Type        : {constraint_type}")
        print(f"Entity type : {entity_type}")
        print(f"Labels      : {labels}")
        print(f"Properties  : {properties}")
        print("-" * 80)


def verify_constraints(
    constraints: list[dict[str, Any]],
) -> set[str]:
    """Return the names of expected constraints that are missing."""
    created_constraint_names = {
        str(constraint.get("name"))
        for constraint in constraints
        if constraint.get("name")
    }

    return EXPECTED_CONSTRAINTS - created_constraint_names


def main() -> int:
    """Create Neo4j constraints and verify the schema."""
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as client:
            print("Verifying Neo4j connection...")

            if not client.verify_connection():
                print("Neo4j connection verification failed.")
                return 1

            print("Neo4j connection verified successfully.")

            schema_manager = GraphSchemaManager(client)

            print("\nCreating Neo4j constraints...")
            schema_manager.create_constraints()

            constraints = schema_manager.list_constraints()

            display_constraints(constraints)

            missing_constraints = verify_constraints(
                constraints
            )

            if missing_constraints:
                print(
                    "\nDatabase schema setup is incomplete."
                )
                print("Missing constraints:")

                for constraint_name in sorted(
                    missing_constraints
                ):
                    print(f"- {constraint_name}")

                return 1

            print(
                "\nDatabase schema setup completed successfully."
            )
            print(
                f"Verified {len(EXPECTED_CONSTRAINTS)} "
                "application constraints."
            )

            return 0

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(
            "Database setup failed: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""Create sample nodes and relationships in Neo4j."""

import json
import sys
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.database.repositories import KnowledgeGraphRepository
from app.logging_config import configure_logging


def display_result(title: str, result: Any) -> None:
    """Display a repository result in readable JSON format."""
    print(f"\n{title}")
    print("-" * 70)
    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


def main() -> int:
    """Seed the Neo4j database and verify repository operations."""
    configure_logging(settings.log_level)

    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                print("Neo4j connection verification failed.")
                return 1

            repository = KnowledgeGraphRepository(client)

            print("Creating sample knowledge graph...")

            project_manager = repository.upsert_person(
                person_id="person-alex-morgan",
                name="Alex Morgan",
                role="AI Project Manager",
            )
            display_result(
                "Created or updated project manager",
                project_manager,
            )

            ai_engineer = repository.upsert_person(
                person_id="person-priya-sharma",
                name="Priya Sharma",
                role="AI Engineer",
            )
            display_result(
                "Created or updated AI engineer",
                ai_engineer,
            )

            project = repository.upsert_project(
                project_id="project-enterprise-graphrag",
                name="Enterprise GraphRAG Assistant",
                description=(
                    "An enterprise knowledge assistant combining "
                    "Neo4j graph traversal and generative AI."
                ),
                status="active",
            )
            display_result(
                "Created or updated project",
                project,
            )

            customer = repository.upsert_customer(
                customer_id="customer-acme",
                name="Acme Corporation",
                industry="Technology",
            )
            display_result(
                "Created or updated customer",
                customer,
            )

            team = repository.upsert_team(
                team_id="team-ai-platform",
                name="AI Platform Team",
                department="Engineering",
            )
            display_result(
                "Created or updated team",
                team,
            )

            neo4j_technology = repository.upsert_technology(
                technology_id="technology-neo4j",
                name="Neo4j",
                category="Graph Database",
            )
            display_result(
                "Created or updated Neo4j technology",
                neo4j_technology,
            )

            python_technology = repository.upsert_technology(
                technology_id="technology-python",
                name="Python",
                category="Programming Language",
            )
            display_result(
                "Created or updated Python technology",
                python_technology,
            )

            document = repository.upsert_document(
                document_id="document-graphrag-overview",
                title="Enterprise GraphRAG Project Overview",
                source="sample-data",
                content=(
                    "The Enterprise GraphRAG Assistant uses Neo4j "
                    "and Python to retrieve connected enterprise data."
                ),
            )
            display_result(
                "Created or updated document",
                document,
            )

            manager_relationship = (
                repository.assign_project_manager(
                    person_id="person-alex-morgan",
                    project_id="project-enterprise-graphrag",
                )
            )
            display_result(
                "Created MANAGES relationship",
                manager_relationship,
            )

            work_relationship = repository.assign_person_to_project(
                person_id="person-priya-sharma",
                project_id="project-enterprise-graphrag",
            )
            display_result(
                "Created WORKS_ON relationship",
                work_relationship,
            )

            customer_relationship = (
                repository.connect_project_to_customer(
                    project_id="project-enterprise-graphrag",
                    customer_id="customer-acme",
                )
            )
            display_result(
                "Created SERVES relationship",
                customer_relationship,
            )

            neo4j_relationship = (
                repository.connect_project_to_technology(
                    project_id="project-enterprise-graphrag",
                    technology_id="technology-neo4j",
                )
            )
            display_result(
                "Created Neo4j USES relationship",
                neo4j_relationship,
            )

            python_relationship = (
                repository.connect_project_to_technology(
                    project_id="project-enterprise-graphrag",
                    technology_id="technology-python",
                )
            )
            display_result(
                "Created Python USES relationship",
                python_relationship,
            )

            project_context = repository.get_project_context(
                project_id="project-enterprise-graphrag"
            )
            display_result(
                "Retrieved project context",
                project_context,
            )

            statistics = repository.get_graph_statistics()
            display_result(
                "Knowledge graph statistics",
                statistics,
            )

            if not project_context:
                print("\nProject context verification failed.")
                return 1

            if not statistics:
                print("\nGraph statistics verification failed.")
                return 1

            print("\nSample knowledge graph created successfully.")
            print("Repository operations verified successfully.")
            return 0

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(f"Sample graph creation failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
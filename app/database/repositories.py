"""Repository operations for the Knowledge Graph AI database."""

from typing import Any

from app.database.neo4j_client import Neo4jClient


class KnowledgeGraphRepository:
    """Create and retrieve knowledge-graph nodes and relationships."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    @staticmethod
    def _first_record(
        records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the first record when one exists."""
        return records[0] if records else None

    def upsert_person(
        self,
        person_id: str,
        name: str,
        email: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a Person node."""
        query = """
        MERGE (person:Person {id: $person_id})
        ON CREATE SET person.created_at = datetime()
        SET
            person.name = $name,
            person.email = $email,
            person.role = $role,
            person.updated_at = datetime()
        RETURN person {
            .id,
            .name,
            .email,
            .role
        } AS person
        """

        records = self.client.execute_query(
            query,
            {
                "person_id": person_id,
                "name": name,
                "email": email,
                "role": role,
            },
        )

        return self._first_record(records)

    def upsert_project(
        self,
        project_id: str,
        name: str,
        description: str | None = None,
        status: str = "active",
    ) -> dict[str, Any] | None:
        """Create or update a Project node."""
        query = """
        MERGE (project:Project {id: $project_id})
        ON CREATE SET project.created_at = datetime()
        SET
            project.name = $name,
            project.description = $description,
            project.status = $status,
            project.updated_at = datetime()
        RETURN project {
            .id,
            .name,
            .description,
            .status
        } AS project
        """

        records = self.client.execute_query(
            query,
            {
                "project_id": project_id,
                "name": name,
                "description": description,
                "status": status,
            },
        )

        return self._first_record(records)

    def upsert_customer(
        self,
        customer_id: str,
        name: str,
        industry: str | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a Customer node."""
        query = """
        MERGE (customer:Customer {id: $customer_id})
        ON CREATE SET customer.created_at = datetime()
        SET
            customer.name = $name,
            customer.industry = $industry,
            customer.updated_at = datetime()
        RETURN customer {
            .id,
            .name,
            .industry
        } AS customer
        """

        records = self.client.execute_query(
            query,
            {
                "customer_id": customer_id,
                "name": name,
                "industry": industry,
            },
        )

        return self._first_record(records)

    def upsert_team(
        self,
        team_id: str,
        name: str,
        department: str | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a Team node."""
        query = """
        MERGE (team:Team {id: $team_id})
        ON CREATE SET team.created_at = datetime()
        SET
            team.name = $name,
            team.department = $department,
            team.updated_at = datetime()
        RETURN team {
            .id,
            .name,
            .department
        } AS team
        """

        records = self.client.execute_query(
            query,
            {
                "team_id": team_id,
                "name": name,
                "department": department,
            },
        )

        return self._first_record(records)

    def upsert_technology(
        self,
        technology_id: str,
        name: str,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a Technology node."""
        query = """
        MERGE (technology:Technology {id: $technology_id})
        ON CREATE SET technology.created_at = datetime()
        SET
            technology.name = $name,
            technology.category = $category,
            technology.updated_at = datetime()
        RETURN technology {
            .id,
            .name,
            .category
        } AS technology
        """

        records = self.client.execute_query(
            query,
            {
                "technology_id": technology_id,
                "name": name,
                "category": category,
            },
        )

        return self._first_record(records)

    def upsert_document(
        self,
        document_id: str,
        title: str,
        source: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a Document node."""
        query = """
        MERGE (document:Document {id: $document_id})
        ON CREATE SET document.created_at = datetime()
        SET
            document.title = $title,
            document.source = $source,
            document.content = $content,
            document.updated_at = datetime()
        RETURN document {
            .id,
            .title,
            .source
        } AS document
        """

        records = self.client.execute_query(
            query,
            {
                "document_id": document_id,
                "title": title,
                "source": source,
                "content": content,
            },
        )

        return self._first_record(records)

    def assign_person_to_project(
        self,
        person_id: str,
        project_id: str,
    ) -> dict[str, Any] | None:
        """Create a WORKS_ON relationship."""
        query = """
        MATCH
            (person:Person {id: $person_id}),
            (project:Project {id: $project_id})
        MERGE (person)-[relationship:WORKS_ON]->(project)
        ON CREATE SET relationship.created_at = datetime()
        RETURN
            person.name AS person,
            type(relationship) AS relationship,
            project.name AS project
        """

        records = self.client.execute_query(
            query,
            {
                "person_id": person_id,
                "project_id": project_id,
            },
        )

        return self._first_record(records)

    def assign_project_manager(
        self,
        person_id: str,
        project_id: str,
    ) -> dict[str, Any] | None:
        """Create a MANAGES relationship."""
        query = """
        MATCH
            (person:Person {id: $person_id}),
            (project:Project {id: $project_id})
        MERGE (person)-[relationship:MANAGES]->(project)
        ON CREATE SET relationship.created_at = datetime()
        RETURN
            person.name AS person,
            type(relationship) AS relationship,
            project.name AS project
        """

        records = self.client.execute_query(
            query,
            {
                "person_id": person_id,
                "project_id": project_id,
            },
        )

        return self._first_record(records)

    def connect_project_to_customer(
        self,
        project_id: str,
        customer_id: str,
    ) -> dict[str, Any] | None:
        """Create a SERVES relationship."""
        query = """
        MATCH
            (project:Project {id: $project_id}),
            (customer:Customer {id: $customer_id})
        MERGE (project)-[relationship:SERVES]->(customer)
        ON CREATE SET relationship.created_at = datetime()
        RETURN
            project.name AS project,
            type(relationship) AS relationship,
            customer.name AS customer
        """

        records = self.client.execute_query(
            query,
            {
                "project_id": project_id,
                "customer_id": customer_id,
            },
        )

        return self._first_record(records)

    def connect_project_to_technology(
        self,
        project_id: str,
        technology_id: str,
    ) -> dict[str, Any] | None:
        """Create a USES relationship."""
        query = """
        MATCH
            (project:Project {id: $project_id}),
            (technology:Technology {id: $technology_id})
        MERGE (project)-[relationship:USES]->(technology)
        ON CREATE SET relationship.created_at = datetime()
        RETURN
            project.name AS project,
            type(relationship) AS relationship,
            technology.name AS technology
        """

        records = self.client.execute_query(
            query,
            {
                "project_id": project_id,
                "technology_id": technology_id,
            },
        )

        return self._first_record(records)

    def get_project_context(
        self,
        project_id: str,
    ) -> list[dict[str, Any]]:
        """Return people, customers, and technologies for a project."""
        query = """
        MATCH (project:Project {id: $project_id})

        OPTIONAL MATCH
            (person:Person)-[person_relationship]->(project)
        WHERE type(person_relationship) IN ["WORKS_ON", "MANAGES"]

        OPTIONAL MATCH
            (project)-[:SERVES]->(customer:Customer)

        OPTIONAL MATCH
            (project)-[:USES]->(technology:Technology)

        RETURN
            project {
                .id,
                .name,
                .description,
                .status
            } AS project,
            collect(
                DISTINCT {
                    id: person.id,
                    name: person.name,
                    relationship: type(person_relationship)
                }
            ) AS people,
            collect(
                DISTINCT {
                    id: customer.id,
                    name: customer.name,
                    industry: customer.industry
                }
            ) AS customers,
            collect(
                DISTINCT {
                    id: technology.id,
                    name: technology.name,
                    category: technology.category
                }
            ) AS technologies
        """

        return self.client.execute_query(
            query,
            {"project_id": project_id},
        )

    def get_graph_statistics(self) -> dict[str, Any] | None:
        """Return counts for the main knowledge-graph node types."""
        query = """
        MATCH (node)
        RETURN
            count(CASE WHEN node:Person THEN 1 END) AS people,
            count(CASE WHEN node:Project THEN 1 END) AS projects,
            count(CASE WHEN node:Customer THEN 1 END) AS customers,
            count(CASE WHEN node:Team THEN 1 END) AS teams,
            count(CASE WHEN node:Technology THEN 1 END) AS technologies,
            count(CASE WHEN node:Document THEN 1 END) AS documents
        """

        records = self.client.execute_query(query)
        return self._first_record(records)
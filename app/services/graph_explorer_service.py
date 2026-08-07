"""Read graph-explorer data from Neo4j."""

import logging
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from app.database.neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


class GraphExplorerError(RuntimeError):
    """Raised when graph-explorer data cannot be loaded."""


class GraphExplorerModel(BaseModel):
    """Base model for graph-explorer data."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class GraphNode(GraphExplorerModel):
    """A Neo4j node displayed in the graph explorer."""

    node_id: str = Field(min_length=1)
    business_id: str | None = None
    display_name: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(GraphExplorerModel):
    """A Neo4j relationship displayed in the graph explorer."""

    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphExplorerResult(GraphExplorerModel):
    """Complete graph data returned to the UI."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    selected_labels: list[str] = Field(default_factory=list)
    selected_relationship_types: list[str] = Field(
        default_factory=list
    )
    node_limit: int = Field(gt=0)
    relationship_limit: int = Field(gt=0)


NODE_TYPE_PRIORITY: Final[tuple[str, ...]] = (
    "Person",
    "Project",
    "Customer",
    "Team",
    "Technology",
    "Document",
    "Chunk",
)


class GraphExplorerService:
    """Load filtered Neo4j nodes and relationships."""

    def __init__(
        self,
        client: Neo4jClient,
    ) -> None:
        """Initialize the graph-explorer service."""
        self.client = client

    def get_available_node_labels(self) -> list[str]:
        """Return all node labels currently stored in Neo4j."""
        query = """
        MATCH (node)
        UNWIND labels(node) AS node_label
        RETURN DISTINCT node_label
        ORDER BY node_label
        """

        try:
            records = self.client.execute_query(query)

        except Exception as error:
            logger.exception(
                "Unable to read Neo4j node labels."
            )

            raise GraphExplorerError(
                "Unable to load available node labels."
            ) from error

        labels = {
            str(record.get("node_label")).strip()
            for record in records
            if record.get("node_label")
        }

        return sorted(labels)

    def get_available_relationship_types(
        self,
    ) -> list[str]:
        """Return relationship types currently stored in Neo4j."""
        query = """
        MATCH ()-[relationship]->()
        RETURN DISTINCT type(relationship)
            AS relationship_type
        ORDER BY relationship_type
        """

        try:
            records = self.client.execute_query(query)

        except Exception as error:
            logger.exception(
                "Unable to read Neo4j relationship types."
            )

            raise GraphExplorerError(
                "Unable to load relationship types."
            ) from error

        relationship_types = {
            str(
                record.get("relationship_type")
            ).strip()
            for record in records
            if record.get("relationship_type")
        }

        return sorted(relationship_types)

    def load_graph(
        self,
        node_labels: list[str] | None = None,
        relationship_types: list[str] | None = None,
        node_limit: int = 75,
        relationship_limit: int = 150,
    ) -> GraphExplorerResult:
        """Load a filtered graph from Neo4j."""
        if node_limit <= 0:
            raise ValueError(
                "node_limit must be greater than zero."
            )

        if node_limit > 500:
            raise ValueError(
                "node_limit cannot exceed 500."
            )

        if relationship_limit <= 0:
            raise ValueError(
                "relationship_limit must be greater than zero."
            )

        if relationship_limit > 1000:
            raise ValueError(
                "relationship_limit cannot exceed 1000."
            )

        normalized_labels = self._normalize_filters(
            node_labels
        )

        normalized_relationship_types = (
            self._normalize_filters(
                relationship_types,
                uppercase=True,
            )
        )

        query = """
        MATCH
            (source)-[relationship]->(target)

        WHERE
            (
                size($node_labels) = 0
                OR (
                    any(
                        node_label IN labels(source)
                        WHERE node_label IN $node_labels
                    )
                    AND any(
                        node_label IN labels(target)
                        WHERE node_label IN $node_labels
                    )
                )
            )
            AND (
                size($relationship_types) = 0
                OR type(relationship)
                    IN $relationship_types
            )

        WITH
            source,
            relationship,
            target

        ORDER BY
            type(relationship),
            coalesce(
                source.name,
                source.title,
                source.filename,
                source.id,
                elementId(source)
            ),
            coalesce(
                target.name,
                target.title,
                target.filename,
                target.id,
                elementId(target)
            )

        LIMIT $relationship_limit

        RETURN
            elementId(source) AS source_node_id,
            labels(source) AS source_labels,
            {
                id: source.id,
                name: source.name,
                title: source.title,
                filename: source.filename,
                description: source.description,
                role: source.role,
                industry: source.industry,
                department: source.department,
                category: source.category,
                document_id: source.document_id,
                chunk_index: source.chunk_index,
                text:
                    CASE
                        WHEN source.text IS NULL
                        THEN NULL
                        ELSE substring(
                            source.text,
                            0,
                            500
                        )
                    END
            } AS source_properties,

            elementId(relationship)
                AS relationship_id,
            type(relationship)
                AS relationship_type,
            properties(relationship)
                AS relationship_properties,

            elementId(target) AS target_node_id,
            labels(target) AS target_labels,
            {
                id: target.id,
                name: target.name,
                title: target.title,
                filename: target.filename,
                description: target.description,
                role: target.role,
                industry: target.industry,
                department: target.department,
                category: target.category,
                document_id: target.document_id,
                chunk_index: target.chunk_index,
                text:
                    CASE
                        WHEN target.text IS NULL
                        THEN NULL
                        ELSE substring(
                            target.text,
                            0,
                            500
                        )
                    END
            } AS target_properties
        """

        try:
            records = self.client.execute_query(
                query,
                {
                    "node_labels": normalized_labels,
                    "relationship_types": (
                        normalized_relationship_types
                    ),
                    "relationship_limit": (
                        relationship_limit
                    ),
                },
            )

        except Exception as error:
            logger.exception(
                "Unable to load Neo4j graph-explorer data."
            )

            raise GraphExplorerError(
                "Unable to load the selected graph."
            ) from error

        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        seen_edge_ids: set[str] = set()

        for record in records:
            source_node = self._create_node(
                node_id=record.get(
                    "source_node_id"
                ),
                labels=record.get(
                    "source_labels"
                ),
                properties=record.get(
                    "source_properties"
                ),
            )

            target_node = self._create_node(
                node_id=record.get(
                    "target_node_id"
                ),
                labels=record.get(
                    "target_labels"
                ),
                properties=record.get(
                    "target_properties"
                ),
            )

            if source_node is None or target_node is None:
                continue

            new_node_ids = {
                node.node_id
                for node in (
                    source_node,
                    target_node,
                )
                if node.node_id not in nodes
            }

            if (
                len(nodes) + len(new_node_ids)
                > node_limit
            ):
                continue

            nodes[source_node.node_id] = source_node
            nodes[target_node.node_id] = target_node

            relationship_type = str(
                record.get("relationship_type")
                or ""
            ).strip()

            relationship_id = str(
                record.get("relationship_id")
                or (
                    f"{source_node.node_id}:"
                    f"{relationship_type}:"
                    f"{target_node.node_id}"
                )
            ).strip()

            if not relationship_type:
                continue

            if relationship_id in seen_edge_ids:
                continue

            relationship_properties = (
                self._normalize_properties(
                    record.get(
                        "relationship_properties"
                    )
                )
            )

            edges.append(
                GraphEdge(
                    edge_id=relationship_id,
                    source_id=source_node.node_id,
                    target_id=target_node.node_id,
                    relationship_type=(
                        relationship_type
                    ),
                    properties=(
                        relationship_properties
                    ),
                )
            )

            seen_edge_ids.add(relationship_id)

            if len(edges) >= relationship_limit:
                break

        return GraphExplorerResult(
            nodes=list(nodes.values()),
            edges=edges,
            selected_labels=normalized_labels,
            selected_relationship_types=(
                normalized_relationship_types
            ),
            node_limit=node_limit,
            relationship_limit=relationship_limit,
        )

    @classmethod
    def _create_node(
        cls,
        node_id: Any,
        labels: Any,
        properties: Any,
    ) -> GraphNode | None:
        """Convert one Neo4j node record into a model."""
        normalized_node_id = str(
            node_id or ""
        ).strip()

        if not normalized_node_id:
            return None

        normalized_labels = (
            cls._normalize_label_collection(
                labels
            )
        )

        normalized_properties = (
            cls._normalize_properties(
                properties
            )
        )

        business_id = cls._optional_string(
            normalized_properties.get("id")
        )

        display_name = cls._resolve_display_name(
            node_id=normalized_node_id,
            labels=normalized_labels,
            properties=normalized_properties,
        )

        node_type = cls._resolve_node_type(
            normalized_labels
        )

        return GraphNode(
            node_id=normalized_node_id,
            business_id=business_id,
            display_name=display_name,
            node_type=node_type,
            labels=normalized_labels,
            properties=normalized_properties,
        )

    @staticmethod
    def _resolve_display_name(
        node_id: str,
        labels: list[str],
        properties: dict[str, Any],
    ) -> str:
        """Determine a readable node name."""
        candidate_fields = (
            "name",
            "title",
            "filename",
            "id",
            "document_id",
        )

        for field_name in candidate_fields:
            value = GraphExplorerService._optional_string(
                properties.get(field_name)
            )

            if value:
                return value

        node_type = (
            GraphExplorerService._resolve_node_type(
                labels
            )
        )

        return f"{node_type} {node_id[-8:]}"

    @staticmethod
    def _resolve_node_type(
        labels: list[str],
    ) -> str:
        """Choose the primary display label for a node."""
        for preferred_label in NODE_TYPE_PRIORITY:
            if preferred_label in labels:
                return preferred_label

        if labels:
            return labels[0]

        return "Node"

    @staticmethod
    def _normalize_filters(
        values: list[str] | None,
        uppercase: bool = False,
    ) -> list[str]:
        """Normalize and deduplicate filter values."""
        if not values:
            return []

        normalized_values: list[str] = []

        for value in values:
            normalized = str(value).strip()

            if uppercase:
                normalized = normalized.upper()

            if not normalized:
                continue

            if normalized not in normalized_values:
                normalized_values.append(normalized)

        return normalized_values

    @staticmethod
    def _normalize_label_collection(
        values: Any,
    ) -> list[str]:
        """Normalize a Neo4j label collection."""
        if not isinstance(values, list):
            return []

        labels = {
            str(value).strip()
            for value in values
            if str(value).strip()
        }

        return sorted(labels)

    @classmethod
    def _normalize_properties(
        cls,
        properties: Any,
    ) -> dict[str, Any]:
        """Convert Neo4j values into JSON-compatible values."""
        if not isinstance(properties, dict):
            return {}

        normalized_properties: dict[str, Any] = {}

        for key, value in properties.items():
            if value is None:
                continue

            normalized_properties[str(key)] = (
                cls._normalize_property_value(
                    value
                )
            )

        return normalized_properties

    @classmethod
    def _normalize_property_value(
        cls,
        value: Any,
    ) -> Any:
        """Normalize one Neo4j property value."""
        if value is None or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(value, list):
            return [
                cls._normalize_property_value(item)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): (
                    cls._normalize_property_value(
                        item
                    )
                )
                for key, item in value.items()
            }

        return str(value)

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """Convert a value into a nonempty optional string."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        return normalized_value or None
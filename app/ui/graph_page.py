"""Interactive Streamlit Neo4j graph explorer."""

import html
import json
import logging
from typing import Any, Final

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from app.database.neo4j_client import Neo4jClient
from app.services.graph_explorer_service import (
    GraphEdge,
    GraphExplorerError,
    GraphExplorerResult,
    GraphExplorerService,
    GraphNode,
)
from app.ui.constants import (
    DEFAULT_GRAPH_NODE_LIMIT,
    DEFAULT_GRAPH_RELATIONSHIP_LIMIT,
    GRAPH_HEIGHT,
    MAX_GRAPH_NODE_LIMIT,
    MAX_GRAPH_RELATIONSHIP_LIMIT,
    MIN_GRAPH_NODE_LIMIT,
    MIN_GRAPH_RELATIONSHIP_LIMIT,
)


logger = logging.getLogger(__name__)


GRAPH_RESULT_SESSION_KEY: Final[str] = (
    "graph_explorer_result"
)


DEFAULT_ENTITY_LABELS: Final[tuple[str, ...]] = (
    "Person",
    "Project",
    "Customer",
    "Team",
    "Technology",
)


DEFAULT_BUSINESS_RELATIONSHIPS: Final[
    tuple[str, ...]
] = (
    "MANAGES",
    "WORKS_ON",
    "SERVES",
    "USES",
    "MEMBER_OF",
    "OWNS",
)


NODE_COLORS: Final[dict[str, str]] = {
    "Person": "#60A5FA",
    "Project": "#A78BFA",
    "Customer": "#F59E0B",
    "Team": "#34D399",
    "Technology": "#F472B6",
    "Document": "#94A3B8",
    "Chunk": "#CBD5E1",
    "Node": "#E5E7EB",
}


NODE_SHAPES: Final[dict[str, str]] = {
    "Person": "dot",
    "Project": "hexagon",
    "Customer": "diamond",
    "Team": "triangle",
    "Technology": "box",
    "Document": "database",
    "Chunk": "text",
    "Node": "dot",
}


def render_graph_page() -> None:
    """Render the Neo4j graph-explorer page."""
    st.subheader("Interactive knowledge graph")

    st.caption(
        "Filter Neo4j entities and relationships, explore "
        "connections, and inspect stored node properties."
    )

    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                raise GraphExplorerError(
                    "Neo4j connection verification failed."
                )

            service = GraphExplorerService(client)

            node_labels = (
                service.get_available_node_labels()
            )

            relationship_types = (
                service
                .get_available_relationship_types()
            )

            if not node_labels:
                st.warning(
                    "The Neo4j database does not contain "
                    "any labeled nodes."
                )
                return

            filters = render_graph_filters(
                node_labels=node_labels,
                relationship_types=(
                    relationship_types
                ),
            )

            (
                selected_labels,
                selected_relationship_types,
                node_limit,
                relationship_limit,
                apply_clicked,
            ) = filters

            refresh_clicked = st.button(
                "Refresh graph",
                type="secondary",
                icon="🔄",
                key="refresh_graph_explorer",
            )

            should_load = (
                apply_clicked
                or refresh_clicked
                or GRAPH_RESULT_SESSION_KEY
                not in st.session_state
            )

            if should_load:
                with st.spinner(
                    "Loading the selected Neo4j graph..."
                ):
                    graph_result = service.load_graph(
                        node_labels=selected_labels,
                        relationship_types=(
                            selected_relationship_types
                        ),
                        node_limit=node_limit,
                        relationship_limit=(
                            relationship_limit
                        ),
                    )

                st.session_state[
                    GRAPH_RESULT_SESSION_KEY
                ] = graph_result.model_dump(
                    mode="json"
                )

            result = get_stored_graph_result()

            if result is None:
                st.info(
                    "Apply the filters to load the graph."
                )
                return

            render_graph_summary(result)

            if not result.nodes:
                st.warning(
                    "No graph records matched the selected "
                    "labels and relationship types."
                )

                st.caption(
                    "Select additional node labels or clear "
                    "the filters to search the full graph."
                )

                return

            render_graph_visualization(result)
            render_node_inspector(result)
            render_relationship_details(result)

    except GraphExplorerError as error:
        logger.exception(
            "The graph-explorer page failed."
        )

        st.error(
            "The knowledge graph could not be loaded."
        )

        st.caption(str(error))

    except ValueError as error:
        st.error(
            "The graph filters are invalid."
        )

        st.caption(str(error))

    except Exception as error:
        logger.exception(
            "Unexpected graph-explorer failure."
        )

        st.error(
            "An unexpected error occurred while "
            "loading the graph explorer."
        )

        st.caption(
            f"{type(error).__name__}: {error}"
        )


def render_graph_filters(
    node_labels: list[str],
    relationship_types: list[str],
) -> tuple[
    list[str],
    list[str],
    int,
    int,
    bool,
]:
    """Render graph filtering controls."""
    default_labels = [
        label
        for label in DEFAULT_ENTITY_LABELS
        if label in node_labels
    ]

    default_relationship_types = [
        relationship_type
        for relationship_type
        in DEFAULT_BUSINESS_RELATIONSHIPS
        if relationship_type
        in relationship_types
    ]

    with st.expander(
        "Graph filters",
        expanded=True,
        icon="🔎",
    ):
        with st.form(
            "graph_explorer_filter_form",
            border=False,
        ):
            first_column, second_column = (
                st.columns(2)
            )

            with first_column:
                selected_labels = st.multiselect(
                    label="Node labels",
                    options=node_labels,
                    default=default_labels,
                    key="graph_selected_labels",
                    help=(
                        "Only relationships whose source "
                        "and target use one of these labels "
                        "will be displayed. An empty selection "
                        "searches all labels."
                    ),
                )

            with second_column:
                selected_relationship_types = (
                    st.multiselect(
                        label="Relationship types",
                        options=relationship_types,
                        default=(
                            default_relationship_types
                        ),
                        key=(
                            "graph_selected_relationships"
                        ),
                        help=(
                            "An empty selection searches all "
                            "relationship types."
                        ),
                    )
                )

            third_column, fourth_column = (
                st.columns(2)
            )

            with third_column:
                node_limit = st.slider(
                    label="Maximum nodes",
                    min_value=MIN_GRAPH_NODE_LIMIT,
                    max_value=MAX_GRAPH_NODE_LIMIT,
                    value=DEFAULT_GRAPH_NODE_LIMIT,
                    step=5,
                    key="graph_node_limit",
                )

            with fourth_column:
                relationship_limit = st.slider(
                    label="Maximum relationships",
                    min_value=(
                        MIN_GRAPH_RELATIONSHIP_LIMIT
                    ),
                    max_value=(
                        MAX_GRAPH_RELATIONSHIP_LIMIT
                    ),
                    value=(
                        DEFAULT_GRAPH_RELATIONSHIP_LIMIT
                    ),
                    step=10,
                    key="graph_relationship_limit",
                )

            apply_clicked = (
                st.form_submit_button(
                    "Apply graph filters",
                    type="primary",
                    use_container_width=True,
                )
            )

    return (
        list(selected_labels),
        list(selected_relationship_types),
        int(node_limit),
        int(relationship_limit),
        bool(apply_clicked),
    )


def get_stored_graph_result(
) -> GraphExplorerResult | None:
    """Read the graph result from Streamlit session state."""
    raw_result = st.session_state.get(
        GRAPH_RESULT_SESSION_KEY
    )

    if isinstance(
        raw_result,
        GraphExplorerResult,
    ):
        return raw_result

    if not isinstance(raw_result, dict):
        return None

    try:
        return GraphExplorerResult.model_validate(
            raw_result
        )

    except Exception:
        logger.warning(
            "Invalid graph result found in session state."
        )

        st.session_state.pop(
            GRAPH_RESULT_SESSION_KEY,
            None,
        )

        return None


def render_graph_summary(
    result: GraphExplorerResult,
) -> None:
    """Display graph-result totals and filters."""
    node_types = {
        node.node_type
        for node in result.nodes
    }

    relationship_types = {
        edge.relationship_type
        for edge in result.edges
    }

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="Displayed nodes",
            value=len(result.nodes),
        )

    with second:
        st.metric(
            label="Relationships",
            value=len(result.edges),
        )

    with third:
        st.metric(
            label="Node types",
            value=len(node_types),
        )

    with fourth:
        st.metric(
            label="Relationship types",
            value=len(relationship_types),
        )


def render_graph_visualization(
    result: GraphExplorerResult,
) -> None:
    """Generate and display the interactive PyVis graph."""
    st.markdown("#### Graph visualization")

    st.caption(
        "Drag nodes, scroll to zoom, hover for details, "
        "and use the navigation controls inside the graph."
    )

    graph_html = build_pyvis_html(result)

    components.html(
        graph_html,
        height=GRAPH_HEIGHT + 40,
        scrolling=False,
    )


def build_pyvis_html(
    result: GraphExplorerResult,
) -> str:
    """Create a self-contained PyVis graph document."""
    network = Network(
        height=f"{GRAPH_HEIGHT}px",
        width="100%",
        directed=True,
        bgcolor="#0E1117",
        font_color="#F8FAFC",
        cdn_resources="in_line",
    )

    network.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true,
              "bindToWindow": false
            },
            "multiselect": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 180,
              "updateInterval": 25
            },
            "barnesHut": {
              "gravitationalConstant": -18000,
              "centralGravity": 0.25,
              "springLength": 170,
              "springConstant": 0.04,
              "damping": 0.15,
              "avoidOverlap": 0.4
            }
          },
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {
              "size": 14,
              "face": "Arial"
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.65
              }
            },
            "font": {
              "size": 10,
              "align": "middle",
              "strokeWidth": 3,
              "strokeColor": "#0E1117"
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          }
        }
        """
    )

    for node in result.nodes:
        network.add_node(
            n_id=node.node_id,
            label=truncate_text(
                node.display_name,
                maximum_length=40,
            ),
            title=create_node_tooltip(node),
            color=NODE_COLORS.get(
                node.node_type,
                NODE_COLORS["Node"],
            ),
            shape=NODE_SHAPES.get(
                node.node_type,
                NODE_SHAPES["Node"],
            ),
            size=calculate_node_size(
                node=node,
                edges=result.edges,
            ),
            group=node.node_type,
        )

    for edge in result.edges:
        network.add_edge(
            source=edge.source_id,
            to=edge.target_id,
            label=edge.relationship_type,
            title=create_edge_tooltip(edge),
        )

    return network.generate_html(
        name="knowledge_graph.html",
        local=True,
        notebook=False,
    )


def calculate_node_size(
    node: GraphNode,
    edges: list[GraphEdge],
) -> int:
    """Size nodes according to their displayed degree."""
    degree = sum(
        1
        for edge in edges
        if (
            edge.source_id == node.node_id
            or edge.target_id == node.node_id
        )
    )

    return min(
        18 + degree * 3,
        45,
    )


def create_node_tooltip(
    node: GraphNode,
) -> str:
    """Create escaped HTML for a node hover tooltip."""
    lines = [
        (
            f"<b>{html.escape(node.display_name)}</b>"
        ),
        (
            "Type: "
            f"{html.escape(node.node_type)}"
        ),
    ]

    if node.business_id:
        lines.append(
            "ID: "
            f"{html.escape(node.business_id)}"
        )

    for field_name in (
        "description",
        "role",
        "industry",
        "department",
        "category",
        "document_id",
        "chunk_index",
        "text",
    ):
        value = node.properties.get(field_name)

        if value is None:
            continue

        normalized_value = truncate_text(
            str(value),
            maximum_length=250,
        )

        lines.append(
            f"{html.escape(field_name.title())}: "
            f"{html.escape(normalized_value)}"
        )

    return "<br>".join(lines)


def create_edge_tooltip(
    edge: GraphEdge,
) -> str:
    """Create escaped HTML for an edge hover tooltip."""
    lines = [
        (
            "<b>"
            f"{html.escape(edge.relationship_type)}"
            "</b>"
        )
    ]

    for key, value in edge.properties.items():
        lines.append(
            f"{html.escape(str(key))}: "
            f"{html.escape(str(value))}"
        )

    return "<br>".join(lines)


def render_node_inspector(
    result: GraphExplorerResult,
) -> None:
    """Render searchable node details."""
    st.divider()
    st.markdown("#### Node inspector")

    nodes_by_option = {
        create_node_option(node): node
        for node in sorted(
            result.nodes,
            key=lambda item: (
                item.node_type,
                item.display_name.casefold(),
                item.node_id,
            ),
        )
    }

    node_options = [
        "Select a node..."
    ] + list(nodes_by_option)

    selected_option = st.selectbox(
        label="Search or select a displayed node",
        options=node_options,
        key="graph_node_inspector",
    )

    if selected_option == "Select a node...":
        st.caption(
            "Choose a node to inspect its properties "
            "and displayed relationships."
        )
        return

    selected_node = nodes_by_option[
        selected_option
    ]

    render_selected_node(
        selected_node=selected_node,
        result=result,
    )


def create_node_option(
    node: GraphNode,
) -> str:
    """Create a unique select-box label for a node."""
    short_id = (
        node.business_id
        or node.node_id[-10:]
    )

    return (
        f"{node.display_name} "
        f"· {node.node_type} "
        f"· {short_id}"
    )


def render_selected_node(
    selected_node: GraphNode,
    result: GraphExplorerResult,
) -> None:
    """Display one node and its connected edges."""
    first_column, second_column = st.columns(
        [2, 1]
    )

    with first_column:
        st.markdown(
            f"### {selected_node.display_name}"
        )

        st.caption(
            f"Node type: {selected_node.node_type}"
        )

        if selected_node.labels:
            st.write(
                "**Labels:** "
                + ", ".join(
                    f"`{label}`"
                    for label in selected_node.labels
                )
            )

    connected_edges = [
        edge
        for edge in result.edges
        if (
            edge.source_id
            == selected_node.node_id
            or edge.target_id
            == selected_node.node_id
        )
    ]

    with second_column:
        st.metric(
            label="Displayed connections",
            value=len(connected_edges),
        )

    st.markdown("##### Properties")

    if selected_node.properties:
        st.json(selected_node.properties)
    else:
        st.info(
            "This node does not have displayed properties."
        )

    st.markdown("##### Connected relationships")

    if not connected_edges:
        st.info(
            "No relationships are displayed for this node."
        )
        return

    node_lookup = {
        node.node_id: node
        for node in result.nodes
    }

    connection_rows: list[dict[str, Any]] = []

    for edge in connected_edges:
        if edge.source_id == selected_node.node_id:
            direction = "Outgoing"
            neighbor_id = edge.target_id
        else:
            direction = "Incoming"
            neighbor_id = edge.source_id

        neighbor = node_lookup.get(neighbor_id)

        connection_rows.append(
            {
                "Direction": direction,
                "Relationship": (
                    edge.relationship_type
                ),
                "Connected node": (
                    neighbor.display_name
                    if neighbor
                    else neighbor_id
                ),
                "Connected type": (
                    neighbor.node_type
                    if neighbor
                    else "Unknown"
                ),
            }
        )

    st.dataframe(
        connection_rows,
        use_container_width=True,
        hide_index=True,
    )


def render_relationship_details(
    result: GraphExplorerResult,
) -> None:
    """Display all relationships in a searchable table."""
    st.divider()
    st.markdown("#### Relationship details")

    if not result.edges:
        st.info(
            "No relationships matched the selected filters."
        )
        return

    node_lookup = {
        node.node_id: node
        for node in result.nodes
    }

    relationship_rows: list[dict[str, Any]] = []

    for edge in result.edges:
        source = node_lookup.get(edge.source_id)
        target = node_lookup.get(edge.target_id)

        relationship_rows.append(
            {
                "Source": (
                    source.display_name
                    if source
                    else edge.source_id
                ),
                "Source type": (
                    source.node_type
                    if source
                    else "Unknown"
                ),
                "Relationship": (
                    edge.relationship_type
                ),
                "Target": (
                    target.display_name
                    if target
                    else edge.target_id
                ),
                "Target type": (
                    target.node_type
                    if target
                    else "Unknown"
                ),
                "Properties": (
                    json.dumps(
                        edge.properties,
                        ensure_ascii=False,
                    )
                    if edge.properties
                    else ""
                ),
            }
        )

    st.dataframe(
        relationship_rows,
        use_container_width=True,
        hide_index=True,
    )

    download_content = json.dumps(
        {
            "nodes": [
                node.model_dump(mode="json")
                for node in result.nodes
            ],
            "edges": [
                edge.model_dump(mode="json")
                for edge in result.edges
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        label="Download displayed graph as JSON",
        data=download_content,
        file_name="knowledge_graph_export.json",
        mime="application/json",
        use_container_width=True,
    )


def truncate_text(
    text: str,
    maximum_length: int,
) -> str:
    """Truncate text to a readable display length."""
    normalized_text = " ".join(
        text.split()
    )

    if len(normalized_text) <= maximum_length:
        return normalized_text

    return (
        normalized_text[
            :maximum_length
        ].rstrip()
        + "..."
    )
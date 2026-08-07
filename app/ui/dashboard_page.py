"""Live Streamlit dashboard for the Neo4j knowledge graph."""

import logging
from typing import Final

import pandas as pd
import streamlit as st

from app.database.neo4j_client import Neo4jClient
from app.services.dashboard_service import (
    DashboardResult,
    DashboardService,
    DashboardServiceError,
)


logger = logging.getLogger(__name__)


DASHBOARD_SESSION_KEY: Final[str] = (
    "knowledge_graph_dashboard_result"
)


def render_dashboard_page() -> None:
    """Render the live Neo4j dashboard."""
    st.subheader("Knowledge graph health and activity")

    st.caption(
        "Review graph totals, entity distribution, "
        "relationship distribution, embedding coverage, "
        "and recent documents."
    )

    refresh_clicked = st.button(
        "Refresh dashboard",
        type="secondary",
        icon="🔄",
        use_container_width=False,
        key="refresh_knowledge_graph_dashboard",
    )

    try:
        should_load = (
            refresh_clicked
            or DASHBOARD_SESSION_KEY
            not in st.session_state
        )

        if should_load:
            with st.spinner(
                "Loading Neo4j dashboard statistics..."
            ):
                result = load_dashboard_result()

            st.session_state[
                DASHBOARD_SESSION_KEY
            ] = result.model_dump(
                mode="json"
            )

        result = get_stored_dashboard_result()

        if result is None:
            st.info(
                "Refresh the dashboard to load "
                "knowledge-graph statistics."
            )
            return

        render_dashboard_metrics(result)
        render_embedding_coverage(result)
        render_distribution_charts(result)
        render_recent_documents(result)
        render_dashboard_health_messages(result)

    except DashboardServiceError as error:
        logger.exception(
            "The knowledge-graph dashboard failed."
        )

        st.error(
            "The dashboard could not load Neo4j "
            "statistics."
        )

        st.caption(str(error))

    except ValueError as error:
        st.error(
            "The dashboard configuration is invalid."
        )

        st.caption(str(error))

    except Exception as error:
        logger.exception(
            "Unexpected dashboard-page failure."
        )

        st.error(
            "An unexpected error occurred while "
            "loading the dashboard."
        )

        st.caption(
            f"{type(error).__name__}: {error}"
        )


def load_dashboard_result() -> DashboardResult:
    """Load dashboard data through a verified Neo4j client."""
    with Neo4jClient.from_settings() as client:
        if not client.verify_connection():
            raise DashboardServiceError(
                "Neo4j connection verification failed."
            )

        service = DashboardService(client)

        return service.load_dashboard(
            recent_document_limit=10
        )


def get_stored_dashboard_result(
) -> DashboardResult | None:
    """Read a dashboard result from Streamlit session state."""
    raw_result = st.session_state.get(
        DASHBOARD_SESSION_KEY
    )

    if isinstance(raw_result, DashboardResult):
        return raw_result

    if not isinstance(raw_result, dict):
        return None

    try:
        return DashboardResult.model_validate(
            raw_result
        )

    except Exception:
        logger.warning(
            "Removed invalid dashboard session data."
        )

        st.session_state.pop(
            DASHBOARD_SESSION_KEY,
            None,
        )

        return None


def render_dashboard_metrics(
    result: DashboardResult,
) -> None:
    """Display top-level knowledge-graph totals."""
    totals = result.totals

    st.markdown("#### Graph totals")

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="Documents",
            value=f"{totals.documents:,}",
            help=(
                "Document nodes stored in the "
                "knowledge graph."
            ),
        )

    with second:
        st.metric(
            label="Text chunks",
            value=f"{totals.chunks:,}",
            help=(
                "Document chunks available for "
                "retrieval."
            ),
        )

    with third:
        st.metric(
            label="Business entities",
            value=f"{totals.entities:,}",
            help=(
                "People, projects, customers, teams, "
                "and technologies."
            ),
        )

    with fourth:
        st.metric(
            label="Relationships",
            value=(
                f"{totals.total_relationships:,}"
            ),
            help=(
                "All Neo4j relationships, including "
                "document and business relationships."
            ),
        )

    fifth, sixth, seventh = st.columns(3)

    with fifth:
        st.metric(
            label="Total nodes",
            value=f"{totals.total_nodes:,}",
        )

    with sixth:
        st.metric(
            label="Business relationships",
            value=(
                f"{totals.business_relationships:,}"
            ),
        )

    with seventh:
        st.metric(
            label="Chunks without embeddings",
            value=(
                f"{totals.chunks_without_embeddings:,}"
            ),
        )


def render_embedding_coverage(
    result: DashboardResult,
) -> None:
    """Display chunk embedding coverage."""
    totals = result.totals
    coverage = totals.embedding_coverage

    st.divider()
    st.markdown("#### Embedding coverage")

    first_column, second_column = st.columns(
        [3, 1]
    )

    with first_column:
        st.progress(coverage)

        st.caption(
            f"{totals.embedded_chunks:,} of "
            f"{totals.chunks:,} chunks contain "
            "vector embeddings."
        )

    with second_column:
        st.metric(
            label="Coverage",
            value=f"{coverage:.1%}",
        )

    if totals.chunks == 0:
        st.info(
            "No document chunks are currently stored."
        )

    elif totals.chunks_without_embeddings > 0:
        st.warning(
            f"{totals.chunks_without_embeddings:,} "
            "chunks still require vector embeddings."
        )

    else:
        st.success(
            "Every stored chunk has a vector embedding."
        )


def render_distribution_charts(
    result: DashboardResult,
) -> None:
    """Display entity and relationship distributions."""
    st.divider()
    st.markdown("#### Graph distribution")

    first_column, second_column = st.columns(2)

    with first_column:
        st.markdown("##### Entity types")

        if result.entity_distribution:
            entity_frame = pd.DataFrame(
                [
                    {
                        "Entity type": item.name,
                        "Count": item.count,
                    }
                    for item
                    in result.entity_distribution
                ]
            )

            st.bar_chart(
                entity_frame.set_index(
                    "Entity type"
                ),
                height=350,
            )

            st.dataframe(
                entity_frame,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No business entities are available."
            )

    with second_column:
        st.markdown("##### Relationship types")

        if result.relationship_distribution:
            relationship_frame = pd.DataFrame(
                [
                    {
                        "Relationship type": (
                            item.name
                        ),
                        "Count": item.count,
                    }
                    for item
                    in result.relationship_distribution
                ]
            )

            st.bar_chart(
                relationship_frame.set_index(
                    "Relationship type"
                ),
                height=350,
            )

            st.dataframe(
                relationship_frame,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No graph relationships are available."
            )


def render_recent_documents(
    result: DashboardResult,
) -> None:
    """Display recently ingested documents."""
    st.divider()
    st.markdown("#### Recent documents")

    if not result.recent_documents:
        st.info(
            "No documents have been ingested yet."
        )
        return

    document_rows = []

    for document in result.recent_documents:
        document_rows.append(
            {
                "Document": document.title,
                "File": document.filename or "—",
                "Type": (
                    document.document_type.upper()
                    if document.document_type
                    else "—"
                ),
                "Chunks": document.chunk_count,
                "Embedded": (
                    document.embedded_chunk_count
                ),
                "Coverage": (
                    f"{document.embedding_coverage:.0%}"
                ),
                "Updated": (
                    document.updated_at or "—"
                ),
                "Document ID": document.document_id,
            }
        )

    st.dataframe(
        document_rows,
        use_container_width=True,
        hide_index=True,
        column_order=[
            "Document",
            "File",
            "Type",
            "Chunks",
            "Embedded",
            "Coverage",
            "Updated",
            "Document ID",
        ],
    )

    selected_document = st.selectbox(
        label="Inspect document embedding coverage",
        options=[
            "Select a document..."
        ]
        + [
            (
                f"{document.title} "
                f"· {document.document_id}"
            )
            for document
            in result.recent_documents
        ],
        key="dashboard_document_inspector",
    )

    if selected_document == "Select a document...":
        return

    selected_index = [
        (
            f"{document.title} "
            f"· {document.document_id}"
        )
        for document
        in result.recent_documents
    ].index(selected_document)

    document = result.recent_documents[
        selected_index
    ]

    first, second, third = st.columns(3)

    with first:
        st.metric(
            label="Chunks",
            value=document.chunk_count,
        )

    with second:
        st.metric(
            label="Embedded chunks",
            value=document.embedded_chunk_count,
        )

    with third:
        st.metric(
            label="Coverage",
            value=(
                f"{document.embedding_coverage:.1%}"
            ),
        )

    st.progress(document.embedding_coverage)

    st.caption(
        f"Document ID: `{document.document_id}`"
    )


def render_dashboard_health_messages(
    result: DashboardResult,
) -> None:
    """Display useful graph-health guidance."""
    totals = result.totals

    st.divider()
    st.markdown("#### Health summary")

    if totals.total_nodes == 0:
        st.warning(
            "The Neo4j database is empty. Upload and "
            "ingest a document to initialize the graph."
        )
        return

    health_messages: list[str] = []

    if totals.documents > 0:
        health_messages.append(
            f"{totals.documents:,} documents are "
            "available for retrieval."
        )

    if totals.entities > 0:
        health_messages.append(
            f"{totals.entities:,} business entities "
            "have been extracted."
        )

    if totals.business_relationships > 0:
        health_messages.append(
            f"{totals.business_relationships:,} "
            "business relationships support graph "
            "context expansion."
        )

    if totals.embedding_coverage == 1.0:
        health_messages.append(
            "Embedding coverage is complete."
        )

    for message in health_messages:
        st.success(message)

    if totals.documents > 0 and totals.entities == 0:
        st.warning(
            "Documents exist, but no business entities "
            "have been extracted."
        )

    if (
        totals.entities > 0
        and totals.business_relationships == 0
    ):
        st.warning(
            "Entities exist, but no business "
            "relationships are currently stored."
        )
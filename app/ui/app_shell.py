"""Shared Streamlit application layout and navigation."""

from collections.abc import Callable
from typing import Final

import streamlit as st

from app.ui.chat_page import render_chat_page
from app.ui.constants import (
    APP_DESCRIPTION,
    APP_ICON,
    APP_TITLE,
    CHAT_PAGE_NAME,
    DASHBOARD_PAGE_NAME,
    DOCUMENT_PAGE_NAME,
    GRAPH_PAGE_NAME,
    NAVIGATION_PAGES,
    SESSION_MESSAGES_KEY,
    SESSION_SELECTED_PAGE_KEY,
)
from app.ui.dashboard_page import render_dashboard_page
from app.ui.document_page import render_document_page
from app.ui.graph_page import render_graph_page
from app.ui.session_state import clear_chat_history


PageRenderer = Callable[[], None]


PAGE_ICONS: Final[dict[str, str]] = {
    CHAT_PAGE_NAME: "💬",
    DOCUMENT_PAGE_NAME: "📄",
    GRAPH_PAGE_NAME: "🕸️",
    DASHBOARD_PAGE_NAME: "📊",
}


PAGE_DESCRIPTIONS: Final[dict[str, str]] = {
    CHAT_PAGE_NAME: (
        "Ask grounded questions and inspect the evidence "
        "supporting each response."
    ),
    DOCUMENT_PAGE_NAME: (
        "Upload enterprise documents and extract entities "
        "and relationships into Neo4j."
    ),
    GRAPH_PAGE_NAME: (
        "Explore entities, relationships, and document "
        "connections in the knowledge graph."
    ),
    DASHBOARD_PAGE_NAME: (
        "Review graph statistics, ingestion activity, "
        "and retrieval health."
    ),
}


PAGE_RENDERERS: Final[dict[str, PageRenderer]] = {
    CHAT_PAGE_NAME: render_chat_page,
    DOCUMENT_PAGE_NAME: render_document_page,
    GRAPH_PAGE_NAME: render_graph_page,
    DASHBOARD_PAGE_NAME: render_dashboard_page,
}


def format_navigation_page(page_name: str) -> str:
    """Add a visual icon to a navigation option."""
    icon = PAGE_ICONS.get(page_name, "•")

    return f"{icon} {page_name}"


def render_sidebar() -> str:
    """Render sidebar navigation and global controls."""
    with st.sidebar:
        st.markdown(f"# {APP_ICON} {APP_TITLE}")

        st.caption(APP_DESCRIPTION)

        st.divider()

        selected_page = st.radio(
            label="Navigation",
            options=NAVIGATION_PAGES,
            key=SESSION_SELECTED_PAGE_KEY,
            format_func=format_navigation_page,
        )

        st.divider()

        messages = st.session_state.get(
            SESSION_MESSAGES_KEY,
            [],
        )

        message_count = (
            len(messages)
            if isinstance(messages, list)
            else 0
        )

        st.metric(
            label="Chat messages",
            value=message_count,
        )

        clear_clicked = st.button(
            label="Clear chat history",
            type="secondary",
            use_container_width=True,
            disabled=message_count == 0,
        )

        if clear_clicked:
            clear_chat_history()
            st.rerun()

        st.divider()

        st.caption(
            "Grounded answers are generated from retrieved "
            "document chunks and Neo4j graph relationships."
        )

    return str(selected_page)


def render_page_header(page_name: str) -> None:
    """Render a consistent page title and description."""
    icon = PAGE_ICONS.get(page_name, APP_ICON)

    description = PAGE_DESCRIPTIONS.get(
        page_name,
        APP_DESCRIPTION,
    )

    st.title(f"{icon} {page_name}")
    st.markdown(description)
    st.divider()


def render_selected_page(page_name: str) -> None:
    """Render the selected application page."""
    render_page_header(page_name)

    renderer = PAGE_RENDERERS.get(page_name)

    if renderer is None:
        st.error(
            f"Unknown application page: {page_name}"
        )
        return

    renderer()


def render_footer() -> None:
    """Render the application footer."""
    st.divider()

    st.caption(
        "Knowledge Graph AI · Neo4j GraphRAG · Streamlit"
    )
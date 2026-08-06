"""Main Streamlit entry point for Knowledge Graph AI."""

import streamlit as st

from app.ui.app_shell import (
    render_footer,
    render_selected_page,
    render_sidebar,
)
from app.ui.constants import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
)
from app.ui.session_state import (
    initialize_session_state,
)


def configure_page() -> None:
    """Configure the browser tab and application layout."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": (
                "Knowledge Graph AI combines Neo4j, "
                "hybrid retrieval, and grounded AI answers."
            ),
        },
    )


def main() -> None:
    """Render the Streamlit application."""
    configure_page()
    initialize_session_state()

    selected_page = render_sidebar()

    render_selected_page(selected_page)
    render_footer()


if __name__ == "__main__":
    main()
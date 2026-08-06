"""Manage Streamlit session-state values."""

from collections.abc import Callable
from typing import Any, Final

import streamlit as st

from app.ui.constants import (
    CHAT_PAGE_NAME,
    SESSION_LAST_ANSWER_KEY,
    SESSION_MESSAGES_KEY,
    SESSION_SELECTED_GRAPH_NODE_KEY,
    SESSION_SELECTED_PAGE_KEY,
    SESSION_UPLOADED_DOCUMENTS_KEY,
)


SESSION_DEFAULT_FACTORIES: Final[
    dict[str, Callable[[], Any]]
] = {
    SESSION_MESSAGES_KEY: list,
    SESSION_SELECTED_PAGE_KEY: lambda: CHAT_PAGE_NAME,
    SESSION_LAST_ANSWER_KEY: lambda: None,
    SESSION_UPLOADED_DOCUMENTS_KEY: list,
    SESSION_SELECTED_GRAPH_NODE_KEY: lambda: None,
}


def initialize_session_state() -> None:
    """Create all required session-state values."""
    for key, factory in SESSION_DEFAULT_FACTORIES.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def clear_chat_history() -> None:
    """Remove chat messages and the latest answer."""
    st.session_state[SESSION_MESSAGES_KEY] = []
    st.session_state[SESSION_LAST_ANSWER_KEY] = None


def clear_uploaded_documents() -> None:
    """Remove uploaded-document UI history."""
    st.session_state[SESSION_UPLOADED_DOCUMENTS_KEY] = []


def get_chat_messages() -> list[dict[str, Any]]:
    """Return the current chat-message collection."""
    messages = st.session_state.get(
        SESSION_MESSAGES_KEY,
        [],
    )

    if not isinstance(messages, list):
        messages = []
        st.session_state[SESSION_MESSAGES_KEY] = messages

    return messages


def add_chat_message(
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a validated message to chat history."""
    normalized_role = role.strip().lower()
    normalized_content = content.strip()

    if normalized_role not in {
        "user",
        "assistant",
        "system",
    }:
        raise ValueError(
            "Chat role must be user, assistant, or system."
        )

    if not normalized_content:
        raise ValueError(
            "Chat-message content cannot be empty."
        )

    messages = get_chat_messages()

    messages.append(
        {
            "role": normalized_role,
            "content": normalized_content,
            "metadata": metadata or {},
        }
    )


def set_last_answer(answer: Any) -> None:
    """Store the latest GraphRAG answer."""
    st.session_state[SESSION_LAST_ANSWER_KEY] = answer


def get_last_answer() -> Any:
    """Return the latest GraphRAG answer."""
    return st.session_state.get(
        SESSION_LAST_ANSWER_KEY
    )
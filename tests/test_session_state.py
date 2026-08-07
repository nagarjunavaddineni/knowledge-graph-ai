"""Tests for Streamlit session state and chat utilities."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.ui import chat_page
from app.ui import session_state as session_state_module
from app.ui.chat_page import parse_stored_answer
from app.ui.constants import (
    CHAT_PAGE_NAME,
    SESSION_LAST_ANSWER_KEY,
    SESSION_MESSAGES_KEY,
    SESSION_SELECTED_GRAPH_NODE_KEY,
    SESSION_SELECTED_PAGE_KEY,
    SESSION_UPLOADED_DOCUMENTS_KEY,
)
from app.ui.session_state import (
    add_chat_message,
    clear_chat_history,
    clear_uploaded_documents,
    get_chat_messages,
    get_last_answer,
    initialize_session_state,
    set_last_answer,
)


def configure_session_state(
    monkeypatch: pytest.MonkeyPatch,
    initial_values: dict[str, object] | None = None,
) -> dict[str, object]:
    """Replace Streamlit session state with a test dictionary."""
    state: dict[str, object] = dict(
        initial_values or {}
    )

    monkeypatch.setattr(
        session_state_module.st,
        "session_state",
        state,
    )

    return state


def test_initialize_session_state_creates_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All required state values should be initialized."""
    state = configure_session_state(monkeypatch)

    initialize_session_state()

    assert state[SESSION_MESSAGES_KEY] == []

    assert state[SESSION_SELECTED_PAGE_KEY] == (
        CHAT_PAGE_NAME
    )

    assert state[SESSION_LAST_ANSWER_KEY] is None

    assert state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ] == []

    assert state[
        SESSION_SELECTED_GRAPH_NODE_KEY
    ] is None


def test_initialize_session_state_preserves_existing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing user-session values should not be replaced."""
    existing_messages = [
        {
            "role": "user",
            "content": "Existing question",
            "metadata": {},
        }
    ]

    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: existing_messages,
            SESSION_SELECTED_PAGE_KEY: "Dashboard",
            SESSION_LAST_ANSWER_KEY: {
                "answer": "Existing answer"
            },
        },
    )

    initialize_session_state()

    assert (
        state[SESSION_MESSAGES_KEY]
        is existing_messages
    )

    assert state[SESSION_SELECTED_PAGE_KEY] == (
        "Dashboard"
    )

    assert state[SESSION_LAST_ANSWER_KEY] == {
        "answer": "Existing answer"
    }

    assert state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ] == []


def test_new_sessions_receive_separate_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutable defaults should not be shared between sessions."""
    first_state = configure_session_state(
        monkeypatch
    )

    initialize_session_state()

    first_messages = first_state[
        SESSION_MESSAGES_KEY
    ]

    first_uploads = first_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ]

    second_state: dict[str, object] = {}

    monkeypatch.setattr(
        session_state_module.st,
        "session_state",
        second_state,
    )

    initialize_session_state()

    second_messages = second_state[
        SESSION_MESSAGES_KEY
    ]

    second_uploads = second_state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ]

    assert first_messages is not second_messages
    assert first_uploads is not second_uploads


def test_get_chat_messages_returns_existing_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing message list should be returned directly."""
    existing_messages = [
        {
            "role": "user",
            "content": "What uses Neo4j?",
            "metadata": {},
        }
    ]

    configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: existing_messages,
        },
    )

    result = get_chat_messages()

    assert result is existing_messages


def test_get_chat_messages_creates_missing_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing chat history should become an empty list."""
    state = configure_session_state(monkeypatch)

    result = get_chat_messages()

    assert result == []

    assert state[SESSION_MESSAGES_KEY] is result


def test_get_chat_messages_replaces_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupted chat history should be safely reset."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: (
                "invalid-chat-history"
            ),
        },
    )

    result = get_chat_messages()

    assert result == []

    assert state[SESSION_MESSAGES_KEY] is result


def test_add_user_chat_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid user message should be appended."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    add_chat_message(
        role="user",
        content=(
            "Which technologies does the project use?"
        ),
    )

    assert state[SESSION_MESSAGES_KEY] == [
        {
            "role": "user",
            "content": (
                "Which technologies does the project use?"
            ),
            "metadata": {},
        }
    ]


def test_add_assistant_message_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assistant answer metadata should be preserved."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    metadata = {
        "answer": {
            "answer": "The project uses Neo4j.",
            "confidence": 0.95,
        }
    }

    add_chat_message(
        role="assistant",
        content="The project uses Neo4j.",
        metadata=metadata,
    )

    messages = state[SESSION_MESSAGES_KEY]

    assert isinstance(messages, list)
    assert len(messages) == 1

    assert messages[0] == {
        "role": "assistant",
        "content": "The project uses Neo4j.",
        "metadata": metadata,
    }


def test_add_chat_message_normalizes_role_and_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer whitespace should be removed from messages."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    add_chat_message(
        role="  USER  ",
        content="  Tell me about the project.  ",
    )

    messages = state[SESSION_MESSAGES_KEY]

    assert isinstance(messages, list)

    assert messages[0]["role"] == "user"

    assert messages[0]["content"] == (
        "Tell me about the project."
    )


@pytest.mark.parametrize(
    "invalid_role",
    [
        "",
        "developer",
        "tool",
        "administrator",
    ],
)
def test_invalid_chat_roles_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invalid_role: str,
) -> None:
    """Only supported chat roles should be accepted."""
    configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "Chat role must be user, assistant, "
            "or system"
        ),
    ):
        add_chat_message(
            role=invalid_role,
            content="Test message",
        )


@pytest.mark.parametrize(
    "invalid_content",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_empty_chat_content_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invalid_content: str,
) -> None:
    """Blank chat content should not enter session history."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    with pytest.raises(
        ValueError,
        match="content cannot be empty",
    ):
        add_chat_message(
            role="user",
            content=invalid_content,
        )

    assert state[SESSION_MESSAGES_KEY] == []


def test_system_chat_message_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System messages should be accepted by state storage."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [],
        },
    )

    add_chat_message(
        role="system",
        content="Use only grounded evidence.",
    )

    messages = state[SESSION_MESSAGES_KEY]

    assert isinstance(messages, list)

    assert messages[0]["role"] == "system"


def test_clear_chat_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing chat should remove messages and last answer."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_MESSAGES_KEY: [
                {
                    "role": "user",
                    "content": "Question",
                    "metadata": {},
                },
                {
                    "role": "assistant",
                    "content": "Answer",
                    "metadata": {},
                },
            ],
            SESSION_LAST_ANSWER_KEY: {
                "answer": "Answer"
            },
            SESSION_SELECTED_PAGE_KEY: (
                CHAT_PAGE_NAME
            ),
            SESSION_UPLOADED_DOCUMENTS_KEY: [
                {
                    "document_id": "document-001",
                }
            ],
        },
    )

    clear_chat_history()

    assert state[SESSION_MESSAGES_KEY] == []

    assert state[SESSION_LAST_ANSWER_KEY] is None

    assert state[SESSION_SELECTED_PAGE_KEY] == (
        CHAT_PAGE_NAME
    )

    assert state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ] == [
        {
            "document_id": "document-001",
        }
    ]


def test_clear_uploaded_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uploaded-document UI history should be removable."""
    state = configure_session_state(
        monkeypatch,
        {
            SESSION_UPLOADED_DOCUMENTS_KEY: [
                {
                    "document_id": "document-001",
                },
                {
                    "document_id": "document-002",
                },
            ],
            SESSION_MESSAGES_KEY: [
                {
                    "role": "user",
                    "content": "Question",
                    "metadata": {},
                }
            ],
        },
    )

    clear_uploaded_documents()

    assert state[
        SESSION_UPLOADED_DOCUMENTS_KEY
    ] == []

    assert len(state[SESSION_MESSAGES_KEY]) == 1


def test_set_and_get_last_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The latest answer object should be stored unchanged."""
    state = configure_session_state(monkeypatch)

    answer = MagicMock(
        name="grounded_answer"
    )

    set_last_answer(answer)

    assert state[SESSION_LAST_ANSWER_KEY] is answer
    assert get_last_answer() is answer


def test_get_last_answer_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing latest-answer state should return None."""
    configure_session_state(monkeypatch)

    assert get_last_answer() is None


def test_parse_stored_answer_returns_none_without_answer() -> None:
    """Metadata without an answer should be ignored."""
    assert parse_stored_answer({}) is None

    assert parse_stored_answer(
        {
            "error": True,
        }
    ) is None


def test_parse_stored_answer_returns_none_for_non_dictionary() -> None:
    """Stored answers must use dictionary data."""
    assert parse_stored_answer(
        {
            "answer": "not-a-dictionary",
        }
    ) is None

    assert parse_stored_answer(
        {
            "answer": None,
        }
    ) is None


def test_parse_stored_answer_uses_model_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid stored metadata should be reconstructed."""
    stored_answer = {
        "answer": "The project uses Neo4j.",
        "answerable": True,
        "confidence": 0.95,
        "sources": [],
        "graph_facts": [],
    }

    expected_answer = MagicMock(
        name="validated_grounded_answer"
    )

    model_validate = MagicMock(
        return_value=expected_answer
    )

    monkeypatch.setattr(
        chat_page.GroundedAnswer,
        "model_validate",
        model_validate,
    )

    result = parse_stored_answer(
        {
            "answer": stored_answer,
        }
    )

    assert result is expected_answer

    model_validate.assert_called_once_with(
        stored_answer
    )


def test_invalid_stored_answer_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid stored answer metadata should be skipped."""
    try:
        chat_page.GroundedAnswer.model_validate(
            {}
        )

    except ValidationError as validation_error:
        expected_error = validation_error

    else:
        pytest.fail(
            "GroundedAnswer unexpectedly accepted "
            "an empty dictionary."
        )

    model_validate = MagicMock(
        side_effect=expected_error
    )

    monkeypatch.setattr(
        chat_page.GroundedAnswer,
        "model_validate",
        model_validate,
    )

    logger_warning = MagicMock()

    monkeypatch.setattr(
        chat_page.logger,
        "warning",
        logger_warning,
    )

    result = parse_stored_answer(
        {
            "answer": {
                "invalid": "metadata",
            }
        }
    )

    assert result is None

    logger_warning.assert_called_once_with(
        "Skipped invalid grounded-answer "
        "session metadata."
    )
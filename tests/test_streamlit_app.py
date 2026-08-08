"""Headless tests for the main Streamlit application."""

from collections.abc import Callable
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.ui.constants import (
    CHAT_PAGE_NAME,
    DASHBOARD_PAGE_NAME,
    DOCUMENT_PAGE_NAME,
    GRAPH_PAGE_NAME,
    NAVIGATION_PAGES,
    SESSION_LAST_ANSWER_KEY,
    SESSION_MESSAGES_KEY,
    SESSION_SELECTED_GRAPH_NODE_KEY,
    SESSION_SELECTED_PAGE_KEY,
    SESSION_UPLOADED_DOCUMENTS_KEY,
)


APP_FILE = "streamlit_app.py"
APP_TEST_TIMEOUT_SECONDS = 20


def create_app_test() -> AppTest:
    """Create an AppTest instance for the main application."""
    return AppTest.from_file(
        APP_FILE,
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    )


def assert_no_uncaught_exceptions(
    app_test: AppTest,
) -> None:
    """Assert that the Streamlit script completed safely."""
    exception_messages = [
        str(exception.value)
        for exception in app_test.exception
    ]

    assert exception_messages == []


def find_button_by_label(
    app_test: AppTest,
    label: str,
):
    """Find one rendered button using its visible label."""
    matching_buttons = [
        button
        for button in app_test.button
        if button.label == label
    ]

    assert len(matching_buttons) == 1

    return matching_buttons[0]


def find_metric_value(
    app_test: AppTest,
    label: str,
) -> str:
    """Return the value of a metric with the selected label."""
    matching_metrics = [
        metric
        for metric in app_test.metric
        if metric.label == label
    ]

    assert len(matching_metrics) == 1

    return str(matching_metrics[0].value)


def render_graph_test_page() -> None:
    """Render a database-free graph page for routing tests."""
    st.subheader("Graph explorer test page")

    st.success(
        "Graph page routing completed successfully."
    )


def render_dashboard_test_page() -> None:
    """Render a database-free dashboard for routing tests."""
    st.subheader("Dashboard test page")

    st.success(
        "Dashboard page routing completed successfully."
    )


def test_application_starts_on_chat_page() -> None:
    """The application should start on the chat screen."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    navigation = app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    assert navigation.value == CHAT_PAGE_NAME

    assert len(app_test.title) == 1
    assert CHAT_PAGE_NAME in app_test.title[0].value

    subheader_values = [
        subheader.value
        for subheader in app_test.subheader
    ]

    assert (
        "Ask your enterprise knowledge graph"
        in subheader_values
    )

    chat_input = app_test.chat_input(
        key="graphrag_chat_input"
    )

    assert chat_input is not None


def test_application_initializes_session_state() -> None:
    """Startup should create every required state value."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_PAGE_KEY
        ]
        == CHAT_PAGE_NAME
    )

    assert (
        app_test.session_state[
            SESSION_LAST_ANSWER_KEY
        ]
        is None
    )

    assert (
        app_test.session_state[
            SESSION_UPLOADED_DOCUMENTS_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_GRAPH_NODE_KEY
        ]
        is None
    )


def test_sidebar_contains_complete_navigation() -> None:
    """The sidebar should expose every application page."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.sidebar.radio) == 1

    navigation = app_test.sidebar.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    assert list(navigation.options) == list(
        NAVIGATION_PAGES
    )


def test_navigation_opens_document_upload_page() -> None:
    """Selecting Upload Documents should render the uploader."""
    app_test = create_app_test().run()

    navigation = app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    navigation.set_value(
        DOCUMENT_PAGE_NAME
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.radio(
            key=SESSION_SELECTED_PAGE_KEY
        ).value
        == DOCUMENT_PAGE_NAME
    )

    assert DOCUMENT_PAGE_NAME in (
        app_test.title[0].value
    )

    subheader_values = [
        subheader.value
        for subheader in app_test.subheader
    ]

    assert (
        "Upload and ingest a document"
        in subheader_values
    )

    uploader = app_test.file_uploader(
        key="knowledge_graph_document_upload"
    )

    assert uploader.label == "Select a document"


def test_navigation_can_return_to_chat_page() -> None:
    """Navigation should support moving between pages."""
    app_test = create_app_test().run()

    app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    ).set_value(
        DOCUMENT_PAGE_NAME
    ).run()

    assert DOCUMENT_PAGE_NAME in (
        app_test.title[0].value
    )

    app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    ).set_value(
        CHAT_PAGE_NAME
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert CHAT_PAGE_NAME in (
        app_test.title[0].value
    )

    assert (
        app_test.chat_input(
            key="graphrag_chat_input"
        )
        is not None
    )


@pytest.mark.parametrize(
    (
        "page_name",
        "patch_target",
        "renderer",
        "expected_subheader",
        "expected_success",
    ),
    [
        (
            GRAPH_PAGE_NAME,
            "app.ui.app_shell.render_graph_page",
            render_graph_test_page,
            "Graph explorer test page",
            (
                "Graph page routing completed "
                "successfully."
            ),
        ),
        (
            DASHBOARD_PAGE_NAME,
            "app.ui.app_shell.render_dashboard_page",
            render_dashboard_test_page,
            "Dashboard test page",
            (
                "Dashboard page routing completed "
                "successfully."
            ),
        ),
    ],
)
def test_navigation_routes_to_service_pages(
    page_name: str,
    patch_target: str,
    renderer: Callable[[], None],
    expected_subheader: str,
    expected_success: str,
) -> None:
    """Graph and dashboard routing should select their renderer."""
    with patch(
        patch_target,
        new=renderer,
    ):
        app_test = create_app_test().run()

        app_test.radio(
            key=SESSION_SELECTED_PAGE_KEY
        ).set_value(
            page_name
        ).run()

        assert_no_uncaught_exceptions(app_test)

        assert page_name in app_test.title[0].value

        subheader_values = [
            subheader.value
            for subheader in app_test.subheader
        ]

        success_values = [
            success.value
            for success in app_test.success
        ]

        assert expected_subheader in subheader_values
        assert expected_success in success_values


def test_existing_chat_history_is_rendered() -> None:
    """Stored chat messages should appear after startup."""
    app_test = create_app_test()

    app_test.session_state[
        SESSION_MESSAGES_KEY
    ] = [
        {
            "role": "user",
            "content": (
                "Which technologies does the "
                "project use?"
            ),
            "metadata": {},
        },
        {
            "role": "assistant",
            "content": (
                "The project uses Neo4j and Python."
            ),
            "metadata": {},
        },
    ]

    app_test.session_state[
        SESSION_SELECTED_PAGE_KEY
    ] = CHAT_PAGE_NAME

    app_test.run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.chat_message) == 2

    assert (
        app_test.chat_message[0].avatar
        == "user"
    )

    assert (
        app_test.chat_message[1].avatar
        == "assistant"
    )

    assert (
        "Which technologies"
        in app_test.chat_message[0].markdown[0].value
    )

    assert (
        "Neo4j and Python"
        in app_test.chat_message[1].markdown[0].value
    )

    assert find_metric_value(
        app_test,
        "Chat messages",
    ) == "2"


def test_clear_chat_button_removes_history() -> None:
    """The sidebar button should reset chat-related state."""
    app_test = create_app_test()

    app_test.session_state[
        SESSION_MESSAGES_KEY
    ] = [
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
    ]

    app_test.session_state[
        SESSION_LAST_ANSWER_KEY
    ] = {
        "answer": "Answer",
    }

    app_test.session_state[
        SESSION_SELECTED_PAGE_KEY
    ] = CHAT_PAGE_NAME

    app_test.run()

    clear_button = find_button_by_label(
        app_test,
        "Clear chat history",
    )

    clear_button.click().run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_LAST_ANSWER_KEY
        ]
        is None
    )

    assert len(app_test.chat_message) == 0

    assert find_metric_value(
        app_test,
        "Chat messages",
    ) == "0"


def test_unknown_page_displays_controlled_error() -> None:
    """An invalid page should show an error without crashing."""
    source = """
from app.ui.app_shell import render_selected_page

render_selected_page("Invalid Test Page")
"""

    app_test = AppTest.from_string(
        source,
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.error) == 1

    assert (
        "Unknown application page"
        in app_test.error[0].value
    )


def test_repeated_reruns_do_not_duplicate_state() -> None:
    """Normal reruns should preserve valid session values."""
    app_test = create_app_test().run()

    app_test.run()
    app_test.run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_UPLOADED_DOCUMENTS_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_PAGE_KEY
        ]
        == CHAT_PAGE_NAME)
"""Headless tests for the main Streamlit application."""

from collections.abc import Callable
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.ui.constants import (
    CHAT_PAGE_NAME,
    DASHBOARD_PAGE_NAME,
    DOCUMENT_PAGE_NAME,
    GRAPH_PAGE_NAME,
    NAVIGATION_PAGES,
    SESSION_LAST_ANSWER_KEY,
    SESSION_MESSAGES_KEY,
    SESSION_SELECTED_GRAPH_NODE_KEY,
    SESSION_SELECTED_PAGE_KEY,
    SESSION_UPLOADED_DOCUMENTS_KEY,
)


APP_FILE = "streamlit_app.py"
APP_TEST_TIMEOUT_SECONDS = 20


def create_app_test() -> AppTest:
    """Create an AppTest instance for the main application."""
    return AppTest.from_file(
        APP_FILE,
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    )


def assert_no_uncaught_exceptions(
    app_test: AppTest,
) -> None:
    """Assert that the Streamlit script completed safely."""
    exception_messages = [
        str(exception.value)
        for exception in app_test.exception
    ]

    assert exception_messages == []


def find_button_by_label(
    app_test: AppTest,
    label: str,
):
    """Find one rendered button using its visible label."""
    matching_buttons = [
        button
        for button in app_test.button
        if button.label == label
    ]

    assert len(matching_buttons) == 1

    return matching_buttons[0]


def find_metric_value(
    app_test: AppTest,
    label: str,
) -> str:
    """Return the value of a metric with the selected label."""
    matching_metrics = [
        metric
        for metric in app_test.metric
        if metric.label == label
    ]

    assert len(matching_metrics) == 1

    return str(matching_metrics[0].value)


def render_graph_test_page() -> None:
    """Render a database-free graph page for routing tests."""
    st.subheader("Graph explorer test page")

    st.success(
        "Graph page routing completed successfully."
    )


def render_dashboard_test_page() -> None:
    """Render a database-free dashboard for routing tests."""
    st.subheader("Dashboard test page")

    st.success(
        "Dashboard page routing completed successfully."
    )


def test_application_starts_on_chat_page() -> None:
    """The application should start on the chat screen."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    navigation = app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    assert navigation.value == CHAT_PAGE_NAME

    assert len(app_test.title) == 1
    assert CHAT_PAGE_NAME in app_test.title[0].value

    subheader_values = [
        subheader.value
        for subheader in app_test.subheader
    ]

    assert (
        "Ask your enterprise knowledge graph"
        in subheader_values
    )

    chat_input = app_test.chat_input(
        key="graphrag_chat_input"
    )

    assert chat_input is not None


def test_application_initializes_session_state() -> None:
    """Startup should create every required state value."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_PAGE_KEY
        ]
        == CHAT_PAGE_NAME
    )

    assert (
        app_test.session_state[
            SESSION_LAST_ANSWER_KEY
        ]
        is None
    )

    assert (
        app_test.session_state[
            SESSION_UPLOADED_DOCUMENTS_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_GRAPH_NODE_KEY
        ]
        is None
    )


def test_sidebar_contains_complete_navigation() -> None:
    """The sidebar should expose every application page."""
    app_test = create_app_test().run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.sidebar.radio) == 1

    navigation = app_test.sidebar.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    assert list(navigation.options) == list(
        NAVIGATION_PAGES
    )


def test_navigation_opens_document_upload_page() -> None:
    """Selecting Upload Documents should render the uploader."""
    app_test = create_app_test().run()

    navigation = app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    )

    navigation.set_value(
        DOCUMENT_PAGE_NAME
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.radio(
            key=SESSION_SELECTED_PAGE_KEY
        ).value
        == DOCUMENT_PAGE_NAME
    )

    assert DOCUMENT_PAGE_NAME in (
        app_test.title[0].value
    )

    subheader_values = [
        subheader.value
        for subheader in app_test.subheader
    ]

    assert (
        "Upload and ingest a document"
        in subheader_values
    )

    uploader = app_test.file_uploader(
        key="knowledge_graph_document_upload"
    )

    assert uploader.label == "Select a document"


def test_navigation_can_return_to_chat_page() -> None:
    """Navigation should support moving between pages."""
    app_test = create_app_test().run()

    app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    ).set_value(
        DOCUMENT_PAGE_NAME
    ).run()

    assert DOCUMENT_PAGE_NAME in (
        app_test.title[0].value
    )

    app_test.radio(
        key=SESSION_SELECTED_PAGE_KEY
    ).set_value(
        CHAT_PAGE_NAME
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert CHAT_PAGE_NAME in (
        app_test.title[0].value
    )

    assert (
        app_test.chat_input(
            key="graphrag_chat_input"
        )
        is not None
    )


@pytest.mark.parametrize(
    (
        "page_name",
        "patch_target",
        "renderer",
        "expected_subheader",
        "expected_success",
    ),
    [
        (
            GRAPH_PAGE_NAME,
            "app.ui.app_shell.render_graph_page",
            render_graph_test_page,
            "Graph explorer test page",
            (
                "Graph page routing completed "
                "successfully."
            ),
        ),
        (
            DASHBOARD_PAGE_NAME,
            "app.ui.app_shell.render_dashboard_page",
            render_dashboard_test_page,
            "Dashboard test page",
            (
                "Dashboard page routing completed "
                "successfully."
            ),
        ),
    ],
)
def test_navigation_routes_to_service_pages(
    page_name: str,
    patch_target: str,
    renderer: Callable[[], None],
    expected_subheader: str,
    expected_success: str,
) -> None:
    """Graph and dashboard routing should select their renderer."""
    with patch(
        patch_target,
        new=renderer,
    ):
        app_test = create_app_test().run()

        app_test.radio(
            key=SESSION_SELECTED_PAGE_KEY
        ).set_value(
            page_name
        ).run()

        assert_no_uncaught_exceptions(app_test)

        assert page_name in app_test.title[0].value

        subheader_values = [
            subheader.value
            for subheader in app_test.subheader
        ]

        success_values = [
            success.value
            for success in app_test.success
        ]

        assert expected_subheader in subheader_values
        assert expected_success in success_values


def test_existing_chat_history_is_rendered() -> None:
    """Stored chat messages should appear after startup."""
    app_test = create_app_test()

    app_test.session_state[
        SESSION_MESSAGES_KEY
    ] = [
        {
            "role": "user",
            "content": (
                "Which technologies does the "
                "project use?"
            ),
            "metadata": {},
        },
        {
            "role": "assistant",
            "content": (
                "The project uses Neo4j and Python."
            ),
            "metadata": {},
        },
    ]

    app_test.session_state[
        SESSION_SELECTED_PAGE_KEY
    ] = CHAT_PAGE_NAME

    app_test.run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.chat_message) == 2

    assert (
        app_test.chat_message[0].avatar
        == "user"
    )

    assert (
        app_test.chat_message[1].avatar
        == "assistant"
    )

    assert (
        "Which technologies"
        in app_test.chat_message[0].markdown[0].value
    )

    assert (
        "Neo4j and Python"
        in app_test.chat_message[1].markdown[0].value
    )

    assert find_metric_value(
        app_test,
        "Chat messages",
    ) == "2"


def test_clear_chat_button_removes_history() -> None:
    """The sidebar button should reset chat-related state."""
    app_test = create_app_test()

    app_test.session_state[
        SESSION_MESSAGES_KEY
    ] = [
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
    ]

    app_test.session_state[
        SESSION_LAST_ANSWER_KEY
    ] = {
        "answer": "Answer",
    }

    app_test.session_state[
        SESSION_SELECTED_PAGE_KEY
    ] = CHAT_PAGE_NAME

    app_test.run()

    clear_button = find_button_by_label(
        app_test,
        "Clear chat history",
    )

    clear_button.click().run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_LAST_ANSWER_KEY
        ]
        is None
    )

    assert len(app_test.chat_message) == 0

    assert find_metric_value(
        app_test,
        "Chat messages",
    ) == "0"


def test_unknown_page_displays_controlled_error() -> None:
    """An invalid page should show an error without crashing."""
    source = """
from app.ui.app_shell import render_selected_page

render_selected_page("Invalid Test Page")
"""

    app_test = AppTest.from_string(
        source,
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()

    assert_no_uncaught_exceptions(app_test)

    assert len(app_test.error) == 1

    assert (
        "Unknown application page"
        in app_test.error[0].value
    )


def test_repeated_reruns_do_not_duplicate_state() -> None:
    """Normal reruns should preserve valid session values."""
    app_test = create_app_test().run()

    app_test.run()
    app_test.run()

    assert_no_uncaught_exceptions(app_test)

    assert (
        app_test.session_state[
            SESSION_MESSAGES_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_UPLOADED_DOCUMENTS_KEY
        ]
        == []
    )

    assert (
        app_test.session_state[
            SESSION_SELECTED_PAGE_KEY
        ]
        == CHAT_PAGE_NAME
    )
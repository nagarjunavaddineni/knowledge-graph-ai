"""Constants shared across the Streamlit user interface."""

from typing import Final


APP_TITLE: Final[str] = "Knowledge Graph AI"
APP_ICON: Final[str] = "🧠"
APP_LAYOUT: Final[str] = "wide"

APP_DESCRIPTION: Final[str] = (
    "Ask grounded questions, inspect sources, upload documents, "
    "and explore the Neo4j knowledge graph."
)

SUPPORTED_UPLOAD_TYPES: Final[tuple[str, ...]] = (
    "txt",
    "pdf",
    "csv",
    "json",
)

MAX_UPLOAD_SIZE_MB: Final[int] = 25

DEFAULT_QUESTION: Final[str] = (
    "Which technologies are used by the "
    "Enterprise GraphRAG Assistant?"
)

SAMPLE_QUESTIONS: Final[tuple[str, ...]] = (
    (
        "Which technologies are used by the "
        "Enterprise GraphRAG Assistant?"
    ),
    "Who manages the Enterprise GraphRAG Assistant?",
    "Which customer does the project serve?",
    "Who works on the knowledge graph project?",
)

DEFAULT_RETRIEVAL_LIMIT: Final[int] = 5
MIN_RETRIEVAL_LIMIT: Final[int] = 1
MAX_RETRIEVAL_LIMIT: Final[int] = 15

DEFAULT_VECTOR_SCORE: Final[float] = 0.0
MIN_VECTOR_SCORE: Final[float] = 0.0
MAX_VECTOR_SCORE: Final[float] = 1.0

DEFAULT_MAX_CONNECTIONS: Final[int] = 10
MIN_MAX_CONNECTIONS: Final[int] = 1
MAX_MAX_CONNECTIONS: Final[int] = 25

GRAPH_HEIGHT: Final[int] = 650

DEFAULT_GRAPH_NODE_LIMIT: Final[int] = 75
MIN_GRAPH_NODE_LIMIT: Final[int] = 10
MAX_GRAPH_NODE_LIMIT: Final[int] = 300

DEFAULT_GRAPH_RELATIONSHIP_LIMIT: Final[int] = 150
MIN_GRAPH_RELATIONSHIP_LIMIT: Final[int] = 10
MAX_GRAPH_RELATIONSHIP_LIMIT: Final[int] = 750

SESSION_MESSAGES_KEY: Final[str] = "messages"
SESSION_SELECTED_PAGE_KEY: Final[str] = "selected_page"
SESSION_LAST_ANSWER_KEY: Final[str] = "last_answer"
SESSION_UPLOADED_DOCUMENTS_KEY: Final[str] = (
    "uploaded_documents"
)
SESSION_SELECTED_GRAPH_NODE_KEY: Final[str] = (
    "selected_graph_node"
)

CHAT_PAGE_NAME: Final[str] = "Ask the Graph"
DOCUMENT_PAGE_NAME: Final[str] = "Upload Documents"
GRAPH_PAGE_NAME: Final[str] = "Explore Graph"
DASHBOARD_PAGE_NAME: Final[str] = "Dashboard"

NAVIGATION_PAGES: Final[tuple[str, ...]] = (
    CHAT_PAGE_NAME,
    DOCUMENT_PAGE_NAME,
    GRAPH_PAGE_NAME,
    DASHBOARD_PAGE_NAME,
)
"""Interactive Streamlit chat interface for GraphRAG."""

import logging
from typing import Any, Final

import streamlit as st
from pydantic import ValidationError

from app.database.neo4j_client import Neo4jClient
from app.services.answer_models import GroundedAnswer
from app.services.graphrag_service import (
    GraphRAGAnswerError,
    GraphRAGAnswerService,
)
from app.ui.constants import (
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_RETRIEVAL_LIMIT,
    DEFAULT_VECTOR_SCORE,
    MAX_MAX_CONNECTIONS,
    MAX_RETRIEVAL_LIMIT,
    MAX_VECTOR_SCORE,
    MIN_MAX_CONNECTIONS,
    MIN_RETRIEVAL_LIMIT,
    MIN_VECTOR_SCORE,
)
from app.ui.session_state import (
    add_chat_message,
    get_chat_messages,
    set_last_answer,
)


logger = logging.getLogger(__name__)


class ChatPageError(RuntimeError):
    """Raised when the chat interface cannot complete a request."""


SAMPLE_QUESTION_BUTTONS: Final[
    tuple[tuple[str, str], ...]
] = (
    (
        "Technologies",
        (
            "Which technologies are used by the "
            "Enterprise GraphRAG Assistant?"
        ),
    ),
    (
        "Project manager",
        (
            "Who manages the Enterprise "
            "GraphRAG Assistant?"
        ),
    ),
    (
        "Customer",
        (
            "Which customer is served by the "
            "Enterprise GraphRAG Assistant?"
        ),
    ),
    (
        "Team members",
        (
            "Who works on the Enterprise "
            "GraphRAG Assistant?"
        ),
    ),
)


def render_chat_page() -> None:
    """Render the grounded GraphRAG chat experience."""
    st.subheader("Ask your enterprise knowledge graph")

    st.caption(
        "Answers are generated from retrieved document "
        "chunks and explicit Neo4j graph relationships."
    )

    (
        retrieval_limit,
        minimum_vector_score,
        max_connections,
    ) = render_retrieval_settings()

    selected_sample = render_sample_questions()

    render_chat_history()

    typed_question = st.chat_input(
        placeholder=(
            "Ask about projects, people, customers, "
            "teams, or technologies..."
        ),
        max_chars=1000,
        key="graphrag_chat_input",
    )

    submitted_question = (
        selected_sample or typed_question
    )

    if submitted_question:
        process_question(
            question=submitted_question,
            retrieval_limit=retrieval_limit,
            minimum_vector_score=minimum_vector_score,
            max_connections=max_connections,
        )


def render_retrieval_settings() -> tuple[int, float, int]:
    """Render configurable GraphRAG retrieval controls."""
    with st.expander(
        "Retrieval settings",
        expanded=False,
        icon="⚙️",
    ):
        first_column, second_column, third_column = (
            st.columns(3)
        )

        with first_column:
            retrieval_limit = st.slider(
                label="Retrieved chunks",
                min_value=MIN_RETRIEVAL_LIMIT,
                max_value=MAX_RETRIEVAL_LIMIT,
                value=DEFAULT_RETRIEVAL_LIMIT,
                step=1,
                key="chat_retrieval_limit",
                help=(
                    "Maximum number of hybrid-ranked "
                    "document chunks used as evidence."
                ),
            )

        with second_column:
            minimum_vector_score = st.slider(
                label="Minimum vector score",
                min_value=MIN_VECTOR_SCORE,
                max_value=MAX_VECTOR_SCORE,
                value=DEFAULT_VECTOR_SCORE,
                step=0.05,
                key="chat_minimum_vector_score",
                help=(
                    "Minimum semantic similarity required "
                    "for vector-search candidates."
                ),
            )

        with third_column:
            max_connections = st.slider(
                label="Connections per entity",
                min_value=MIN_MAX_CONNECTIONS,
                max_value=MAX_MAX_CONNECTIONS,
                value=DEFAULT_MAX_CONNECTIONS,
                step=1,
                key="chat_max_connections",
                help=(
                    "Maximum graph relationships retrieved "
                    "for each mentioned entity."
                ),
            )

    return (
        int(retrieval_limit),
        float(minimum_vector_score),
        int(max_connections),
    )


def render_sample_questions() -> str | None:
    """Render buttons for common demonstration questions."""
    st.caption("Try a sample question")

    columns = st.columns(
        len(SAMPLE_QUESTION_BUTTONS)
    )

    selected_question: str | None = None

    for index, (
        label,
        question,
    ) in enumerate(SAMPLE_QUESTION_BUTTONS):
        with columns[index]:
            clicked = st.button(
                label,
                key=f"sample_question_{index}",
                use_container_width=True,
            )

            if clicked:
                selected_question = question

    return selected_question


def render_chat_history() -> None:
    """Render all previously stored chat messages."""
    for message in get_chat_messages():
        role = str(
            message.get("role") or "assistant"
        )

        if role not in {
            "user",
            "assistant",
        }:
            role = "assistant"

        content = str(
            message.get("content") or ""
        )

        metadata = message.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        with st.chat_message(role):
            st.write(content)

            if role == "assistant":
                answer = parse_stored_answer(
                    metadata
                )

                if answer is not None:
                    render_answer_details(answer)

                if metadata.get("error"):
                    st.caption(
                        "The request did not complete "
                        "successfully."
                    )


def process_question(
    question: str,
    retrieval_limit: int,
    minimum_vector_score: float,
    max_connections: int,
) -> None:
    """Submit a question and display the grounded answer."""
    normalized_question = " ".join(
        question.split()
    )

    if not normalized_question:
        st.warning(
            "Enter a question before submitting."
        )
        return

    add_chat_message(
        role="user",
        content=normalized_question,
    )

    with st.chat_message("user"):
        st.write(normalized_question)

    with st.chat_message("assistant"):
        with st.status(
            "Searching the knowledge graph...",
            expanded=True,
        ) as status:
            status.write(
                "Running keyword and semantic retrieval."
            )

            try:
                result = run_graphrag_question(
                    question=normalized_question,
                    retrieval_limit=retrieval_limit,
                    minimum_vector_score=(
                        minimum_vector_score
                    ),
                    max_connections=max_connections,
                )

                status.write(
                    "Expanding entities through Neo4j "
                    "relationships."
                )

                status.write(
                    "Validating the answer against "
                    "retrieved sources."
                )

                status.update(
                    label=(
                        "Grounded answer generated"
                        if result.answerable
                        else "Insufficient evidence"
                    ),
                    state="complete",
                    expanded=False,
                )

            except ChatPageError as error:
                logger.exception(
                    "The Streamlit chat request failed."
                )

                status.update(
                    label="Unable to generate an answer",
                    state="error",
                    expanded=True,
                )

                error_message = (
                    "I could not complete this request. "
                    "Verify the Neo4j and OpenAI "
                    "configuration, then try again."
                )

                st.error(error_message)
                st.caption(str(error))

                add_chat_message(
                    role="assistant",
                    content=error_message,
                    metadata={
                        "error": True,
                    },
                )

                return

        st.write(result.answer)
        render_answer_details(result)

    set_last_answer(result)

    add_chat_message(
        role="assistant",
        content=result.answer,
        metadata={
            "answer": result.model_dump(
                mode="json"
            )
        },
    )


def run_graphrag_question(
    question: str,
    retrieval_limit: int,
    minimum_vector_score: float,
    max_connections: int,
) -> GroundedAnswer:
    """Run the complete Day 4 GraphRAG pipeline."""
    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                raise ChatPageError(
                    "Neo4j connection verification failed."
                )

            service = (
                GraphRAGAnswerService.from_settings(
                    client
                )
            )

            return service.answer_question(
                question=question,
                limit=retrieval_limit,
                minimum_vector_score=(
                    minimum_vector_score
                ),
                max_connections_per_entity=(
                    max_connections
                ),
            )

    except GraphRAGAnswerError as error:
        raise ChatPageError(
            str(error)
        ) from error

    except ChatPageError:
        raise

    except ValueError as error:
        raise ChatPageError(
            f"Configuration validation failed: {error}"
        ) from error

    except Exception as error:
        logger.exception(
            "Unexpected GraphRAG chat failure."
        )

        raise ChatPageError(
            "An unexpected error occurred while "
            "running the GraphRAG pipeline."
        ) from error


def render_answer_details(
    answer: GroundedAnswer,
) -> None:
    """Render confidence, sources, and graph evidence."""
    confidence_column, source_column = st.columns(
        2
    )

    with confidence_column:
        st.metric(
            label="Confidence",
            value=f"{answer.confidence:.0%}",
        )

    with source_column:
        st.metric(
            label="Supporting sources",
            value=len(answer.sources),
        )

    if not answer.answerable:
        st.warning(
            answer.insufficient_evidence_reason
            or (
                "The available knowledge graph does "
                "not contain enough evidence."
            )
        )

    if answer.sources:
        with st.expander(
            f"Sources ({len(answer.sources)})",
            expanded=False,
            icon="📚",
        ):
            for index, source in enumerate(
                answer.sources
            ):
                source_name = (
                    source.document_title
                    or source.document_id
                    or "Unknown document"
                )

                st.markdown(
                    f"**[{source.source_id}] "
                    f"{source_name}**"
                )

                first_column, second_column = (
                    st.columns(2)
                )

                with first_column:
                    st.caption(
                        f"Chunk: {source.chunk_id}"
                    )

                with second_column:
                    st.caption(
                        "Retrieval score: "
                        f"{source.retrieval_score:.6f}"
                    )

                st.write(source.excerpt)

                if index < len(answer.sources) - 1:
                    st.divider()

    if answer.graph_facts:
        with st.expander(
            f"Graph facts ({len(answer.graph_facts)})",
            expanded=False,
            icon="🕸️",
        ):
            for fact in answer.graph_facts:
                st.markdown(f"- `{fact}`")


def parse_stored_answer(
    metadata: dict[str, Any],
) -> GroundedAnswer | None:
    """Rebuild a stored GroundedAnswer from session metadata."""
    raw_answer = metadata.get("answer")

    if not isinstance(raw_answer, dict):
        return None

    try:
        return GroundedAnswer.model_validate(
            raw_answer
        )

    except ValidationError:
        logger.warning(
            "Skipped invalid grounded-answer "
            "session metadata."
        )

        return None
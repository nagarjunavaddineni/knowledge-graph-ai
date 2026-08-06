"""Unit tests for grounded GraphRAG answer generation."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.context_models import (
    ContextEntity,
    ExpandedChunkContext,
    GraphContextResult,
)
from app.retrieval.graph_context_expander import (
    GraphContextExpansionError,
)
from app.retrieval.hybrid_retriever import (
    HybridRetrievalError,
)
from app.retrieval.models import (
    GraphConnection,
    HybridChunkSearchResult,
    HybridSearchResult,
)
from app.services.graphrag_service import (
    MAX_SOURCE_EXCERPT_CHARACTERS,
    GraphRAGAnswerError,
    GraphRAGAnswerService,
    ModelAnswerDraft,
)


TEST_QUESTION = (
    "Which technologies are used by the "
    "Enterprise GraphRAG Assistant?"
)


def create_hybrid_result() -> HybridSearchResult:
    """Create a mocked hybrid retrieval result."""
    chunk = HybridChunkSearchResult(
        chunk_id="chunk-001",
        document_id="document-001",
        document_title="Project Overview",
        chunk_index=0,
        text=(
            "The Enterprise GraphRAG Assistant "
            "uses Neo4j and Python."
        ),
        rrf_score=0.0325,
        fulltext_score=3.1,
        vector_score=0.94,
        fulltext_rank=1,
        vector_rank=1,
        matched_by=[
            "fulltext",
            "vector",
        ],
        entities=[],
    )

    return HybridSearchResult(
        question=TEST_QUESTION,
        rrf_constant=60,
        candidate_limit=15,
        fulltext_candidate_count=1,
        vector_candidate_count=1,
        results=[chunk],
    )


def create_graph_context(
    text: str = (
        "The Enterprise GraphRAG Assistant "
        "uses Neo4j and Python."
    ),
) -> GraphContextResult:
    """Create expanded graph context for testing."""
    project_entity = ContextEntity(
        entity_id="project-001",
        name="Enterprise GraphRAG Assistant",
        entity_type="Project",
        description="Enterprise knowledge assistant",
        connections=[
            GraphConnection(
                relationship_type="USES",
                direction="outgoing",
                neighbor_id="technology-001",
                neighbor_name="Neo4j",
                neighbor_type="Technology",
            ),
            GraphConnection(
                relationship_type="USES",
                direction="outgoing",
                neighbor_id="technology-002",
                neighbor_name="Python",
                neighbor_type="Technology",
            ),
        ],
    )

    chunk = ExpandedChunkContext(
        chunk_id="chunk-001",
        document_id="document-001",
        document_title="Project Overview",
        chunk_index=0,
        text=text,
        rrf_score=0.0325,
        fulltext_score=3.1,
        vector_score=0.94,
        fulltext_rank=1,
        vector_rank=1,
        matched_by=[
            "fulltext",
            "vector",
        ],
        entities=[project_entity],
    )

    return GraphContextResult(
        question=TEST_QUESTION,
        retrieved_chunk_count=1,
        expanded_chunk_count=1,
        chunks=[chunk],
    )


def create_completion(
    draft: ModelAnswerDraft | None,
    refusal: str | None = None,
    include_choice: bool = True,
) -> MagicMock:
    """Create a mocked OpenAI parsed completion."""
    completion = MagicMock()

    if not include_choice:
        completion.choices = []
        return completion

    message = MagicMock()
    message.refusal = refusal
    message.parsed = draft

    choice = MagicMock()
    choice.message = message

    completion.choices = [choice]

    return completion


def create_service(
    draft: ModelAnswerDraft | None = None,
) -> tuple[
    GraphRAGAnswerService,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create a GraphRAG service with mocked dependencies."""
    retriever = MagicMock()
    context_expander = MagicMock()
    openai_client = MagicMock()

    retriever.search.return_value = create_hybrid_result()

    context_expander.expand.return_value = (
        create_graph_context()
    )

    default_draft = ModelAnswerDraft(
        answerable=True,
        answer=(
            "The Enterprise GraphRAG Assistant uses "
            "Neo4j and Python. [S1]"
        ),
        cited_source_ids=["S1"],
        insufficient_evidence_reason=None,
        confidence=0.95,
    )

    openai_client.chat.completions.parse.return_value = (
        create_completion(
            draft=(
                default_draft
                if draft is None
                else draft
            )
        )
    )

    service = GraphRAGAnswerService(
        retriever=retriever,
        context_expander=context_expander,
        api_key="test-openai-key",
        model="test-answer-model",
        openai_client=openai_client,
    )

    return (
        service,
        retriever,
        context_expander,
        openai_client,
    )


def test_successful_answer_contains_valid_source() -> None:
    """A supported answer should return its cited source."""
    (
        service,
        retriever,
        context_expander,
        openai_client,
    ) = create_service()

    result = service.answer_question(
        question=f"  {TEST_QUESTION}  ",
        limit=5,
        minimum_vector_score=0.5,
        max_connections_per_entity=7,
    )

    assert result.question == TEST_QUESTION
    assert result.answerable is True
    assert result.confidence == 0.95

    assert result.answer == (
        "The Enterprise GraphRAG Assistant uses "
        "Neo4j and Python. [S1]"
    )

    assert result.insufficient_evidence_reason is None

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.source_id == "S1"
    assert source.chunk_id == "chunk-001"
    assert source.document_id == "document-001"
    assert source.document_title == "Project Overview"
    assert source.chunk_index == 0
    assert source.retrieval_score == 0.0325
    assert "Neo4j and Python" in source.excerpt

    assert result.graph_facts == [
        (
            "Enterprise GraphRAG Assistant "
            "-[USES]-> Neo4j"
        ),
        (
            "Enterprise GraphRAG Assistant "
            "-[USES]-> Python"
        ),
    ]

    retriever.search.assert_called_once_with(
        question=TEST_QUESTION,
        limit=5,
        minimum_vector_score=0.5,
    )

    hybrid_result = retriever.search.return_value

    context_expander.expand.assert_called_once_with(
        retrieval_result=hybrid_result,
        max_connections_per_entity=7,
    )

    parse_arguments = (
        openai_client.chat.completions.parse
        .call_args
        .kwargs
    )

    assert parse_arguments["model"] == (
        "test-answer-model"
    )
    assert parse_arguments["temperature"] == 0
    assert parse_arguments["response_format"] is (
        ModelAnswerDraft
    )

    messages = parse_arguments["messages"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "[S1]" in messages[1]["content"]
    assert "Project Overview" in messages[1]["content"]
    assert "Graph facts:" in messages[1]["content"]


def test_duplicate_source_ids_are_normalized() -> None:
    """Duplicate source references should appear only once."""
    draft = ModelAnswerDraft(
        answerable=True,
        answer="The project uses Neo4j. [S1]",
        cited_source_ids=[
            "s1",
            " S1 ",
            "S1",
        ],
        confidence=0.9,
    )

    service, _, _, _ = create_service(draft)

    result = service.answer_question(TEST_QUESTION)

    assert result.answerable is True
    assert len(result.sources) == 1
    assert result.sources[0].source_id == "S1"


def test_no_retrieved_chunks_returns_insufficient_answer() -> None:
    """No context should produce an insufficient-evidence result."""
    (
        service,
        _,
        context_expander,
        openai_client,
    ) = create_service()

    context_expander.expand.return_value = (
        GraphContextResult(
            question=TEST_QUESTION,
            retrieved_chunk_count=0,
            expanded_chunk_count=0,
            chunks=[],
        )
    )

    result = service.answer_question(TEST_QUESTION)

    assert result.answerable is False
    assert result.confidence == 0.0
    assert result.sources == []
    assert result.graph_facts == []

    assert result.insufficient_evidence_reason == (
        "No relevant document chunks were found "
        "in the knowledge graph."
    )

    assert (
        "does not contain enough evidence"
        in result.answer
    )

    openai_client.chat.completions.parse.assert_not_called()


def test_oversized_context_without_source_is_insufficient() -> None:
    """Unusable oversized evidence should not reach the model."""
    (
        service,
        _,
        context_expander,
        openai_client,
    ) = create_service()

    context_expander.expand.return_value = (
        create_graph_context(
            text="x" * 17_000
        )
    )

    result = service.answer_question(TEST_QUESTION)

    assert result.answerable is False

    assert result.insufficient_evidence_reason == (
        "Relevant records were found, but no usable "
        "source text was available."
    )

    openai_client.chat.completions.parse.assert_not_called()


def test_model_can_report_insufficient_evidence() -> None:
    """The model may explicitly mark evidence as insufficient."""
    draft = ModelAnswerDraft(
        answerable=False,
        answer=(
            "The graph does not provide the "
            "requested budget."
        ),
        cited_source_ids=[],
        insufficient_evidence_reason=(
            "No budget information exists in the "
            "retrieved sources."
        ),
        confidence=0.2,
    )

    service, _, _, _ = create_service(draft)

    result = service.answer_question(
        "What is the project budget?"
    )

    assert result.answerable is False
    assert result.confidence == 0.2

    assert result.insufficient_evidence_reason == (
        "No budget information exists in the "
        "retrieved sources."
    )

    assert result.sources == []
    assert result.graph_facts == []


def test_invalid_source_id_causes_insufficient_answer() -> None:
    """An answer without a valid source must be rejected."""
    draft = ModelAnswerDraft(
        answerable=True,
        answer="The project uses an unknown technology.",
        cited_source_ids=["S99"],
        confidence=0.9,
    )

    service, _, _, _ = create_service(draft)

    result = service.answer_question(TEST_QUESTION)

    assert result.answerable is False
    assert result.confidence == 0.0
    assert result.sources == []

    assert result.insufficient_evidence_reason == (
        "The generated answer did not include a "
        "valid supporting source."
    )


def test_retrieval_failure_is_wrapped() -> None:
    """Hybrid retrieval errors should become answer errors."""
    service, retriever, _, _ = create_service()

    retriever.search.side_effect = HybridRetrievalError(
        "Hybrid search failed"
    )

    with pytest.raises(
        GraphRAGAnswerError,
        match="Unable to retrieve relevant knowledge",
    ):
        service.answer_question(TEST_QUESTION)


def test_context_expansion_failure_is_wrapped() -> None:
    """Graph expansion errors should become answer errors."""
    (
        service,
        _,
        context_expander,
        _,
    ) = create_service()

    context_expander.expand.side_effect = (
        GraphContextExpansionError(
            "Graph expansion failed"
        )
    )

    with pytest.raises(
        GraphRAGAnswerError,
        match="Unable to expand the retrieved graph context",
    ):
        service.answer_question(TEST_QUESTION)


def test_empty_openai_choices_are_rejected() -> None:
    """OpenAI must return at least one answer choice."""
    (
        service,
        _,
        _,
        openai_client,
    ) = create_service()

    openai_client.chat.completions.parse.return_value = (
        create_completion(
            draft=None,
            include_choice=False,
        )
    )

    with pytest.raises(
        GraphRAGAnswerError,
        match="returned no answer choices",
    ):
        service.answer_question(TEST_QUESTION)


def test_openai_refusal_is_rejected() -> None:
    """A model refusal should produce a clear error."""
    (
        service,
        _,
        _,
        openai_client,
    ) = create_service()

    openai_client.chat.completions.parse.return_value = (
        create_completion(
            draft=None,
            refusal="Unable to process this request.",
        )
    )

    with pytest.raises(
        GraphRAGAnswerError,
        match="request was refused",
    ):
        service.answer_question(TEST_QUESTION)


def test_missing_parsed_answer_is_rejected() -> None:
    """The completion must contain parsed structured output."""
    (
        service,
        _,
        _,
        openai_client,
    ) = create_service()

    openai_client.chat.completions.parse.return_value = (
        create_completion(
            draft=None,
            refusal=None,
        )
    )

    with pytest.raises(
        GraphRAGAnswerError,
        match="returned no parsed answer",
    ):
        service.answer_question(TEST_QUESTION)


def test_blank_question_is_rejected() -> None:
    """A blank GraphRAG question is invalid."""
    service, retriever, _, _ = create_service()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        service.answer_question("   ")

    retriever.search.assert_not_called()


def test_source_excerpt_is_truncated() -> None:
    """Long source text should produce a bounded excerpt."""
    text = "a" * (
        MAX_SOURCE_EXCERPT_CHARACTERS + 100
    )

    excerpt = GraphRAGAnswerService._create_excerpt(
        text
    )

    assert excerpt.endswith("...")

    assert len(excerpt) == (
        MAX_SOURCE_EXCERPT_CHARACTERS + 3
    )


def test_short_source_excerpt_is_unchanged() -> None:
    """Short source text should remain readable."""
    excerpt = GraphRAGAnswerService._create_excerpt(
        "Neo4j stores connected enterprise data."
    )

    assert excerpt == (
        "Neo4j stores connected enterprise data."
    )


def test_missing_api_key_is_rejected() -> None:
    """The answer service requires an OpenAI API key."""
    with pytest.raises(
        ValueError,
        match="OpenAI API key is required",
    ):
        GraphRAGAnswerService(
            retriever=MagicMock(),
            context_expander=MagicMock(),
            api_key="",
            model="test-model",
            openai_client=MagicMock(),
        )


def test_missing_answer_model_is_rejected() -> None:
    """An answer-generation model name is required."""
    with pytest.raises(
        ValueError,
        match="answer-generation model is required",
    ):
        GraphRAGAnswerService(
            retriever=MagicMock(),
            context_expander=MagicMock(),
            api_key="test-key",
            model="",
            openai_client=MagicMock(),
        )


def test_invalid_timeout_is_rejected() -> None:
    """The OpenAI timeout must be positive."""
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero",
    ):
        GraphRAGAnswerService(
            retriever=MagicMock(),
            context_expander=MagicMock(),
            api_key="test-key",
            model="test-model",
            openai_client=MagicMock(),
            timeout_seconds=0,
        )


def test_negative_retry_count_is_rejected() -> None:
    """The OpenAI retry count cannot be negative."""
    with pytest.raises(
        ValueError,
        match="max_retries cannot be negative",
    ):
        GraphRAGAnswerService(
            retriever=MagicMock(),
            context_expander=MagicMock(),
            api_key="test-key",
            model="test-model",
            openai_client=MagicMock(),
            max_retries=-1,
        )
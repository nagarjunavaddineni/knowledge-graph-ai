"""Unit tests for reciprocal-rank hybrid retrieval."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
)
from app.retrieval.fulltext_retriever import (
    FullTextRetrievalError,
)
from app.retrieval.hybrid_retriever import (
    HybridRetrievalError,
    RankFusionHybridRetriever,
)
from app.retrieval.models import (
    ChunkSearchResult,
    EntityReference,
)
from app.retrieval.vector_retriever import (
    VectorRetrievalError,
)


def create_chunk(
    chunk_id: str,
    score: float,
    text: str | None = None,
    entities: list[EntityReference] | None = None,
) -> ChunkSearchResult:
    """Create one mocked retrieval result."""
    return ChunkSearchResult(
        chunk_id=chunk_id,
        document_id="document-001",
        document_title="Project Overview",
        chunk_index=0,
        text=text or f"Text for {chunk_id}",
        score=score,
        entities=entities or [],
    )


def create_retriever(
    fulltext_retriever: MagicMock | None = None,
    vector_retriever: MagicMock | None = None,
    default_top_k: int = 5,
    candidate_multiplier: int = 3,
    rrf_constant: int = 60,
) -> RankFusionHybridRetriever:
    """Create a hybrid retriever for testing."""
    return RankFusionHybridRetriever(
        fulltext_retriever=(
            fulltext_retriever or MagicMock()
        ),
        vector_retriever=(
            vector_retriever or MagicMock()
        ),
        default_top_k=default_top_k,
        candidate_multiplier=candidate_multiplier,
        rrf_constant=rrf_constant,
    )


def test_search_combines_fulltext_and_vector_results() -> None:
    """Chunks found by both methods should receive both scores."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = [
        create_chunk(
            chunk_id="chunk-001",
            score=3.0,
        ),
        create_chunk(
            chunk_id="chunk-002",
            score=2.0,
        ),
    ]

    vector_retriever.search.return_value = [
        create_chunk(
            chunk_id="chunk-002",
            score=0.95,
        ),
        create_chunk(
            chunk_id="chunk-003",
            score=0.90,
        ),
    ]

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    result = retriever.search(
        question="Which project uses Neo4j?",
        limit=3,
        minimum_vector_score=0.5,
    )

    assert result.question == (
        "Which project uses Neo4j?"
    )
    assert result.rrf_constant == 60
    assert result.candidate_limit == 9
    assert result.fulltext_candidate_count == 2
    assert result.vector_candidate_count == 2

    assert len(result.results) == 3

    first_result = result.results[0]

    assert first_result.chunk_id == "chunk-002"
    assert first_result.matched_by == [
        "fulltext",
        "vector",
    ]
    assert first_result.fulltext_rank == 2
    assert first_result.vector_rank == 1
    assert first_result.fulltext_score == 2.0
    assert first_result.vector_score == 0.95

    expected_score = (
        1.0 / (60 + 2)
        + 1.0 / (60 + 1)
    )

    assert first_result.rrf_score == pytest.approx(
        expected_score
    )

    assert result.results[1].chunk_id == "chunk-001"
    assert result.results[1].matched_by == [
        "fulltext"
    ]

    assert result.results[2].chunk_id == "chunk-003"
    assert result.results[2].matched_by == [
        "vector"
    ]


def test_search_passes_candidate_limit_to_retrievers() -> None:
    """Each retriever should receive the expanded candidate limit."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = []
    vector_retriever.search.return_value = []

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
        candidate_multiplier=4,
    )

    result = retriever.search(
        question="What technologies are used?",
        limit=4,
        minimum_vector_score=0.6,
    )

    assert result.candidate_limit == 16

    fulltext_retriever.search_chunks.assert_called_once_with(
        question="What technologies are used?",
        limit=16,
    )

    vector_retriever.search.assert_called_once_with(
        question="What technologies are used?",
        limit=16,
        minimum_score=0.6,
    )


def test_candidate_limit_is_capped_at_fifty() -> None:
    """Hybrid candidate retrieval should remain bounded."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = []
    vector_retriever.search.return_value = []

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
        default_top_k=20,
        candidate_multiplier=4,
    )

    result = retriever.search(
        question="Knowledge graph",
        limit=20,
    )

    assert result.candidate_limit == 50

    fulltext_retriever.search_chunks.assert_called_once_with(
        question="Knowledge graph",
        limit=50,
    )

    vector_retriever.search.assert_called_once_with(
        question="Knowledge graph",
        limit=50,
        minimum_score=0.0,
    )


def test_search_uses_default_top_k() -> None:
    """The configured default should be used without a limit."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = [
        create_chunk(
            chunk_id=f"chunk-{index}",
            score=float(10 - index),
        )
        for index in range(1, 8)
    ]

    vector_retriever.search.return_value = []

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
        default_top_k=3,
    )

    result = retriever.search("Neo4j projects")

    assert len(result.results) == 3
    assert result.candidate_limit == 9


def test_empty_result_lists_return_empty_hybrid_result() -> None:
    """No candidates should produce an empty result list."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = []
    vector_retriever.search.return_value = []

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    result = retriever.search("Missing information")

    assert result.results == []
    assert result.fulltext_candidate_count == 0
    assert result.vector_candidate_count == 0


def test_entities_are_merged_without_duplicates() -> None:
    """Entity references from both retrievers should be merged."""
    project_entity = EntityReference(
        entity_id="project-001",
        name="Enterprise GraphRAG Assistant",
        entity_type="Project",
    )

    technology_entity = EntityReference(
        entity_id="technology-001",
        name="Neo4j",
        entity_type="Technology",
    )

    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = [
        create_chunk(
            chunk_id="chunk-001",
            score=3.0,
            entities=[
                project_entity,
                technology_entity,
            ],
        )
    ]

    vector_retriever.search.return_value = [
        create_chunk(
            chunk_id="chunk-001",
            score=0.94,
            entities=[
                project_entity,
            ],
        )
    ]

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    result = retriever.search("Neo4j")

    assert len(result.results) == 1
    assert len(result.results[0].entities) == 2

    entity_ids = {
        entity.entity_id
        for entity in result.results[0].entities
    }

    assert entity_ids == {
        "project-001",
        "technology-001",
    }


def test_fulltext_failure_is_wrapped() -> None:
    """Full-text failures should become hybrid errors."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.side_effect = (
        FullTextRetrievalError(
            "Full-text index unavailable"
        )
    )

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    with pytest.raises(
        HybridRetrievalError,
        match="full-text search",
    ):
        retriever.search("Neo4j")

    vector_retriever.search.assert_not_called()


def test_vector_failure_is_wrapped() -> None:
    """Vector failures should become hybrid errors."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = []

    vector_retriever.search.side_effect = (
        VectorRetrievalError(
            "Vector index unavailable"
        )
    )

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    with pytest.raises(
        HybridRetrievalError,
        match="semantic vector search",
    ):
        retriever.search("Neo4j")


def test_embedding_failure_is_wrapped() -> None:
    """Embedding errors during vector retrieval should be wrapped."""
    fulltext_retriever = MagicMock()
    vector_retriever = MagicMock()

    fulltext_retriever.search_chunks.return_value = []

    vector_retriever.search.side_effect = (
        EmbeddingGenerationError(
            "Embedding service unavailable"
        )
    )

    retriever = create_retriever(
        fulltext_retriever=fulltext_retriever,
        vector_retriever=vector_retriever,
    )

    with pytest.raises(
        HybridRetrievalError,
        match="semantic vector search",
    ):
        retriever.search("Neo4j")


def test_blank_question_is_rejected() -> None:
    """A blank hybrid-search question is invalid."""
    retriever = create_retriever()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retriever.search("   ")


def test_non_string_question_is_rejected() -> None:
    """The hybrid-search question must be a string."""
    retriever = create_retriever()

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        retriever.search(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 51],
)
def test_invalid_result_limits_are_rejected(
    invalid_limit: int,
) -> None:
    """Final hybrid result limits must be between 1 and 50."""
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.search(
            "Neo4j",
            limit=invalid_limit,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [-0.01, 1.01],
)
def test_invalid_vector_thresholds_are_rejected(
    invalid_score: float,
) -> None:
    """Vector thresholds must remain between zero and one."""
    retriever = create_retriever()

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        retriever.search(
            "Neo4j",
            minimum_vector_score=invalid_score,
        )


def test_invalid_constructor_values_are_rejected() -> None:
    """Hybrid configuration values must be positive."""
    with pytest.raises(
        ValueError,
        match="default_top_k",
    ):
        create_retriever(
            default_top_k=0
        )

    with pytest.raises(
        ValueError,
        match="candidate_multiplier",
    ):
        create_retriever(
            candidate_multiplier=0
        )

    with pytest.raises(
        ValueError,
        match="rrf_constant",
    ):
        create_retriever(
            rrf_constant=0
        )
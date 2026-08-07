"""Unit tests for the Neo4j dashboard service."""

from unittest.mock import MagicMock

import pytest

from app.services.dashboard_service import (
    DashboardService,
    DashboardServiceError,
    DashboardTotals,
    DistributionItem,
    RecentDocument,
)


def create_service(
    mock_client: MagicMock | None = None,
) -> tuple[DashboardService, MagicMock]:
    """Create a dashboard service with a mocked Neo4j client."""
    client = mock_client or MagicMock()

    return DashboardService(client), client


def test_load_dashboard_parses_all_statistics() -> None:
    """The service should combine all dashboard query results."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [
            {
                "total_nodes": 20,
                "total_relationships": 35,
                "documents": 2,
                "chunks": 8,
                "embedded_chunks": 7,
                "entities": 10,
                "business_relationships": 12,
            }
        ],
        [
            {
                "name": "Technology",
                "count": 4,
            },
            {
                "name": "Project",
                "count": 3,
            },
            {
                "name": "Person",
                "count": 3,
            },
        ],
        [
            {
                "name": "MENTIONS",
                "count": 10,
            },
            {
                "name": "USES",
                "count": 5,
            },
            {
                "name": "MANAGES",
                "count": 2,
            },
        ],
        [
            {
                "document_id": "document-002",
                "title": "Architecture Guide",
                "filename": "architecture.pdf",
                "document_type": "pdf",
                "chunk_count": 5,
                "embedded_chunk_count": 5,
                "updated_at": (
                    "2026-08-05T18:30:00Z"
                ),
            },
            {
                "document_id": "document-001",
                "title": "Project Overview",
                "filename": "project_overview.txt",
                "document_type": "txt",
                "chunk_count": 3,
                "embedded_chunk_count": 2,
                "updated_at": (
                    "2026-08-04T14:00:00Z"
                ),
            },
        ],
    ]

    result = service.load_dashboard(
        recent_document_limit=10
    )

    assert result.totals.total_nodes == 20
    assert result.totals.total_relationships == 35
    assert result.totals.documents == 2
    assert result.totals.chunks == 8
    assert result.totals.embedded_chunks == 7
    assert result.totals.entities == 10

    assert (
        result.totals.business_relationships
        == 12
    )

    assert (
        result.totals.chunks_without_embeddings
        == 1
    )

    assert result.totals.embedding_coverage == (
        pytest.approx(7 / 8)
    )

    assert len(result.entity_distribution) == 3
    assert (
        result.entity_distribution[0].name
        == "Technology"
    )
    assert result.entity_distribution[0].count == 4

    assert len(
        result.relationship_distribution
    ) == 3

    assert (
        result.relationship_distribution[0].name
        == "MENTIONS"
    )

    assert len(result.recent_documents) == 2

    first_document = result.recent_documents[0]

    assert first_document.document_id == (
        "document-002"
    )
    assert first_document.title == (
        "Architecture Guide"
    )
    assert first_document.chunk_count == 5
    assert first_document.embedding_coverage == 1.0

    second_document = result.recent_documents[1]

    assert second_document.embedding_coverage == (
        pytest.approx(2 / 3)
    )

    assert mock_client.execute_query.call_count == 4


def test_dashboard_queries_use_expected_parameters() -> None:
    """Dashboard queries should pass safe parameters."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [],
        [],
        [],
        [],
    ]

    service.load_dashboard(
        recent_document_limit=7
    )

    totals_call = (
        mock_client.execute_query
        .call_args_list[0]
    )

    totals_query = totals_call.args[0]
    totals_parameters = totals_call.args[1]

    assert "total_nodes" in totals_query
    assert "business_relationships" in totals_query

    assert totals_parameters == {
        "entity_labels": [
            "Person",
            "Project",
            "Customer",
            "Team",
            "Technology",
        ],
        "business_relationship_types": [
            "MANAGES",
            "WORKS_ON",
            "SERVES",
            "USES",
            "MEMBER_OF",
            "OWNS",
        ],
    }

    recent_documents_call = (
        mock_client.execute_query
        .call_args_list[3]
    )

    recent_query = recent_documents_call.args[0]
    recent_parameters = (
        recent_documents_call.args[1]
    )

    assert "MATCH (document:Document)" in (
        recent_query
    )

    assert recent_parameters == {
        "limit": 7,
    }


def test_empty_database_returns_zero_totals() -> None:
    """An empty database should return safe default values."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [],
        [],
        [],
        [],
    ]

    result = service.load_dashboard()

    assert result.totals == DashboardTotals()
    assert result.entity_distribution == []
    assert result.relationship_distribution == []
    assert result.recent_documents == []

    assert result.totals.embedding_coverage == 0.0

    assert (
        result.totals.chunks_without_embeddings
        == 0
    )


def test_embedding_coverage_is_calculated() -> None:
    """Embedding coverage should use embedded and total chunks."""
    totals = DashboardTotals(
        chunks=10,
        embedded_chunks=8,
    )

    assert totals.embedding_coverage == 0.8

    assert totals.chunks_without_embeddings == 2


def test_embedding_coverage_is_zero_without_chunks() -> None:
    """A graph without chunks should have zero coverage."""
    totals = DashboardTotals(
        chunks=0,
        embedded_chunks=0,
    )

    assert totals.embedding_coverage == 0.0

    assert totals.chunks_without_embeddings == 0


def test_chunks_without_embeddings_cannot_be_negative() -> None:
    """Unexpected counts should not produce a negative result."""
    totals = DashboardTotals(
        chunks=2,
        embedded_chunks=3,
    )

    assert totals.chunks_without_embeddings == 0

    assert totals.embedding_coverage == 1.0


def test_recent_document_embedding_coverage() -> None:
    """Each document should expose its embedding coverage."""
    document = RecentDocument(
        document_id="document-001",
        title="Project Overview",
        chunk_count=4,
        embedded_chunk_count=3,
    )

    assert document.embedding_coverage == 0.75


def test_recent_document_without_chunks_has_zero_coverage() -> None:
    """A document without chunks should have zero coverage."""
    document = RecentDocument(
        document_id="document-001",
        title="Empty Document",
        chunk_count=0,
        embedded_chunk_count=0,
    )

    assert document.embedding_coverage == 0.0


def test_invalid_distribution_records_are_skipped() -> None:
    """Distribution rows without names should be ignored."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [],
        [
            {
                "name": None,
                "count": 10,
            },
            {
                "name": "",
                "count": 5,
            },
            {
                "name": "Technology",
                "count": 3,
            },
        ],
        [
            {
                "name": None,
                "count": 7,
            },
            {
                "name": "USES",
                "count": 2,
            },
        ],
        [],
    ]

    result = service.load_dashboard()

    assert result.entity_distribution == [
        DistributionItem(
            name="Technology",
            count=3,
        )
    ]

    assert result.relationship_distribution == [
        DistributionItem(
            name="USES",
            count=2,
        )
    ]


def test_invalid_recent_documents_are_skipped() -> None:
    """Documents without an ID or title should be ignored."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [],
        [],
        [],
        [
            {
                "document_id": None,
                "title": "Missing ID",
                "chunk_count": 2,
                "embedded_chunk_count": 2,
            },
            {
                "document_id": "document-002",
                "title": None,
                "chunk_count": 1,
                "embedded_chunk_count": 1,
            },
            {
                "document_id": "document-003",
                "title": "Valid Document",
                "filename": "valid.txt",
                "document_type": "txt",
                "chunk_count": 3,
                "embedded_chunk_count": 3,
                "updated_at": None,
            },
        ],
    ]

    result = service.load_dashboard()

    assert len(result.recent_documents) == 1

    document = result.recent_documents[0]

    assert document.document_id == "document-003"
    assert document.title == "Valid Document"
    assert document.filename == "valid.txt"
    assert document.document_type == "txt"


def test_missing_numeric_values_default_to_zero() -> None:
    """Missing count values should safely become zero."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [
            {
                "total_nodes": None,
                "total_relationships": None,
                "documents": None,
                "chunks": None,
                "embedded_chunks": None,
                "entities": None,
                "business_relationships": None,
            }
        ],
        [
            {
                "name": "Technology",
                "count": None,
            }
        ],
        [],
        [
            {
                "document_id": "document-001",
                "title": "Project Overview",
                "chunk_count": None,
                "embedded_chunk_count": None,
            }
        ],
    ]

    result = service.load_dashboard()

    assert result.totals == DashboardTotals()

    assert result.entity_distribution[0].count == 0

    assert (
        result.recent_documents[0].chunk_count
        == 0
    )

    assert (
        result.recent_documents[0]
        .embedded_chunk_count
        == 0
    )


def test_negative_numeric_values_are_normalized_to_zero() -> None:
    """Negative Neo4j counts should be normalized to zero."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [
            {
                "total_nodes": -1,
                "total_relationships": -2,
                "documents": -3,
                "chunks": -4,
                "embedded_chunks": -5,
                "entities": -6,
                "business_relationships": -7,
            }
        ],
        [],
        [],
        [],
    ]

    result = service.load_dashboard()

    assert result.totals == DashboardTotals()


def test_invalid_numeric_value_is_rejected() -> None:
    """Non-numeric count values should produce a clear error."""
    service, mock_client = create_service()

    mock_client.execute_query.return_value = [
        {
            "total_nodes": "not-a-number",
        }
    ]

    with pytest.raises(
        DashboardServiceError,
        match="Invalid numeric dashboard value",
    ):
        service._load_totals()


@pytest.mark.parametrize(
    "invalid_limit",
    [
        0,
        -1,
        101,
    ],
)
def test_invalid_recent_document_limits_are_rejected(
    invalid_limit: int,
) -> None:
    """Recent-document limits must remain between 1 and 100."""
    service, mock_client = create_service()

    with pytest.raises(ValueError):
        service.load_dashboard(
            recent_document_limit=invalid_limit
        )

    mock_client.execute_query.assert_not_called()


def test_neo4j_failure_is_wrapped() -> None:
    """Neo4j failures should become dashboard service errors."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    with pytest.raises(
        DashboardServiceError,
        match="Unable to load graph totals",
    ):
        service.load_dashboard()


def test_entity_distribution_failure_stops_loading() -> None:
    """A distribution-query failure should stop the dashboard."""
    service, mock_client = create_service()

    mock_client.execute_query.side_effect = [
        [
            {
                "total_nodes": 1,
                "total_relationships": 0,
                "documents": 0,
                "chunks": 0,
                "embedded_chunks": 0,
                "entities": 1,
                "business_relationships": 0,
            }
        ],
        RuntimeError(
            "Entity query failed"
        ),
    ]

    with pytest.raises(
        DashboardServiceError,
        match="Unable to load entity distribution",
    ):
        service.load_dashboard()

    assert mock_client.execute_query.call_count == 2
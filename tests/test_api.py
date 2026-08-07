"""Tests for the Knowledge Graph AI FastAPI application."""

from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    """Root endpoint should confirm the API is running."""
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "name": "Knowledge Graph AI API",
        "status": "running",
    }


def test_health_endpoint() -> None:
    """Health endpoint should report healthy."""
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }


def test_ask_requires_question() -> None:
    """Ask endpoint should require a question."""
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rejects_empty_question() -> None:
    """Ask endpoint should reject an empty question."""
    response = client.post(
        "/ask",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_limit() -> None:
    """Retrieval limit must remain within allowed values."""
    response = client.post(
        "/ask",
        json={
            "question": "What technologies are used?",
            "limit": 100,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_vector_score() -> None:
    """Vector score must remain between zero and one."""
    response = client.post(
        "/ask",
        json={
            "question": "What technologies are used?",
            "minimum_vector_score": 2.0,
        },
    )

    assert response.status_code == 422
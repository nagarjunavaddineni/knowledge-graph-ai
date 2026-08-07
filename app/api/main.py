"""FastAPI entry point for Knowledge Graph AI."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.services.dashboard_service import (
    DashboardService,
    DashboardServiceError,
)
from app.database.neo4j_client import Neo4jClient
from app.services.graphrag_service import (
    GraphRAGAnswerError,
    GraphRAGAnswerService,
)


app = FastAPI(
    title="Knowledge Graph AI API",
    description=(
        "REST API for the Knowledge Graph AI "
        "and GraphRAG application."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    """Request body for GraphRAG questions."""

    question: str = Field(
        min_length=2,
        max_length=1000,
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=15,
    )

    minimum_vector_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    max_connections_per_entity: int = Field(
        default=10,
        ge=1,
        le=25,
    )


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "name": "Knowledge Graph AI API",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {
        "status": "healthy",
    }
@app.get("/stats")
def get_stats() -> dict:
    """Return knowledge graph statistics."""
    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                raise HTTPException(
                    status_code=503,
                    detail="Neo4j is unavailable.",
                )

            service = DashboardService(client)

            result = service.load_dashboard(
                recent_document_limit=5
            )

            return result.model_dump(
                mode="json"
            )

    except HTTPException:
        raise

    except DashboardServiceError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to load graph statistics.",
        ) from error

@app.post("/ask")
def ask_graph(
    request: AskRequest,
) -> dict:
    """Ask a grounded question against the knowledge graph."""
    try:
        with Neo4jClient.from_settings() as client:
            if not client.verify_connection():
                raise HTTPException(
                    status_code=503,
                    detail="Neo4j is unavailable.",
                )

            service = GraphRAGAnswerService.from_settings(
                client
            )

            result = service.answer_question(
                question=request.question,
                limit=request.limit,
                minimum_vector_score=(
                    request.minimum_vector_score
                ),
                max_connections_per_entity=(
                    request.max_connections_per_entity
                ),
            )

            return result.model_dump(
                mode="json"
            )

    except HTTPException:
        raise

    except GraphRAGAnswerError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate GraphRAG answer.",
        ) from error
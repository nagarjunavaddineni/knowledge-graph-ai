"""Read and update Neo4j Chunk embeddings."""

from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.database.neo4j_client import Neo4jClient


class ChunkEmbeddingError(RuntimeError):
    """Raised when chunk embeddings cannot be stored."""


@dataclass(frozen=True)
class ChunkForEmbedding:
    """A Neo4j Chunk requiring an embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str


class ChunkEmbeddingManager:
    """Manage embedding properties on Neo4j Chunk nodes."""

    def __init__(
        self,
        client: Neo4jClient,
        model: str,
        dimensions: int,
    ) -> None:
        """Initialize the chunk embedding manager."""
        if not model:
            raise ValueError(
                "The embedding model cannot be empty."
            )

        if dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero."
            )

        self.client = client
        self.model = model
        self.dimensions = dimensions

    @classmethod
    def from_settings(
        cls,
        client: Neo4jClient,
    ) -> "ChunkEmbeddingManager":
        """Create the manager from application settings."""
        return cls(
            client=client,
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def get_chunks_requiring_embeddings(
        self,
        force: bool = False,
        limit: int | None = None,
    ) -> list[ChunkForEmbedding]:
        """Return chunks that need new embeddings."""
        if limit is not None and limit <= 0:
            raise ValueError(
                "limit must be greater than zero when supplied."
            )

        effective_limit = limit or 2_147_483_647

        query = """
        MATCH (chunk:Chunk)
        WHERE
            chunk.text IS NOT NULL
            AND trim(chunk.text) <> ""
            AND (
                $force = true
                OR chunk.embedding IS NULL
                OR coalesce(
                    chunk.embedding_model,
                    ""
                ) <> $embedding_model
                OR coalesce(
                    chunk.embedding_dimensions,
                    -1
                ) <> $embedding_dimensions
            )
        RETURN
            chunk.id AS chunk_id,
            chunk.document_id AS document_id,
            chunk.chunk_index AS chunk_index,
            chunk.text AS text
        ORDER BY
            chunk.document_id,
            chunk.chunk_index,
            chunk.id
        LIMIT $limit
        """

        records = self.client.execute_query(
            query,
            {
                "force": force,
                "embedding_model": self.model,
                "embedding_dimensions": self.dimensions,
                "limit": effective_limit,
            },
        )

        chunks: list[ChunkForEmbedding] = []

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("text")

            if not chunk_id or not text:
                continue

            chunks.append(
                ChunkForEmbedding(
                    chunk_id=str(chunk_id),
                    document_id=str(
                        record.get("document_id") or ""
                    ),
                    chunk_index=int(
                        record.get("chunk_index") or 0
                    ),
                    text=str(text),
                )
            )

        return chunks

    def store_embeddings(
        self,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """Store generated vectors on Chunk nodes."""
        if not embeddings:
            return 0

        validated_items = self._validate_embeddings(
            embeddings
        )

        query = """
        UNWIND $items AS item

        MATCH (chunk:Chunk {id: item.chunk_id})

        CALL db.create.setNodeVectorProperty(
            chunk,
            "embedding",
            item.embedding
        )

        SET
            chunk.embedding_model = $embedding_model,
            chunk.embedding_dimensions =
                $embedding_dimensions,
            chunk.embedded_at = datetime(),
            chunk.updated_at = datetime()

        RETURN count(chunk) AS updated_count
        """

        records = self.client.execute_query(
            query,
            {
                "items": validated_items,
                "embedding_model": self.model,
                "embedding_dimensions": self.dimensions,
            },
        )

        if not records:
            raise ChunkEmbeddingError(
                "Neo4j returned no result while storing "
                "chunk embeddings."
            )

        updated_count = int(
            records[0].get("updated_count") or 0
        )

        if updated_count != len(validated_items):
            raise ChunkEmbeddingError(
                f"Neo4j updated {updated_count} chunks, but "
                f"{len(validated_items)} embeddings were supplied."
            )

        return updated_count

    def get_statistics(self) -> dict[str, int]:
        """Return chunk embedding statistics."""
        query = """
        MATCH (chunk:Chunk)
        RETURN
            count(chunk) AS total_chunks,
            count(chunk.embedding) AS embedded_chunks,
            count(chunk)
                - count(chunk.embedding)
                AS chunks_without_embeddings
        """

        records = self.client.execute_query(query)

        if not records:
            return {
                "total_chunks": 0,
                "embedded_chunks": 0,
                "chunks_without_embeddings": 0,
            }

        record = records[0]

        return {
            "total_chunks": int(
                record.get("total_chunks") or 0
            ),
            "embedded_chunks": int(
                record.get("embedded_chunks") or 0
            ),
            "chunks_without_embeddings": int(
                record.get(
                    "chunks_without_embeddings"
                ) or 0
            ),
        }

    def _validate_embeddings(
        self,
        embeddings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate vectors before writing them to Neo4j."""
        validated_items: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for item in embeddings:
            chunk_id = str(
                item.get("chunk_id") or ""
            ).strip()

            vector = item.get("embedding")

            if not chunk_id:
                raise ChunkEmbeddingError(
                    "Every embedding must include a chunk_id."
                )

            if chunk_id in seen_chunk_ids:
                raise ChunkEmbeddingError(
                    f"Duplicate chunk embedding: {chunk_id}"
                )

            if not isinstance(vector, list):
                raise ChunkEmbeddingError(
                    f"Embedding for '{chunk_id}' must be a list."
                )

            if len(vector) != self.dimensions:
                raise ChunkEmbeddingError(
                    f"Embedding for '{chunk_id}' contains "
                    f"{len(vector)} dimensions; expected "
                    f"{self.dimensions}."
                )

            numeric_vector: list[float] = []

            for value in vector:
                if not isinstance(
                    value,
                    (int, float),
                ):
                    raise ChunkEmbeddingError(
                        f"Embedding for '{chunk_id}' contains "
                        "a non-numeric value."
                    )

                numeric_vector.append(float(value))

            validated_items.append(
                {
                    "chunk_id": chunk_id,
                    "embedding": numeric_vector,
                }
            )

            seen_chunk_ids.add(chunk_id)

        return validated_items
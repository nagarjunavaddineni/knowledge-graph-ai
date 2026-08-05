"""Generate and store embeddings for Neo4j Chunk nodes."""

import argparse
import sys
from collections.abc import Iterator, Sequence
from typing import TypeVar

from app.config import settings
from app.database.neo4j_client import Neo4jClient
from app.logging_config import configure_logging
from app.retrieval.chunk_embedding_manager import (
    ChunkEmbeddingError,
    ChunkEmbeddingManager,
    ChunkForEmbedding,
)
from app.retrieval.embedding_service import (
    EmbeddingGenerationError,
    EmbeddingService,
)


ItemType = TypeVar("ItemType")


def batched(
    items: Sequence[ItemType],
    batch_size: int,
) -> Iterator[Sequence[ItemType]]:
    """Yield fixed-size batches."""
    for start_index in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start_index:
            start_index + batch_size
        ]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenAI embeddings for Neo4j Chunk nodes."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Generate and save embeddings. Without this option, "
            "the command performs a dry run."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate embeddings even when compatible "
            "embeddings already exist."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks sent in each OpenAI request.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to process.",
    )

    return parser.parse_args()


def display_candidates(
    chunks: list[ChunkForEmbedding],
) -> None:
    """Display chunks selected for embedding."""
    print("\nChunks selected for embedding")
    print("-" * 72)

    preview_count = min(len(chunks), 5)

    for chunk in chunks[:preview_count]:
        preview = " ".join(
            chunk.text[:100].split()
        )

        print(
            f"{chunk.chunk_id} | "
            f"document={chunk.document_id} | "
            f"index={chunk.chunk_index}"
        )
        print(f"  {preview}")

    if len(chunks) > preview_count:
        print(
            f"... and {len(chunks) - preview_count} "
            "additional chunks."
        )


def main() -> int:
    """Generate and store missing chunk embeddings."""
    arguments = parse_arguments()
    configure_logging(settings.log_level)

    if arguments.batch_size <= 0:
        print(
            "Configuration error: --batch-size must be "
            "greater than zero."
        )
        return 1

    if (
        arguments.limit is not None
        and arguments.limit <= 0
    ):
        print(
            "Configuration error: --limit must be "
            "greater than zero."
        )
        return 1

    try:
        with Neo4jClient.from_settings() as client:
            print("Verifying Neo4j connection...")

            if not client.verify_connection():
                print(
                    "Neo4j connection verification failed."
                )
                return 1

            print("Neo4j connection verified.")

            manager = ChunkEmbeddingManager.from_settings(
                client
            )

            initial_statistics = manager.get_statistics()

            print("\nCurrent embedding statistics")
            print("-" * 72)
            print(
                "Total chunks             : "
                f"{initial_statistics['total_chunks']}"
            )
            print(
                "Embedded chunks          : "
                f"{initial_statistics['embedded_chunks']}"
            )
            print(
                "Chunks without embeddings: "
                f"{initial_statistics['chunks_without_embeddings']}"
            )

            chunks = (
                manager.get_chunks_requiring_embeddings(
                    force=arguments.force,
                    limit=arguments.limit,
                )
            )

            print(
                "\nChunks requiring generation: "
                f"{len(chunks)}"
            )
            print(
                "Embedding model            : "
                f"{settings.openai_embedding_model}"
            )
            print(
                "Embedding dimensions       : "
                f"{settings.embedding_dimensions}"
            )
            print(
                "Batch size                 : "
                f"{arguments.batch_size}"
            )

            if not chunks:
                print(
                    "\nNo chunks require embeddings."
                )
                return 0

            display_candidates(chunks)

            if not arguments.execute:
                request_count = (
                    len(chunks)
                    + arguments.batch_size
                    - 1
                ) // arguments.batch_size

                print("\nDry run completed successfully.")
                print("No OpenAI request was made.")
                print("No Neo4j data was updated.")
                print(
                    "Estimated embedding requests: "
                    f"{request_count}"
                )
                print(
                    "\nRun again with --execute to generate "
                    "and store the vectors."
                )

                return 0

            embedding_service = (
                EmbeddingService.from_settings()
            )

            total_updated = 0
            chunk_batches = list(
                batched(
                    chunks,
                    arguments.batch_size,
                )
            )

            print("\nGenerating embeddings...")

            for batch_number, chunk_batch in enumerate(
                chunk_batches,
                start=1,
            ):
                texts = [
                    chunk.text
                    for chunk in chunk_batch
                ]

                vectors = embedding_service.embed_texts(
                    texts
                )

                embedding_items = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "embedding": vector,
                    }
                    for chunk, vector in zip(
                        chunk_batch,
                        vectors,
                        strict=True,
                    )
                ]

                updated_count = manager.store_embeddings(
                    embedding_items
                )

                total_updated += updated_count

                print(
                    f"Batch {batch_number}/"
                    f"{len(chunk_batches)}: "
                    f"{updated_count} chunks updated"
                )

            final_statistics = manager.get_statistics()

            print("\nEmbedding process completed")
            print("-" * 72)
            print(
                f"Chunks updated            : "
                f"{total_updated}"
            )
            print(
                "Total chunks              : "
                f"{final_statistics['total_chunks']}"
            )
            print(
                "Embedded chunks           : "
                f"{final_statistics['embedded_chunks']}"
            )
            print(
                "Chunks without embeddings : "
                f"{final_statistics['chunks_without_embeddings']}"
            )

            return 0

    except EmbeddingGenerationError as error:
        print(f"Embedding generation failed: {error}")
        return 1

    except ChunkEmbeddingError as error:
        print(f"Embedding storage failed: {error}")
        return 1

    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    except Exception as error:
        print(
            "Unexpected embedding failure: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""Generate text embeddings using the OpenAI API."""

import logging
from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from app.config import settings


logger = logging.getLogger(__name__)


class EmbeddingGenerationError(RuntimeError):
    """Raised when text embeddings cannot be generated."""


class EmbeddingService:
    """Generate validated embeddings for text collections."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the OpenAI embedding service."""
        if not api_key:
            raise ValueError(
                "An OpenAI API key is required."
            )

        if not model:
            raise ValueError(
                "An embedding model is required."
            )

        if dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        self.model = model
        self.dimensions = dimensions

        self.client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @classmethod
    def from_settings(cls) -> "EmbeddingService":
        """Create the service from application settings."""
        return cls(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate one embedding."""
        embeddings = self.embed_texts([text])

        return embeddings[0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for a collection of texts."""
        normalized_texts = self._validate_texts(texts)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=normalized_texts,
                dimensions=self.dimensions,
                encoding_format="float",
            )

            if not response.data:
                raise EmbeddingGenerationError(
                    "OpenAI returned no embedding data."
                )

            ordered_embeddings = sorted(
                response.data,
                key=lambda item: item.index,
            )

            if len(ordered_embeddings) != len(
                normalized_texts
            ):
                raise EmbeddingGenerationError(
                    "The number of returned embeddings does not "
                    "match the number of supplied texts."
                )

            vectors: list[list[float]] = []

            for item in ordered_embeddings:
                vector = [
                    float(value)
                    for value in item.embedding
                ]

                if len(vector) != self.dimensions:
                    raise EmbeddingGenerationError(
                        "OpenAI returned an embedding with "
                        f"{len(vector)} dimensions; expected "
                        f"{self.dimensions}."
                    )

                vectors.append(vector)

            logger.info(
                "Generated %s embeddings using model '%s'.",
                len(vectors),
                self.model,
            )

            return vectors

        except RateLimitError as error:
            logger.exception(
                "OpenAI embedding rate limit reached."
            )

            raise EmbeddingGenerationError(
                "OpenAI rate limit reached while generating "
                "embeddings."
            ) from error

        except APIConnectionError as error:
            logger.exception(
                "Unable to connect to OpenAI for embeddings."
            )

            raise EmbeddingGenerationError(
                "Unable to connect to OpenAI while generating "
                "embeddings."
            ) from error

        except APIStatusError as error:
            logger.exception(
                "OpenAI embedding request returned status %s.",
                error.status_code,
            )

            raise EmbeddingGenerationError(
                "OpenAI embedding request failed with status "
                f"{error.status_code}."
            ) from error

    @staticmethod
    def _validate_texts(
        texts: Sequence[str],
    ) -> list[str]:
        """Validate and normalize embedding inputs."""
        if not texts:
            raise ValueError(
                "At least one text value is required."
            )

        normalized_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Embedding input {index} must be a string."
                )

            normalized_text = text.strip()

            if not normalized_text:
                raise ValueError(
                    f"Embedding input {index} cannot be empty."
                )

            normalized_texts.append(normalized_text)

        return normalized_texts
"""Split loaded documents into overlapping text chunks."""

import hashlib
import re

from app.ingestion.models import LoadedDocument, TextChunk


class TextSplitter:
    """Split document text into manageable overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        minimum_chunk_size: int = 80,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if minimum_chunk_size <= 0:
            raise ValueError(
                "minimum_chunk_size must be greater than zero."
            )

        if minimum_chunk_size > chunk_size:
            raise ValueError(
                "minimum_chunk_size cannot exceed chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_size = minimum_chunk_size

    def split_document(
        self,
        document: LoadedDocument,
    ) -> list[TextChunk]:
        """Split a loaded document into validated text chunks."""
        content = document.content

        if len(content) <= self.chunk_size:
            return [
                self._create_chunk(
                    document_id=document.document_id,
                    chunk_index=0,
                    text=content,
                    start_char=0,
                    end_char=len(content),
                )
            ]

        chunks: list[TextChunk] = []
        start_char = 0
        chunk_index = 0

        while start_char < len(content):
            proposed_end = min(
                start_char + self.chunk_size,
                len(content),
            )

            end_char = self._find_natural_boundary(
                content=content,
                start_char=start_char,
                proposed_end=proposed_end,
            )

            if end_char <= start_char:
                end_char = proposed_end

            chunk_text = content[start_char:end_char].strip()

            if chunk_text:
                if (
                    len(chunk_text) >= self.minimum_chunk_size
                    or not chunks
                    or end_char == len(content)
                ):
                    chunks.append(
                        self._create_chunk(
                            document_id=document.document_id,
                            chunk_index=chunk_index,
                            text=chunk_text,
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )
                    chunk_index += 1

            if end_char >= len(content):
                break

            next_start = end_char - self.chunk_overlap

            if next_start <= start_char:
                next_start = end_char

            start_char = self._move_to_word_boundary(
                content=content,
                position=next_start,
            )

        return chunks

    def split_documents(
        self,
        documents: list[LoadedDocument],
    ) -> list[TextChunk]:
        """Split multiple documents into a single chunk list."""
        chunks: list[TextChunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def _find_natural_boundary(
        self,
        content: str,
        start_char: int,
        proposed_end: int,
    ) -> int:
        """Find a nearby paragraph, sentence, or word boundary."""
        if proposed_end >= len(content):
            return len(content)

        search_start = max(
            start_char + self.minimum_chunk_size,
            proposed_end - 250,
        )

        search_section = content[search_start:proposed_end]

        boundary_patterns = (
            r"\n\n",
            r"(?<=[.!?])\s+",
            r"\n",
            r"\s+",
        )

        for pattern in boundary_patterns:
            matches = list(
                re.finditer(
                    pattern,
                    search_section,
                )
            )

            if matches:
                last_match = matches[-1]

                return (
                    search_start
                    + last_match.end()
                )

        return proposed_end

    @staticmethod
    def _move_to_word_boundary(
        content: str,
        position: int,
    ) -> int:
        """Move a starting offset to the next complete word."""
        if position <= 0:
            return 0

        if position >= len(content):
            return len(content)

        if content[position - 1].isspace():
            return position

        next_space = content.find(" ", position)

        if next_space == -1:
            return position

        return next_space + 1

    @staticmethod
    def _create_chunk(
        document_id: str,
        chunk_index: int,
        text: str,
        start_char: int,
        end_char: int,
    ) -> TextChunk:
        """Create a stable validated chunk."""
        chunk_hash = hashlib.sha256(
            (
                f"{document_id}:"
                f"{chunk_index}:"
                f"{start_char}:"
                f"{end_char}:"
                f"{text}"
            ).encode("utf-8")
        ).hexdigest()

        return TextChunk(
            chunk_id=f"chunk-{chunk_hash[:24]}",
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            start_char=start_char,
            end_char=end_char,
        )
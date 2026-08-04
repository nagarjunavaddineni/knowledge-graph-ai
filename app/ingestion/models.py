"""Validated models used by the document-ingestion pipeline."""

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DocumentType(str, Enum):
    """Supported document formats."""

    TXT = "txt"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"


class EntityType(str, Enum):
    """Node types supported by the knowledge graph."""

    PERSON = "Person"
    PROJECT = "Project"
    CUSTOMER = "Customer"
    TEAM = "Team"
    TECHNOLOGY = "Technology"
    DOCUMENT = "Document"


class RelationshipType(str, Enum):
    """Relationship types supported by the knowledge graph."""

    MANAGES = "MANAGES"
    WORKS_ON = "WORKS_ON"
    SERVES = "SERVES"
    USES = "USES"
    MEMBER_OF = "MEMBER_OF"
    OWNS = "OWNS"
    MENTIONS = "MENTIONS"


class SourceProperty(StrictModel):
    """A key-value property extracted from a source document."""

    key: str = Field(
        min_length=1,
        max_length=100,
        description="Property name.",
    )
    value: str = Field(
        min_length=1,
        max_length=1000,
        description="Property value.",
    )


class LoadedDocument(StrictModel):
    """A document after text has been loaded from disk."""

    document_id: str = Field(
        min_length=1,
        description="Stable identifier for the document.",
    )
    filename: str = Field(
        min_length=1,
        description="Original document filename.",
    )
    file_type: DocumentType
    source_path: str = Field(
        min_length=1,
        description="Original local path of the document.",
    )
    content: str = Field(
        min_length=1,
        description="Text extracted from the document.",
    )
    metadata: list[SourceProperty] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject documents that contain no meaningful text."""
        if not value.strip():
            raise ValueError(
                "Document content cannot be empty or whitespace only."
            )

        return value


class TextChunk(StrictModel):
    """A smaller section of a loaded document."""

    chunk_id: str = Field(
        min_length=1,
        description="Stable identifier for this chunk.",
    )
    document_id: str = Field(
        min_length=1,
        description="Identifier of the parent document.",
    )
    chunk_index: int = Field(
        ge=0,
        description="Zero-based position of the chunk.",
    )
    text: str = Field(
        min_length=1,
        description="Text contained in the chunk.",
    )
    start_char: int = Field(
        ge=0,
        description="Starting character offset.",
    )
    end_char: int = Field(
        ge=1,
        description="Ending character offset.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject empty chunks."""
        if not value.strip():
            raise ValueError(
                "Chunk text cannot be empty or whitespace only."
            )

        return value

    @model_validator(mode="after")
    def validate_character_offsets(self) -> "TextChunk":
        """Ensure the ending offset follows the starting offset."""
        if self.end_char <= self.start_char:
            raise ValueError(
                "end_char must be greater than start_char."
            )

        return self


class GraphEntity(StrictModel):
    """An entity extracted from a document chunk."""

    name: str = Field(
        min_length=1,
        max_length=250,
        description="Canonical entity name.",
    )
    entity_type: EntityType
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Short description supported by the source.",
    )
    properties: list[SourceProperty] = Field(default_factory=list)
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence between zero and one.",
    )


class GraphRelationship(StrictModel):
    """A relationship extracted between two entities."""

    source_name: str = Field(
        min_length=1,
        max_length=250,
    )
    source_type: EntityType
    relationship_type: RelationshipType
    target_name: str = Field(
        min_length=1,
        max_length=250,
    )
    target_type: EntityType
    evidence: str | None = Field(
        default=None,
        max_length=1500,
        description="Text supporting the relationship.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def prevent_self_relationship(self) -> "GraphRelationship":
        """Reject relationships from an entity to itself."""
        same_name = (
            self.source_name.casefold()
            == self.target_name.casefold()
        )
        same_type = self.source_type == self.target_type

        if same_name and same_type:
            raise ValueError(
                "A relationship cannot connect an entity to itself."
            )

        return self


class ExtractionResult(StrictModel):
    """Structured entities and relationships returned by the AI model."""

    chunk_id: str = Field(
        min_length=1,
        description="Chunk used to create this extraction.",
    )
    entities: list[GraphEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(
        default_factory=list
    )
    summary: str | None = Field(
        default=None,
        max_length=1500,
    )
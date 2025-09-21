"""Core data models for the simplified PathRAG retriever."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EntityMatch:
    """Represents a node match returned from the entity vector index."""

    name: str
    type: Optional[str]
    description: str
    score: float


@dataclass(frozen=True)
class RelationMatch:
    """Represents an edge match returned from the relation vector index."""

    source_name: str
    target_name: str
    description: str
    keywords: str
    score: float


@dataclass(frozen=True)
class ChunkMatch:
    """Represents a chunk retrieved from the chunk vector index."""

    chunk_uuid: str
    document_id: str
    filename: str
    text: str
    score: float


@dataclass(frozen=True)
class ContextWindow:
    """A context window summarising graph and chunk evidence."""

    label: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Structured output returned by :class:`PathRAG`."""

    answer: str
    context_windows: List[ContextWindow]
    entity_matches: List[EntityMatch]
    relation_matches: List[RelationMatch]
    chunk_matches: List[ChunkMatch]

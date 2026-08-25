from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SparseVectorData:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    content: str
    dense_embedding: list[float]
    sparse_embedding: SparseVectorData
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    content: str
    source_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    sources: list[str] = field(default_factory=list)
    cached: bool = False

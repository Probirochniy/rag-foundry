from collections.abc import Sequence
from typing import Protocol

from src.core.entities.rag import DocumentChunk, SearchResult


class VectorStoreProtocol(Protocol):
    async def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None: ...

    async def is_healthy(self) -> bool: ...

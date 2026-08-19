from collections.abc import Sequence
from typing import Protocol

from src.core.entities.rag import DocumentChunk, SearchResult
from src.core.protocols.system.health import HealthProtocol


class VectorStoreProtocol(HealthProtocol, Protocol):
    async def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None: ...

    async def ensure_collection_exists(self) -> None: ...

    async def close(self) -> None: ...

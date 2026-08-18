from collections.abc import Sequence

from src.core.entities.rag import DocumentChunk, SearchResult
from src.core.protocols.vector_store import VectorStoreProtocol


class VectorStoreMock(VectorStoreProtocol):
    def __init__(self, mock_results: list[SearchResult] | None = None) -> None:
        self.mock_results = mock_results or []
        self.search_called = False
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    async def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        self.search_called = True
        self.last_query = query
        self.last_top_k = top_k
        return self.mock_results

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        pass

    async def is_healthy(self) -> bool:
        return True

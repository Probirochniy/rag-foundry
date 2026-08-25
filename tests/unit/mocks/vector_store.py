from collections.abc import Sequence

from src.core.entities.rag import DocumentChunk, SearchResult, SparseVectorData
from src.core.protocols.vector_store import VectorStoreProtocol


class VectorStoreMock(VectorStoreProtocol):
    def __init__(self, mock_results: list[SearchResult] | None = None) -> None:
        self.mock_results = mock_results or []
        self.search_called = False
        self.upsert_called = False
        self.last_upserted_chunks: list[DocumentChunk] = []
        self.last_dense_vector: list[float] | None = None
        self.last_sparse_vector: SparseVectorData | None = None
        self.last_top_k: int | None = None

    async def search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 3,
    ) -> list[SearchResult]:
        self.search_called = True
        self.last_dense_vector = dense_vector
        self.last_sparse_vector = sparse_vector
        self.last_top_k = top_k
        return self.mock_results

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        self.upsert_called = True
        self.last_upserted_chunks = list(chunks)

    async def is_healthy(self) -> bool:
        return True

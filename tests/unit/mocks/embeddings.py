from collections.abc import Sequence

from src.core.entities.rag import SparseVectorData
from src.core.protocols.embeddings import DenseEmbeddingsProtocol, SparseEmbeddingsProtocol


class DenseEmbeddingsMock(DenseEmbeddingsProtocol):
    def __init__(self, default_vector: list[float] | None = None) -> None:
        self.default_vector = default_vector or [0.1, 0.2, 0.3]
        self.embed_query_called = False
        self.embed_documents_called = False
        self.last_query: str | None = None
        self.last_documents: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.embed_query_called = True
        self.last_query = text
        return self.default_vector

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_documents_called = True
        self.last_documents = list(texts)
        return [self.default_vector for _ in texts]

    async def is_healthy(self) -> bool:
        return True


class SparseEmbeddingsMock(SparseEmbeddingsProtocol):
    def __init__(self, default_vector: SparseVectorData | None = None) -> None:
        self.default_vector = default_vector or SparseVectorData(indices=[1, 2], values=[0.5, 0.9])
        self.embed_query_called = False
        self.embed_documents_called = False
        self.last_query: str | None = None
        self.last_documents: list[str] = []

    async def embed_query(self, text: str) -> SparseVectorData:
        self.embed_query_called = True
        self.last_query = text
        return self.default_vector

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        self.embed_documents_called = True
        self.last_documents = list(texts)
        return [self.default_vector for _ in texts]

    async def is_healthy(self) -> bool:
        return True

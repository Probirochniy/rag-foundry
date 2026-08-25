from collections.abc import Sequence
from typing import Protocol

from src.core.entities.rag import SparseVectorData


class DenseEmbeddingsProtocol(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def is_healthy(self) -> bool: ...


class SparseEmbeddingsProtocol(Protocol):
    async def embed_query(self, text: str) -> SparseVectorData: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]: ...

    async def is_healthy(self) -> bool: ...

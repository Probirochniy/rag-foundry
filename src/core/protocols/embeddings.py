from collections.abc import Sequence
from typing import Protocol


class EmbeddingsProtocol(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

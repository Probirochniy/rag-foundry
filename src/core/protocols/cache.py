from typing import Protocol

from src.core.entities.rag import GeneratedAnswer


class CacheStoreProtocol(Protocol):
    async def get(self, query: str) -> GeneratedAnswer | None: ...

    async def set(self, query: str, answer: GeneratedAnswer, ttl_seconds: int = 3600) -> None: ...

    async def is_healthy(self) -> bool: ...

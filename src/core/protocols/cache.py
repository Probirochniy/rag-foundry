from typing import Protocol

from src.core.entities.rag import GeneratedAnswer
from src.core.protocols.system.health import HealthProtocol


class CacheStoreProtocol(HealthProtocol, Protocol):
    async def get(self, query: str) -> GeneratedAnswer | None: ...

    async def set(self, query: str, answer: GeneratedAnswer, ttl_seconds: int = 3600) -> None: ...

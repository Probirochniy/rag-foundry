from typing import Protocol

from src.core.entities.rag import GeneratedAnswer
from src.core.protocols.system.health import HealthProtocol


class CacheStoreProtocol(HealthProtocol, Protocol):
    async def get(self, query: str, top_k: int = 3) -> GeneratedAnswer | None: ...

    async def set(
        self,
        query: str,
        answer: GeneratedAnswer,
        top_k: int = 3,
        ttl_seconds: int = 3600,
    ) -> None: ...

    async def close(self) -> None: ...

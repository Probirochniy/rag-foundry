from src.core.entities.rag import GeneratedAnswer
from src.core.protocols.cache import CacheStoreProtocol


class CacheStoreMock(CacheStoreProtocol):
    def __init__(self, initial_data: dict[str, GeneratedAnswer] | None = None) -> None:
        self.storage: dict[str, GeneratedAnswer] = initial_data or {}
        self.set_called = False

    def _key(self, query: str, top_k: int) -> str:
        return f"{query}:{top_k}"

    async def get(self, query: str, top_k: int = 3) -> GeneratedAnswer | None:
        return self.storage.get(self._key(query, top_k)) or self.storage.get(query)

    async def set(
        self,
        query: str,
        answer: GeneratedAnswer,
        top_k: int = 3,
        ttl_seconds: int = 3600,
    ) -> None:
        self.set_called = True
        self.storage[self._key(query, top_k)] = answer

    async def is_healthy(self) -> bool:
        return True

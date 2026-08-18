from src.core.entities.rag import GeneratedAnswer
from src.core.protocols.cache import CacheStoreProtocol


class CacheStoreMock(CacheStoreProtocol):
    def __init__(self, initial_data: dict[str, GeneratedAnswer] | None = None) -> None:
        self.storage: dict[str, GeneratedAnswer] = initial_data or {}
        self.set_called = False

    async def get(self, query: str) -> GeneratedAnswer | None:
        return self.storage.get(query)

    async def set(self, query: str, answer: GeneratedAnswer, ttl_seconds: int = 3600) -> None:
        self.set_called = True
        self.storage[query] = answer

    async def is_healthy(self) -> bool:
        return True

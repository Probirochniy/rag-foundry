from collections.abc import AsyncIterator
from typing import Protocol

from src.core.entities.rag import GeneratedAnswer, SearchResult


class LLMClientProtocol(Protocol):
    async def generate_answer(self, query: str, context: list[SearchResult]) -> GeneratedAnswer: ...

    async def generate_stream(
        self, query: str, context: list[SearchResult]
    ) -> AsyncIterator[str]: ...

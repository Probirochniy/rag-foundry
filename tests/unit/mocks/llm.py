from collections.abc import AsyncIterator

from src.core.entities.rag import GeneratedAnswer, SearchResult
from src.core.protocols.llm import LLMClientProtocol


class LLMClientMock(LLMClientProtocol):
    def __init__(self, default_answer: str = "Mock answer") -> None:
        self.default_answer = default_answer
        self.generate_called = False

    async def generate_answer(self, query: str, context: list[SearchResult]) -> GeneratedAnswer:
        self.generate_called = True
        sources = [c.source_id for c in context]
        return GeneratedAnswer(
            answer=f"{self.default_answer}: {query}",
            sources=sources,
            cached=False,
        )

    async def generate_stream(self, query: str, context: list[SearchResult]) -> AsyncIterator[str]:
        for token in ["chunk1 ", "chunk2 ", "chunk3"]:
            yield token

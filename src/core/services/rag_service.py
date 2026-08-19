from collections.abc import AsyncIterator, Sequence

from src.core.entities.rag import DocumentChunk, GeneratedAnswer
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.llm import LLMClientProtocol
from src.core.protocols.vector_store import VectorStoreProtocol


class RAGService:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        cache_store: CacheStoreProtocol,
        llm_client: LLMClientProtocol,
    ) -> None:
        self._vector_store = vector_store
        self._cache_store = cache_store
        self._llm_client = llm_client

    async def ingest_documents(self, chunks: Sequence[DocumentChunk]) -> None:
        await self._vector_store.upsert(chunks=chunks)

    async def ask(self, query: str, top_k: int = 3) -> GeneratedAnswer:
        cached_result = await self._cache_store.get(query)
        if cached_result:
            return cached_result

        search_results = await self._vector_store.search(query=query, top_k=top_k)

        generated = await self._llm_client.generate_answer(query=query, context=search_results)

        await self._cache_store.set(query=query, answer=generated, ttl_seconds=3600)

        return generated

    async def ask_stream(self, query: str, top_k: int = 3) -> AsyncIterator[str]:
        search_results = await self._vector_store.search(query=query, top_k=top_k)
        async for chunk in self._llm_client.generate_stream(query=query, context=search_results):
            yield chunk

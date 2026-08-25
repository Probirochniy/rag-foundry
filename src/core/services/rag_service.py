import asyncio
from collections.abc import AsyncIterator

from src.core.config import settings
from src.core.entities.rag import DocumentChunk, GeneratedAnswer, SparseVectorData
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.embeddings import DenseEmbeddingsProtocol, SparseEmbeddingsProtocol
from src.core.protocols.llm import LLMClientProtocol
from src.core.protocols.splitter import TextSplitterProtocol
from src.core.protocols.vector_store import VectorStoreProtocol


class RAGService:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        cache_store: CacheStoreProtocol,
        llm_client: LLMClientProtocol,
        text_splitter: TextSplitterProtocol,
        dense_embeddings: DenseEmbeddingsProtocol,
        sparse_embeddings: SparseEmbeddingsProtocol,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._vector_store = vector_store
        self._cache_store = cache_store
        self._llm_client = llm_client
        self._text_splitter = text_splitter
        self._dense_embeddings = dense_embeddings
        self._sparse_embeddings = sparse_embeddings
        self._cache_ttl_seconds = cache_ttl_seconds

    async def _embed_query(self, query: str) -> tuple[list[float], SparseVectorData]:
        dense_vec, sparse_raw = await asyncio.gather(
            self._dense_embeddings.embed_query(query),
            self._sparse_embeddings.embed_query(query),
        )
        sparse_vec = SparseVectorData(
            indices=sparse_raw.indices,
            values=sparse_raw.values,
        )
        return dense_vec, sparse_vec

    async def ingest_document(self, source_id: str, content: str) -> int:
        raw_chunks = self._text_splitter.split(content)
        if not raw_chunks:
            return 0

        dense_vectors, sparse_vectors = await asyncio.gather(
            self._dense_embeddings.embed_documents(raw_chunks),
            self._sparse_embeddings.embed_documents(raw_chunks),
        )

        chunks: list[DocumentChunk] = [
            DocumentChunk(
                id=f"{source_id}#{idx}",
                content=chunk_text,
                metadata={"source_id": source_id, "chunk_index": idx},
                dense_embedding=dense_vec,
                sparse_embedding=SparseVectorData(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
            )
            for idx, (chunk_text, dense_vec, sparse_vec) in enumerate(
                zip(raw_chunks, dense_vectors, sparse_vectors, strict=True)
            )
        ]

        await self._vector_store.upsert(chunks=chunks)
        return len(chunks)

    async def ask(self, query: str, top_k: int = 3) -> GeneratedAnswer:
        cached_result = await self._cache_store.get(query=query, top_k=top_k)
        if cached_result:
            return cached_result

        dense_vec, sparse_vec = await self._embed_query(query)
        search_results = await self._vector_store.search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
        )

        generated = await self._llm_client.generate_answer(query=query, context=search_results)

        await self._cache_store.set(
            query=query,
            answer=generated,
            top_k=top_k,
            ttl_seconds=self._cache_ttl_seconds,
        )

        return generated

    async def ask_stream(self, query: str, top_k: int = 3) -> AsyncIterator[str]:
        cached_result = await self._cache_store.get(query=query, top_k=top_k)
        if cached_result:
            yield cached_result.answer
            return

        dense_vec, sparse_vec = await self._embed_query(query)
        search_results = await self._vector_store.search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
        )
        sources = list({c.source_id for c in search_results})

        collected_chunks: list[str] = []

        async for chunk in self._llm_client.generate_stream(query=query, context=search_results):
            if chunk == settings.hallucination_marker:
                collected_chunks.clear()
                yield chunk
                continue

            collected_chunks.append(chunk)
            yield chunk

        if collected_chunks:
            full_answer = "".join(collected_chunks).strip()
            if full_answer:
                answer_entity = GeneratedAnswer(
                    answer=full_answer,
                    sources=sources,
                    cached=False,
                )
                await self._cache_store.set(
                    query=query,
                    answer=answer_entity,
                    top_k=top_k,
                    ttl_seconds=self._cache_ttl_seconds,
                )

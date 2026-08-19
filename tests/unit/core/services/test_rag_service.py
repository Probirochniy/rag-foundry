import pytest

from src.core.entities.rag import DocumentChunk, GeneratedAnswer, SearchResult
from src.core.services.rag_service import RAGService
from tests.unit.mocks.cache import CacheStoreMock
from tests.unit.mocks.llm import LLMClientMock
from tests.unit.mocks.vector_store import VectorStoreMock


@pytest.mark.asyncio
async def test_rag_service_cache_hit() -> None:
    test_question = "как поднять кубер"

    cached_answer = GeneratedAnswer(answer="сложно...", sources=["doc_cached.pdf"], cached=True)
    cache = CacheStoreMock(initial_data={test_question: cached_answer})
    vector_store = VectorStoreMock()
    llm = LLMClientMock()

    service = RAGService(vector_store=vector_store, cache_store=cache, llm_client=llm)

    result = await service.ask(query=test_question, top_k=3)

    assert result == cached_answer
    assert vector_store.search_called is False
    assert llm.generate_called is False


@pytest.mark.asyncio
async def test_rag_service_cache_miss() -> None:
    test_question = "how to run kubernetes"
    test_file = "k8s.pdf"

    cache = CacheStoreMock()
    search_mock = [SearchResult(content="Инструкция по куберу", source_id=test_file, score=0.95)]
    vector_store = VectorStoreMock(mock_results=search_mock)
    llm = LLMClientMock(default_answer="LLM response")

    service = RAGService(vector_store=vector_store, cache_store=cache, llm_client=llm)

    result = await service.ask(query=test_question, top_k=5)

    assert vector_store.search_called is True
    assert vector_store.last_query == test_question
    assert vector_store.last_top_k == 5
    assert llm.generate_called is True
    assert cache.set_called is True

    assert f"LLM response: {test_question}" in result.answer
    assert result.sources == ["k8s.pdf"]

    cached_val = await cache.get(test_question)
    assert cached_val is not None
    assert cached_val.answer == result.answer


@pytest.mark.asyncio
async def test_rag_service_streaming() -> None:
    cache = CacheStoreMock()
    vector_store = VectorStoreMock()
    llm = LLMClientMock()

    service = RAGService(vector_store=vector_store, cache_store=cache, llm_client=llm)

    chunks: list[str] = []
    async for chunk in service.ask_stream(query="стрим запрос", top_k=3):
        chunks.append(chunk)

    assert chunks == ["chunk1 ", "chunk2 ", "chunk3"]


@pytest.mark.asyncio
async def test_rag_service_ingest_documents() -> None:
    cache = CacheStoreMock()
    vector_store = VectorStoreMock()
    llm = LLMClientMock()

    service = RAGService(vector_store=vector_store, cache_store=cache, llm_client=llm)

    chunks_to_ingest = [
        DocumentChunk(id="1", content="Content 1", metadata={"source_id": "doc1.pdf"}),
        DocumentChunk(id="2", content="Content 2", metadata={"source_id": "doc2.pdf"}),
    ]

    await service.ingest_documents(chunks=chunks_to_ingest)

    assert vector_store.upsert_called is True
    assert vector_store.last_upserted_chunks == chunks_to_ingest

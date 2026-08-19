import pytest

from src.core.entities.rag import DocumentChunk, GeneratedAnswer, SearchResult
from src.core.services.rag_service import RAGService
from tests.unit.mocks.cache import CacheStoreMock
from tests.unit.mocks.llm import LLMClientMock
from tests.unit.mocks.splitter import TextSplitterMock
from tests.unit.mocks.vector_store import VectorStoreMock


@pytest.mark.asyncio
async def test_rag_service_cache_hit() -> None:
    test_question = "как поднять кубер"

    cached_answer = GeneratedAnswer(answer="сложно...", sources=["doc_cached.pdf"], cached=True)
    cache = CacheStoreMock(initial_data={test_question: cached_answer})
    vector_store = VectorStoreMock()
    llm = LLMClientMock()
    text_splitter = TextSplitterMock()

    service = RAGService(
        vector_store=vector_store, cache_store=cache, llm_client=llm, text_splitter=text_splitter
    )

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
    text_splitter = TextSplitterMock()

    service = RAGService(
        vector_store=vector_store, cache_store=cache, llm_client=llm, text_splitter=text_splitter
    )

    result = await service.ask(query=test_question, top_k=5)

    assert vector_store.search_called is True
    assert vector_store.last_query == test_question
    assert vector_store.last_top_k == 5
    assert llm.generate_called is True
    assert cache.set_called is True

    assert f"LLM response: {test_question}" in result.answer
    assert result.sources == ["k8s.pdf"]

    cached_val = await cache.get(query=test_question, top_k=5)
    assert cached_val is not None
    assert cached_val.answer == result.answer


@pytest.mark.asyncio
async def test_rag_service_streaming_cache_hit() -> None:
    test_question = "stream query"
    cached_answer = GeneratedAnswer(answer="streamed answer", sources=["doc1.pdf"], cached=True)

    cache = CacheStoreMock(initial_data={test_question: cached_answer})
    vector_store = VectorStoreMock()
    llm = LLMClientMock()
    text_splitter = TextSplitterMock()

    service = RAGService(
        vector_store=vector_store, cache_store=cache, llm_client=llm, text_splitter=text_splitter
    )

    chunks: list[str] = []
    async for chunk in service.ask_stream(query=test_question, top_k=3):
        chunks.append(chunk)

    assert chunks == ["streamed answer"]
    assert vector_store.search_called is False
    assert llm.generate_called is False


@pytest.mark.asyncio
async def test_rag_service_streaming_cache_miss() -> None:
    cache = CacheStoreMock()
    vector_store = VectorStoreMock()
    llm = LLMClientMock()
    text_splitter = TextSplitterMock()

    service = RAGService(
        vector_store=vector_store, cache_store=cache, llm_client=llm, text_splitter=text_splitter
    )

    chunks: list[str] = []
    async for chunk in service.ask_stream(query="stream query", top_k=3):
        chunks.append(chunk)

    assert chunks == ["chunk1 ", "chunk2 ", "chunk3"]

    assert vector_store.search_called is True
    cached_val = await cache.get(query="stream query", top_k=3)
    assert cached_val is not None


@pytest.mark.asyncio
async def test_rag_service_ingest_document() -> None:
    cache = CacheStoreMock()
    vector_store = VectorStoreMock()
    llm = LLMClientMock()
    text_splitter = TextSplitterMock()

    service = RAGService(
        vector_store=vector_store, cache_store=cache, llm_client=llm, text_splitter=text_splitter
    )

    count = await service.ingest_document(source_id="doc1.pdf", content="Content to split")

    assert vector_store.upsert_called is True
    assert count == 3
    assert vector_store.last_upserted_chunks == [
        DocumentChunk(
            id="doc1.pdf#0",
            content="chunk1",
            metadata={"source_id": "doc1.pdf", "chunk_index": 0},
        ),
        DocumentChunk(
            id="doc1.pdf#1",
            content="chunk2",
            metadata={"source_id": "doc1.pdf", "chunk_index": 1},
        ),
        DocumentChunk(
            id="doc1.pdf#2",
            content="chunk3",
            metadata={"source_id": "doc1.pdf", "chunk_index": 2},
        ),
    ]

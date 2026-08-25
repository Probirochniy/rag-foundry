import pytest

from src.core.entities.rag import DocumentChunk, GeneratedAnswer, SearchResult, SparseVectorData
from src.core.services.rag_service import RAGService
from tests.unit.mocks.cache import CacheStoreMock
from tests.unit.mocks.embeddings import DenseEmbeddingsMock, SparseEmbeddingsMock
from tests.unit.mocks.llm import LLMClientMock
from tests.unit.mocks.splitter import TextSplitterMock
from tests.unit.mocks.vector_store import VectorStoreMock


@pytest.fixture
def rag_deps():
    return {
        "cache": CacheStoreMock(),
        "vector_store": VectorStoreMock(),
        "llm": LLMClientMock(),
        "text_splitter": TextSplitterMock(),
        "dense_embeddings": DenseEmbeddingsMock(),
        "sparse_embeddings": SparseEmbeddingsMock(),
    }


def create_service(deps: dict) -> RAGService:
    return RAGService(
        vector_store=deps["vector_store"],
        cache_store=deps["cache"],
        llm_client=deps["llm"],
        text_splitter=deps["text_splitter"],
        dense_embeddings=deps["dense_embeddings"],
        sparse_embeddings=deps["sparse_embeddings"],
    )


@pytest.mark.asyncio
async def test_rag_service_cache_hit(rag_deps) -> None:
    test_question = "как поднять кубер"
    cached_answer = GeneratedAnswer(answer="сложно...", sources=["doc_cached.pdf"], cached=True)
    rag_deps["cache"] = CacheStoreMock(initial_data={test_question: cached_answer})

    service = create_service(rag_deps)
    result = await service.ask(query=test_question, top_k=3)

    assert result == cached_answer
    assert rag_deps["vector_store"].search_called is False
    assert rag_deps["llm"].generate_called is False
    assert rag_deps["dense_embeddings"].embed_query_called is False
    assert rag_deps["sparse_embeddings"].embed_query_called is False


@pytest.mark.asyncio
async def test_rag_service_cache_miss(rag_deps) -> None:
    test_question = "how to run kubernetes"
    test_file = "k8s.pdf"

    search_mock = [SearchResult(content="Инструкция по куберу", source_id=test_file, score=0.95)]
    rag_deps["vector_store"] = VectorStoreMock(mock_results=search_mock)
    rag_deps["llm"] = LLMClientMock(default_answer="LLM response")

    service = create_service(rag_deps)
    result = await service.ask(query=test_question, top_k=5)

    assert rag_deps["dense_embeddings"].embed_query_called is True
    assert rag_deps["sparse_embeddings"].embed_query_called is True
    assert rag_deps["vector_store"].search_called is True
    assert rag_deps["vector_store"].last_dense_vector == [0.1, 0.2, 0.3]
    assert rag_deps["vector_store"].last_sparse_vector == SparseVectorData(
        indices=[1, 2], values=[0.5, 0.9]
    )
    assert rag_deps["vector_store"].last_top_k == 5
    assert rag_deps["llm"].generate_called is True
    assert rag_deps["cache"].set_called is True

    assert f"LLM response: {test_question}" in result.answer
    assert result.sources == ["k8s.pdf"]

    cached_val = await rag_deps["cache"].get(query=test_question, top_k=5)
    assert cached_val is not None
    assert cached_val.answer == result.answer


@pytest.mark.asyncio
async def test_rag_service_streaming_cache_hit(rag_deps) -> None:
    test_question = "stream query"
    cached_answer = GeneratedAnswer(answer="streamed answer", sources=["doc1.pdf"], cached=True)
    rag_deps["cache"] = CacheStoreMock(initial_data={test_question: cached_answer})

    service = create_service(rag_deps)

    chunks: list[str] = []
    async for chunk in service.ask_stream(query=test_question, top_k=3):
        chunks.append(chunk)

    assert chunks == ["streamed answer"]
    assert rag_deps["vector_store"].search_called is False
    assert rag_deps["llm"].generate_called is False
    assert rag_deps["dense_embeddings"].embed_query_called is False


@pytest.mark.asyncio
async def test_rag_service_streaming_cache_miss(rag_deps) -> None:
    service = create_service(rag_deps)

    chunks: list[str] = []
    async for chunk in service.ask_stream(query="stream query", top_k=3):
        chunks.append(chunk)

    assert chunks == ["chunk1 ", "chunk2 ", "chunk3"]
    assert rag_deps["dense_embeddings"].embed_query_called is True
    assert rag_deps["sparse_embeddings"].embed_query_called is True
    assert rag_deps["vector_store"].search_called is True

    cached_val = await rag_deps["cache"].get(query="stream query", top_k=3)
    assert cached_val is not None


@pytest.mark.asyncio
async def test_rag_service_ingest_document(rag_deps) -> None:
    service = create_service(rag_deps)

    count = await service.ingest_document(source_id="doc1.pdf", content="Content to split")

    assert rag_deps["dense_embeddings"].embed_documents_called is True
    assert rag_deps["sparse_embeddings"].embed_documents_called is True
    assert rag_deps["vector_store"].upsert_called is True
    assert count == 3

    expected_dense = [0.1, 0.2, 0.3]
    expected_sparse = SparseVectorData(indices=[1, 2], values=[0.5, 0.9])

    assert rag_deps["vector_store"].last_upserted_chunks == [
        DocumentChunk(
            id="doc1.pdf#0",
            content="chunk1",
            metadata={"source_id": "doc1.pdf", "chunk_index": 0},
            dense_embedding=expected_dense,
            sparse_embedding=expected_sparse,
        ),
        DocumentChunk(
            id="doc1.pdf#1",
            content="chunk2",
            metadata={"source_id": "doc1.pdf", "chunk_index": 1},
            dense_embedding=expected_dense,
            sparse_embedding=expected_sparse,
        ),
        DocumentChunk(
            id="doc1.pdf#2",
            content="chunk3",
            metadata={"source_id": "doc1.pdf", "chunk_index": 2},
            dense_embedding=expected_dense,
            sparse_embedding=expected_sparse,
        ),
    ]

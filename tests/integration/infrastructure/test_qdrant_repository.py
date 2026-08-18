import pytest

from src.core.config import settings
from src.core.entities.rag import DocumentChunk
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@pytest.mark.asyncio
async def test_qdrant_repository_upsert_and_search() -> None:
    repo = QdrantRepository(
        url=settings.qdrant_url,
        collection_name="test_integration_collection",
        vector_size=settings.embedding_dimension,
    )

    assert await repo.is_healthy() is True

    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="Kubernetes управляет подами и нодами",
            metadata={"source_id": "k8s_guide.pdf"},
        ),
        DocumentChunk(
            id="chunk-2",
            content="FastAPI это современный асинхронный фреймворк для Python",
            metadata={"source_id": "fastapi_guide.pdf"},
        ),
    ]

    await repo.upsert(chunks)

    results = await repo.search(query="Kubernetes управляет подами и нодами", top_k=1)

    assert len(results) == 1
    assert results[0].source_id == "k8s_guide.pdf"
    assert "Kubernetes" in results[0].content

    await repo.close()

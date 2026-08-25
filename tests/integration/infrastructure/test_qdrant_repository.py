import pytest

from src.core.config import settings
from src.core.entities.rag import DocumentChunk
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@pytest.mark.asyncio
async def test_qdrant_hybrid_repository() -> None:
    repo = QdrantRepository(
        url=settings.qdrant_url,
        collection_name="test_hybrid_collection",
        vector_size=384,
    )

    is_healthy = await repo.is_healthy()
    assert is_healthy is True

    chunks = [
        DocumentChunk(
            id="doc-toaster",
            content="Error code ERR-8492 occurs when the toaster overheating protection activates."
            "Please reset the thermal fuse.",
            metadata={"source_id": "toaster_manual.md"},
        ),
        DocumentChunk(
            id="doc-devops",
            content="Kubernetes automates deployment, scaling, and "
            "management of containerized apps.",
            metadata={"source_id": "k8s.md"},
        ),
        DocumentChunk(
            id="doc-food",
            content="I can eat pepperoni pizza for breakfast lunch and dinner every single day.",
            metadata={"source_id": "pizza.md"},
        ),
    ]

    await repo.upsert(chunks)

    # Exact keyword search test (Sparse)
    sparse_results = await repo.search(query="my machine gives ERR-8492 what to do", top_k=1)
    assert len(sparse_results) == 1
    assert sparse_results[0].source_id == "toaster_manual.md"
    assert "ERR-8492" in sparse_results[0].content

    # Semanting meaning search test (Dense)
    dense_results = await repo.search(query="how to orchestrate docker nodes in cluster", top_k=1)
    assert len(dense_results) == 1
    assert dense_results[0].source_id == "k8s.md"

    await repo.close()

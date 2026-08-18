import pytest

from src.core.config import settings
from src.core.entities.rag import DocumentChunk
from src.infrastructure.embeddings.fastembed_adapter import FastEmbedAdapter
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@pytest.mark.asyncio
async def test_qdrant_repository() -> None:
    embeddings = FastEmbedAdapter()
    repo = QdrantRepository(
        url=settings.qdrant_url,
        collection_name="test_semantic_collection",
        embeddings=embeddings,
        vector_size=384,
    )

    is_healthy = await repo.is_healthy()
    assert is_healthy is True

    chunks = [
        DocumentChunk(
            id="doc-k8s",
            content="Kubernetes (also known as K8s) is an open-source system."
            "It automates how you run, update, and scale computer programs"
            "across many servers.",
            metadata={"source_id": "k8s.md"},
        ),
        DocumentChunk(
            id="doc-food",
            content="god i love pizza.",
            metadata={"source_id": "pizza.md"},
        ),
    ]

    await repo.upsert(chunks)

    # Same meaning, different wording
    results = await repo.search(
        query="How to manage docker containers in production clusters?", top_k=1
    )

    assert len(results) == 1
    assert results[0].source_id == "k8s.md"
    assert results[0].score > 0.5

    await repo.close()

import asyncio

import pytest

from src.core.config import settings
from src.core.entities.rag import DocumentChunk
from src.infrastructure.embeddings.sparse_embed_adapter import FastEmbedSparseAdapter
from src.infrastructure.embeddings.tei_embed_adapter import (
    TEIEmbedDenseAdapter,
)
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@pytest.mark.asyncio
async def test_qdrant_hybrid_repository() -> None:
    dense_embedder = TEIEmbedDenseAdapter(base_url=settings.tei_url)
    sparse_embedder = FastEmbedSparseAdapter()

    repo = QdrantRepository(
        url=settings.qdrant_url,
        collection_name="test_hybrid_collection",
        vector_size=384,
    )

    is_healthy = await repo.is_healthy()
    assert is_healthy is True

    raw_data = [
        (
            "doc-toaster",
            "Error code ERR-8492 occurs when the toaster overheating protection activates. "
            "Please reset the thermal fuse.",
            {"source_id": "toaster_manual.md"},
        ),
        (
            "doc-devops",
            "Kubernetes automates deployment, scaling, and management of containerized apps.",
            {"source_id": "k8s.md"},
        ),
        (
            "doc-food",
            "I can eat pepperoni pizza for breakfast lunch and dinner every single day.",
            {"source_id": "italian.md"},
        ),
    ]

    texts = [item[1] for item in raw_data]

    dense_vectors, sparse_vectors = await asyncio.gather(
        dense_embedder.embed_documents(texts),
        sparse_embedder.embed_documents(texts),
    )

    chunks = [
        DocumentChunk(
            id=item[0],
            content=item[1],
            metadata=item[2],
            dense_embedding=dense_vec,
            sparse_embedding=sparse_vec,
        )
        for item, dense_vec, sparse_vec in zip(raw_data, dense_vectors, sparse_vectors, strict=True)
    ]

    await repo.upsert(chunks)

    # Keyword search test (Sparse)
    sparse_query = "my machine gives ERR-8492 what to do"
    q1_dense, q1_sparse = await asyncio.gather(
        dense_embedder.embed_query(sparse_query),
        sparse_embedder.embed_query(sparse_query),
    )

    sparse_results = await repo.search(
        dense_vector=q1_dense,
        sparse_vector=q1_sparse,
        top_k=1,
    )
    assert len(sparse_results) == 1
    assert sparse_results[0].source_id == "toaster_manual.md"
    assert "ERR-8492" in sparse_results[0].content

    # Semanting meaning search test (Dense)
    dense_query = "how to orchestrate docker nodes in cluster"
    q2_dense, q2_sparse = await asyncio.gather(
        dense_embedder.embed_query(dense_query),
        sparse_embedder.embed_query(dense_query),
    )

    dense_results = await repo.search(
        dense_vector=q2_dense,
        sparse_vector=q2_sparse,
        top_k=1,
    )
    assert len(dense_results) == 1
    assert dense_results[0].source_id == "k8s.md"

    await repo.close()

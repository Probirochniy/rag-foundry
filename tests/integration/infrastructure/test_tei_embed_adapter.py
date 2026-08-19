import math

import pytest

from src.core.config import settings
from src.infrastructure.embeddings.tei_embed_adapter import TEIEmbedAdapter


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot_product / (norm_a * norm_b)


@pytest.mark.asyncio
async def test_tei_adapter_integration() -> None:
    adapter = TEIEmbedAdapter(
        base_url=settings.tei_url,
        batch_size=2,
    )

    is_healthy = await adapter.is_healthy()
    assert is_healthy is True

    query = "how to manage docker containers in kubernetes"
    query_vector = await adapter.embed_query(query)

    assert len(query_vector) == settings.embedding_dimension
    assert all(isinstance(x, float) for x in query_vector)
    assert any(x != 0.0 for x in query_vector)

    docs = [
        "k8s is an open source system for automating deployment and scaling",
        "docker compose is a tool for running multi-container applications",
        "i really love eating hot pepperoni pizza with cheese",
    ]
    doc_vectors = await adapter.embed_documents(docs)

    assert len(doc_vectors) == 3
    for vec in doc_vectors:
        assert len(vec) == settings.embedding_dimension

    empty_vectors = await adapter.embed_documents([])
    assert empty_vectors == []

    sim_k8s = cosine_similarity(query_vector, doc_vectors[0])
    sim_pizza = cosine_similarity(query_vector, doc_vectors[2])

    assert sim_k8s > sim_pizza

    await adapter.close()

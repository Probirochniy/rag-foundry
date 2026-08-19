import pytest

from src.infrastructure.embeddings.fastembed_adapter import FastEmbedAdapter


@pytest.mark.asyncio
async def test_fastembed_adapter_embeds_queries_and_documents() -> None:
    adapter = FastEmbedAdapter()

    query_embedding = await adapter.embed_query("What is retrieval augmented generation?")
    document_embeddings = await adapter.embed_documents(
        ["RAG combines retrieval with generation.", "FastEmbed provides local embeddings."]
    )

    assert await adapter.is_healthy()
    assert query_embedding
    assert len(document_embeddings) == 2
    assert all(document_embedding for document_embedding in document_embeddings)
    assert len(query_embedding) == len(document_embeddings[0])

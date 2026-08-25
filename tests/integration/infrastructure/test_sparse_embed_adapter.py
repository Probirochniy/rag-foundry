import pytest

from src.infrastructure.embeddings.sparse_embed_adapter import FastEmbedSparseAdapter


@pytest.mark.asyncio
async def test_fastembed_sparse_adapter_embeds_queries_and_documents() -> None:
    adapter = FastEmbedSparseAdapter()

    assert await adapter.is_healthy() is True

    query = "nuclear reactor core meltdown HELP"
    sparse_query = await adapter.embed_query(query)

    assert len(sparse_query.indices) > 0
    assert len(sparse_query.indices) == len(sparse_query.values)
    assert all(isinstance(idx, int) for idx in sparse_query.indices)
    assert all(isinstance(val, float) for val in sparse_query.values)

    docs = [
        "reactor cooling systems maintenance manual",
        "how to bake amazing chocolate cookies",
    ]
    sparse_docs = await adapter.embed_documents(docs)

    assert len(sparse_docs) == 2
    assert len(sparse_docs[0].indices) > 0
    assert len(sparse_docs[1].indices) > 0

    empty_res = await adapter.embed_documents([])
    assert empty_res == []

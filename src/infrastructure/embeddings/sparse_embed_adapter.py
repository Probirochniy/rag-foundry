import asyncio
from collections.abc import Sequence

from fastembed import SparseTextEmbedding

from src.core.entities.rag import SparseVectorData
from src.core.protocols.embeddings import SparseEmbeddingsProtocol


class FastEmbedSparseAdapter(SparseEmbeddingsProtocol):
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    async def embed_query(self, text: str) -> SparseVectorData:
        result = await asyncio.to_thread(lambda: list(self._model.embed(text)))
        sparse_vec = result[0]
        return SparseVectorData(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        result = await asyncio.to_thread(lambda: list(self._model.embed(texts)))
        return [
            SparseVectorData(
                indices=vec.indices.tolist(),
                values=vec.values.tolist(),
            )
            for vec in result
        ]

    async def is_healthy(self) -> bool:
        return True

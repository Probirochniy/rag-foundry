import asyncio
from collections.abc import Sequence
from typing import cast

from fastembed import TextEmbedding

from src.core.protocols.embeddings import EmbeddingsProtocol


class FastEmbedAdapter(EmbeddingsProtocol):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name=model_name)

    async def embed_query(self, text: str) -> list[float]:
        # fastembed is synchronous, need to_thread
        embeddings = await asyncio.to_thread(
            lambda: cast(list[Sequence[float]], list(self._model.embed(text)))
        )
        return [float(value) for value in embeddings[0]]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            lambda: cast(list[Sequence[float]], list(self._model.embed(texts)))
        )
        return [[float(value) for value in embedding] for embedding in embeddings]

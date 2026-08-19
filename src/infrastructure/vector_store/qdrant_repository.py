import logging
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.core.entities.rag import DocumentChunk, SearchResult
from src.core.protocols.embeddings import EmbeddingsProtocol
from src.core.protocols.vector_store import VectorStoreProtocol

logger = logging.getLogger(__name__)


class QdrantRepository(VectorStoreProtocol):
    def __init__(
        self,
        url: str,
        collection_name: str,
        embeddings: EmbeddingsProtocol,
        vector_size: int = 384,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._embeddings = embeddings
        self._client = AsyncQdrantClient(url=self._url)

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection_exists(self) -> None:
        collections_response = await self._client.get_collections()
        existing = [c.name for c in collections_response.collections]

        if self._collection_name not in existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection_name}")

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        texts_to_embed = [chunk.content for chunk in chunks]
        vectors = await self._embeddings.embed_documents(texts_to_embed)

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "source_id": chunk.metadata.get("source_id", "unknown"),
                "metadata": chunk.metadata,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )

    async def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = await self._embeddings.embed_query(query)

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                content=str(point.payload.get("content", "") if point.payload else ""),
                source_id=str(point.payload.get("source_id", "unknown") if point.payload else ""),
                score=float(point.score),
                metadata=dict(point.payload.get("metadata", {}) if point.payload else {}),
            )
            for point in response.points
        ]

    async def is_healthy(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

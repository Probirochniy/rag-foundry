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
        vector_size: int = 1536,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._embeddings = embeddings
        self._client: AsyncQdrantClient | None = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _ensure_collection_exists(self) -> None:
        client = self._get_client()
        collections_response = await client.get_collections()
        existing = [c.name for c in collections_response.collections]

        if self._collection_name not in existing:
            await client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection_name}")

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        client = self._get_client()
        await self._ensure_collection_exists()

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
            await client.upsert(
                collection_name=self._collection_name,
                points=points,
            )

    async def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        client = self._get_client()
        await self._ensure_collection_exists()

        query_vector = await self._embeddings.embed_query(query)

        response = await client.query_points(
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
            client = self._get_client()
            await client.get_collections()
            return True
        except Exception:
            return False

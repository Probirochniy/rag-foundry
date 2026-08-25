import asyncio
import itertools
import logging
import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from src.core.config import settings
from src.core.entities.rag import DocumentChunk, SearchResult, SparseVectorData
from src.core.protocols.vector_store import VectorStoreProtocol

logger = logging.getLogger(__name__)


class QdrantRepository(VectorStoreProtocol):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int = 384,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = AsyncQdrantClient(url=self._url)
        self._collection_ensured = False
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection_exists(self) -> None:
        if self._collection_ensured:
            return

        async with self._lock:
            collections_response = await self._client.get_collections()
            existing = [c.name for c in collections_response.collections]

            if self._collection_name not in existing:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        settings.dense_vector_name: models.VectorParams(
                            size=self._vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        settings.sparse_vector_name: models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    },
                )
                logger.info(f"Created hybrid Qdrant collection: {self._collection_name}")

            self._collection_ensured = True

    async def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        await self.ensure_collection_exists()
        if not chunks:
            return

        batch_size = settings.qdrant_upsert_batch_size

        for chunk_batch in itertools.batched(chunks, batch_size):
            points: list[models.PointStruct] = []
            for chunk in chunk_batch:
                if not chunk.dense_embedding or not chunk.sparse_embedding:
                    raise ValueError(f"chunk {chunk.id} is missing embeddings you dumb fuck")

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
                payload = {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "source_id": chunk.metadata.get("source_id", "unknown"),
                    "metadata": chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            settings.dense_vector_name: chunk.dense_embedding,
                            settings.sparse_vector_name: models.SparseVector(
                                indices=chunk.sparse_embedding.indices,
                                values=chunk.sparse_embedding.values,
                            ),
                        },
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=False,
            )

    async def search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 3,
    ) -> list[SearchResult]:
        await self.ensure_collection_exists()

        response = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=settings.dense_vector_name,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using=settings.sparse_vector_name,
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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

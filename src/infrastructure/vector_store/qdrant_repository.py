from qdrant_client import AsyncQdrantClient

from src.core.protocols.vector_store import VectorStoreProtocol


class QdrantRepository(VectorStoreProtocol):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int = 1536,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client: AsyncQdrantClient | None = None

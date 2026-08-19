import itertools
import logging
from collections.abc import Sequence

import httpx

from src.core.protocols.embeddings import EmbeddingsProtocol

logger = logging.getLogger(__name__)


class TEIEmbedAdapter(EmbeddingsProtocol):
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        batch_size: int = 32,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._batch_size = batch_size
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.post(
            "/embed",
            json={"inputs": text, "truncate": True},
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                return [float(x) for x in data[0]]
            return [float(x) for x in data]

        raise ValueError(f"Unexpected response format from TEI: {data}")

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch in itertools.batched(texts, self._batch_size):
            response = await self._client.post(
                "/embed",
                json={"inputs": list(batch), "truncate": True},
            )
            response.raise_for_status()
            batch_data: list[list[float]] = response.json()
            all_embeddings.extend([[float(val) for val in vec] for vec in batch_data])

        return all_embeddings

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

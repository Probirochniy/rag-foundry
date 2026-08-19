import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from src.core.entities.rag import GeneratedAnswer
from src.core.protocols.cache import CacheStoreProtocol

logger = logging.getLogger(__name__)


class RedisCacheRepository(CacheStoreProtocol):
    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _generate_key(self, query: str, top_k: int) -> str:
        normalized_query = query.strip().lower()
        composite_raw = f"{normalized_query}:top_k={top_k}"
        query_hash = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()
        return f"rag:cache:{query_hash}"

    async def get(self, query: str, top_k: int = 3) -> GeneratedAnswer | None:
        try:
            key = self._generate_key(query, top_k)
            data = await self._client.get(key)
            if not data:
                return None

            payload: dict[str, Any] = json.loads(data)
            return GeneratedAnswer(
                answer=payload["answer"],
                sources=payload.get("sources", []),
                cached=True,
            )
        except Exception as err:
            logger.warning(f"Failed to read from Redis cache: {err}")
            return None

    async def set(
        self,
        query: str,
        answer: GeneratedAnswer,
        top_k: int = 3,
        ttl_seconds: int = 3600,
    ) -> None:
        try:
            key = self._generate_key(query, top_k)
            payload = {
                "answer": answer.answer,
                "sources": answer.sources,
            }
            await self._client.set(
                name=key,
                value=json.dumps(payload),
                ex=ttl_seconds,
            )
        except Exception as err:
            logger.warning(f"Failed to write to Redis cache: {err}")

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.ping()
            return bool(response)
        except Exception:
            return False

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
        self._client: aioredis.Redis | None = None

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _generate_key(self, query: str) -> str:
        normalized_query = query.strip().lower()
        query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        return f"rag:cache:{query_hash}"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, query: str) -> GeneratedAnswer | None:
        try:
            client = self._get_client()
            key = self._generate_key(query)
            data = await client.get(key)
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

    async def set(self, query: str, answer: GeneratedAnswer, ttl_seconds: int = 3600) -> None:
        try:
            client = self._get_client()
            key = self._generate_key(query)
            payload = {
                "answer": answer.answer,
                "sources": answer.sources,
            }
            await client.set(
                name=key,
                value=json.dumps(payload),
                ex=ttl_seconds,
            )
        except Exception as err:
            logger.warning(f"Failed to write to Redis cache: {err}")

    async def is_healthy(self) -> bool:
        try:
            client = self._get_client()
            response = await client.ping()
            return bool(response)
        except Exception:
            return False

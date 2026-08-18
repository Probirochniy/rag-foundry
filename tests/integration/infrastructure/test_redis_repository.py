import pytest

from src.core.config import settings
from src.core.entities.rag import GeneratedAnswer
from src.infrastructure.cache.redis_repository import RedisCacheRepository


@pytest.mark.asyncio
async def test_redis_cache_set_and_get() -> None:
    repo = RedisCacheRepository(redis_url=settings.redis_url)

    is_healthy = await repo.is_healthy()
    assert is_healthy is True

    query = "что такое kubernetes"
    original_answer = GeneratedAnswer(
        answer="не знаю :(",
        sources=["k8s_docs.md"],
        cached=False,
    )

    await repo.set(query=query, answer=original_answer, ttl_seconds=60)

    cached_result = await repo.get("  ЧТО ТАКОЕ KUBERNETES  ")
    assert cached_result is not None
    assert cached_result.answer == original_answer.answer
    assert cached_result.sources == original_answer.sources
    assert cached_result.cached is True

    await repo.close()

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.di_container import get_cache_repository, get_embeddings_adapter, get_vector_repository
from src.api.system.models.health import HealthResponse
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.embeddings import EmbeddingsProtocol
from src.core.protocols.vector_store import VectorStoreProtocol

router = APIRouter(tags=["System & Probes"])


@router.get("/healthz", response_model=HealthResponse, status_code=200)
async def liveness_probe() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/readyz", response_model=HealthResponse, status_code=200)
async def readiness_probe(
    cache: Annotated[CacheStoreProtocol, Depends(get_cache_repository)],
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_repository)],
    embeddings: Annotated[EmbeddingsProtocol, Depends(get_embeddings_adapter)],
) -> HealthResponse:
    cache_ok = await cache.is_healthy()
    vector_ok = await vector_store.is_healthy()
    embeddings_ok = await embeddings.is_healthy()

    if not (cache_ok and vector_ok and embeddings_ok):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Dependencies unavailable: Redis={cache_ok},"
                f" Qdrant={vector_ok}, Embeddings={embeddings_ok}"
            ),
        )

    return HealthResponse(status="ready")

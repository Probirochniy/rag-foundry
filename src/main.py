from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.api.di_container import get_cache_repository, get_vector_repository
from src.api.router import root_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    vector_repo = get_vector_repository()
    await vector_repo.ensure_collection_exists()

    yield

    cache_repo = get_cache_repository()
    await cache_repo.close()
    await vector_repo.close()


def create_app() -> FastAPI:
    application = FastAPI(
        title="RAG Foundry Engine",
        description="Production-ready RAG microservice with observability",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.include_router(root_router)

    return application


app = create_app()

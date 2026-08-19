from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.api.router import root_router
from src.core.config import settings
from src.infrastructure.cache.redis_repository import RedisCacheRepository
from src.infrastructure.embeddings.tei_embed_adapter import TEIEmbedAdapter
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    embeddings = TEIEmbedAdapter(base_url=settings.tei_url)
    app.state.embeddings_repo = embeddings
    app.state.cache_repo = RedisCacheRepository(settings.redis_url)
    app.state.vector_repo = QdrantRepository(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        embeddings=embeddings,
        vector_size=settings.embedding_dimension,
    )
    await app.state.vector_repo.ensure_collection_exists()

    yield

    await app.state.cache_repo.close()
    await app.state.vector_repo.close()
    await app.state.embeddings_repo.close()


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

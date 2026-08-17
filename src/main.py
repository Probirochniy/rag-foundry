from fastapi import FastAPI

from src.api.router import root_router


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

from fastapi import APIRouter

from src.api.v1.routers.rag import router as rag_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(rag_router)

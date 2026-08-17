from fastapi import APIRouter

from src.api.system.routers.health import router as health_router

system_router = APIRouter(tags=["System & Probes"])
system_router.include_router(health_router)

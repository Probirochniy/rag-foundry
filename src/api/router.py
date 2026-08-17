from fastapi import APIRouter

from src.api.system.router import system_router
from src.api.v1.router import v1_router

root_router = APIRouter()

root_router.include_router(system_router)
root_router.include_router(v1_router)

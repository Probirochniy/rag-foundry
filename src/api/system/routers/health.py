from fastapi import APIRouter

from src.api.system.models.responses.health.health_response import HealthResponse

router = APIRouter(tags=["System & Probes"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=200,
)
async def liveness_probe() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get(
    "/readyz",
    response_model=HealthResponse,
    status_code=200,
)
async def readiness_probe() -> HealthResponse:
    return HealthResponse(status="ready")

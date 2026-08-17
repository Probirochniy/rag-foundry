import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_healthz_returns_200(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_readyz_returns_200(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

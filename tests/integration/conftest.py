from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.entities.rag import SearchResult
from src.main import app, lifespan


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
def mock_context():
    return [
        SearchResult(
            source_id="based_doc",
            content="Based is when you agree with something; or when you want to recognize"
            " someone for being themselves.",
            score=0.95,
            metadata={"title": "Based Doc"},
        ),
        SearchResult(
            source_id="cringe_doc",
            content="Cringe is when someone acts/ or is so embarrassing or awkward,"
            " it makes you feel extemely ashamed and/or embarrassed.",
            score=0.5,
            metadata={"title": "Cringe Doc"},
        ),
    ]

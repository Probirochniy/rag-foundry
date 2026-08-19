from collections.abc import Callable

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter


@pytest.fixture
def make_adapter() -> Callable[[list[str]], LangChainLLMAdapter]:
    def _factory(responses: list[str]) -> LangChainLLMAdapter:
        return LangChainLLMAdapter(llm=FakeListChatModel(responses=responses))

    return _factory


@pytest.mark.asyncio
async def test_generate_answer(mock_context, make_adapter):
    adapter = make_adapter(
        [
            "Based answer.",
        ]
    )
    result = await adapter.generate_answer(
        query="am i based?",
        context=mock_context,
    )

    assert result.answer == "Based answer."
    assert set(result.sources) == {"based_doc", "cringe_doc"}


@pytest.mark.asyncio
async def test_generate_stream(mock_context, make_adapter):
    adapter = make_adapter(
        [
            "Based answer.",
        ]
    )
    chunks: list[str] = []
    async for chunk in adapter.generate_stream(
        query="am i based?",
        context=mock_context,
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "Based answer."

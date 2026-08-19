from collections.abc import Callable

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.infrastructure.llm.langgraph_adapter import LangGraphLLMAdapter


@pytest.fixture
def make_adapter() -> Callable[[list[str]], LangGraphLLMAdapter]:
    def _factory(responses: list[str]) -> LangGraphLLMAdapter:
        return LangGraphLLMAdapter(llm=FakeListChatModel(responses=responses))

    return _factory


@pytest.mark.asyncio
async def test_generate_answer_happy_path(mock_context, make_adapter):
    adapter = make_adapter(
        [
            "Based answer.",
            '{"is_faithful": true, "reasoning": "strictly grounded in context"}',
        ]
    )
    result = await adapter.generate_answer(
        query="am i based?",
        context=mock_context,
    )

    assert result.answer == "Based answer."
    assert set(result.sources) == {"based_doc", "cringe_doc"}


@pytest.mark.asyncio
async def test_generate_answer_hallucination_retry(mock_context, make_adapter):
    adapter = make_adapter(
        [
            "Cringe answer.",
            '{"is_faithful": false, "reasoning": "hallucination found"}',
            "Based answer.",
            '{"is_faithful": true, "reasoning": "perfect match"}',
        ]
    )
    result = await adapter.generate_answer(
        query="am i based?",
        context=mock_context,
    )

    assert result.answer == "Based answer."


@pytest.mark.asyncio
async def test_generate_stream_hallucination_retry(mock_context, make_adapter):
    adapter = make_adapter(
        [
            "Cringe answer.",
            '{"is_faithful": false, "reasoning": "hallucination"}',
            "Based answer.",
            '{"is_faithful": true, "reasoning": "all good"}',
        ]
    )

    chunks = []
    async for chunk in adapter.generate_stream(
        query="am i based?",
        context=mock_context,
    ):
        chunks.append(chunk)

    full_output = "".join(chunks)

    assert "Cringe answer." in full_output
    assert "[RESET]" in full_output
    assert "Based answer." in full_output

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.infrastructure.llm.langgraph_adapter import LangGraphLLMAdapter


@pytest.mark.asyncio
async def test_generate_answer_happy_path(mock_context):
    mock_llm = FakeListChatModel(
        responses=[
            "Based answer.",
            "YES, strictly grounded in context",
        ]
    )

    adapter = LangGraphLLMAdapter(llm=mock_llm)
    result = await adapter.generate_answer(
        query="am i based?",
        context=mock_context,
    )

    assert result.answer == "Based answer."
    assert set(result.sources) == {"based_doc", "cringe_doc"}


@pytest.mark.asyncio
async def test_generate_answer_with_hallucination_retry(mock_context):
    mock_llm = FakeListChatModel(
        responses=[
            "Cringe answer.",
            "NO, hallucination found",
            "Based answer.",
            "YES, perfect context match",
        ]
    )

    adapter = LangGraphLLMAdapter(llm=mock_llm)
    result = await adapter.generate_answer(
        query="am i based?",
        context=mock_context,
    )

    assert result.answer == "Based answer."


@pytest.mark.asyncio
async def test_generate_stream_hallucination_retry_warning(mock_context):
    mock_llm = FakeListChatModel(
        responses=[
            "Cringe answer.",
            "NO",
            "Based answer.",
            "YES",
        ]
    )

    adapter = LangGraphLLMAdapter(llm=mock_llm)

    chunks = []
    async for chunk in adapter.generate_stream(
        query="am i based?",
        context=mock_context,
    ):
        chunks.append(chunk)

    full_output = "".join(chunks)

    assert "Cringe answer." in full_output
    assert "⚠️ HALLUCINATION DETECTED! Retrying:" in full_output
    assert "Based answer." in full_output

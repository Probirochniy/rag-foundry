import logging
from collections.abc import AsyncIterator

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import settings
from src.core.entities.rag import GeneratedAnswer, SearchResult
from src.core.protocols.llm import LLMClientProtocol

logger = logging.getLogger(__name__)


class LangChainLLMAdapter(LLMClientProtocol):
    """Simple adapter for LangChain's ChatOpenAI to fit the LLMClientProtocol interface."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def _format_context(self, context: list[SearchResult]) -> str:
        if not context:
            return "Context is empty."
        return "\n\n---\n\n".join([f"[Source: {c.source_id}]:\n{c.content}" for c in context])

    async def generate_answer(self, query: str, context: list[SearchResult]) -> GeneratedAnswer:
        formatted = self._format_context(context)
        system_msg = SystemMessage(content=settings.rag_system_prompt.format(context=formatted))
        user_msg = HumanMessage(content=query)

        response = await self._llm.ainvoke([system_msg, user_msg])
        sources = list({c.source_id for c in context})
        return GeneratedAnswer(answer=str(response.content), sources=sources, cached=False)

    async def generate_stream(self, query: str, context: list[SearchResult]) -> AsyncIterator[str]:
        formatted = self._format_context(context)
        system_msg = SystemMessage(content=settings.rag_system_prompt.format(context=formatted))
        user_msg = HumanMessage(content=query)

        async for chunk in self._llm.astream([system_msg, user_msg]):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.core.config import settings
from src.core.entities.rag import GeneratedAnswer, SearchResult
from src.core.protocols.llm import LLMClientProtocol

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


class GraphState(TypedDict):
    query: str
    context_text: str
    sources: list[str]
    answer: str
    is_faithful: bool
    retry_count: int


class LangGraphLLMAdapter(LLMClientProtocol):
    """Enhanced adapter with a graph-based approach for hallucination mitigation."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._graph = self._build_graph()

    def _format_context(self, context: list[SearchResult]) -> str:
        if not context:
            return "Context is empty."
        return "\n\n---\n\n".join([f"[Source: {c.source_id}]:\n{c.content}" for c in context])

    def _build_graph(self) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
        workflow = StateGraph(GraphState)

        async def generate_node(state: GraphState) -> dict[str, Any]:
            prompt = settings.rag_system_prompt.format(context=state["context_text"])

            if state["retry_count"] > 0:
                prompt += settings.hallucination_warning_prompt

            response = await self._llm.ainvoke(
                [SystemMessage(content=prompt), HumanMessage(content=state["query"])]
            )
            return {
                "answer": str(response.content),
                "retry_count": state["retry_count"] + 1,
            }

        async def validate_node(state: GraphState) -> dict[str, Any]:
            critic_prompt = settings.hallucination_critic_prompt.format(
                answer=state["answer"], context=state["context_text"]
            )
            response = await self._llm.ainvoke([HumanMessage(content=critic_prompt)])
            is_faithful = "YES" in str(response.content).upper()
            return {"is_faithful": is_faithful}

        def route_validation(state: GraphState) -> Literal["end", "retry"]:
            if state["is_faithful"] or state["retry_count"] > MAX_RETRIES:
                return "end"
            logger.warning("Hallucination detected in RAG graph, retrying...")
            return "retry"

        workflow.add_node("generate", generate_node)
        workflow.add_node("validate", validate_node)

        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            route_validation,
            {
                "end": END,
                "retry": "generate",
            },
        )

        return workflow.compile()

    async def generate_answer(self, query: str, context: list[SearchResult]) -> GeneratedAnswer:
        formatted = self._format_context(context)
        sources = list({c.source_id for c in context})

        initial_state: GraphState = {
            "query": query,
            "context_text": formatted,
            "sources": sources,
            "answer": "",
            "is_faithful": True,
            "retry_count": 0,
        }

        final_state = await self._graph.ainvoke(initial_state)

        return GeneratedAnswer(
            answer=str(final_state["answer"]),
            sources=sources,
            cached=False,
        )

    async def generate_stream(self, query: str, context: list[SearchResult]) -> AsyncIterator[str]:
        formatted = self._format_context(context)
        sources = list({c.source_id for c in context})

        initial_state: GraphState = {
            "query": query,
            "context_text": formatted,
            "sources": sources,
            "answer": "",
            "is_faithful": True,
            "retry_count": 0,
        }

        seen_validate = False
        is_retrying = False

        async for msg_chunk, metadata in self._graph.astream(initial_state, stream_mode="messages"):
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None

            if node == "validate":
                seen_validate = True
                continue

            if node == "generate":
                if seen_validate and not is_retrying:
                    yield "\n\n⚠️ HALLUCINATION DETECTED! Retrying:\n\n"
                    is_retrying = True

                if isinstance(msg_chunk, BaseMessage) and msg_chunk.content:
                    if isinstance(msg_chunk.content, str):
                        yield msg_chunk.content
                    elif isinstance(msg_chunk.content, list):
                        for part in msg_chunk.content:
                            if isinstance(part, str):
                                yield part
                            elif isinstance(part, dict) and "text" in part:
                                yield str(part["text"])

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.entities.rag import GeneratedAnswer, SearchResult
from src.core.protocols.llm import LLMClientProtocol

logger = logging.getLogger(__name__)


class FactCheckEvaluation(BaseModel):
    is_faithful: bool = Field(
        description="True if the answer is grounded EXCLUSIVELY in the context, False otherwise."
    )
    reasoning: str = Field(default="", description="Brief reasoning behind the verdict.")


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

    async def _evaluate_with_fallback(self, critic_prompt: str) -> bool:
        # Attempt structured output evaluation first
        try:
            structured_llm = self._llm.with_structured_output(FactCheckEvaluation)
            res = await structured_llm.ainvoke([HumanMessage(content=critic_prompt)])
            if isinstance(res, FactCheckEvaluation):
                return res.is_faithful
            if isinstance(res, dict):
                return bool(res.get("is_faithful", True))
        except NotImplementedError:
            logger.info(
                "with_structured_output not supported, falling back to prompt-based JSON parsing"
            )
        except Exception as e:
            logger.warning(f"Structured output failed: {e}, attempting fallback")

        # Fallback to simple questioning
        prompt = (
            f"{critic_prompt}\n\n"
            "IMPORTANT:"
            "Evaluate the answer and respond EXCLUSIVELY with a JSON object in this format:\n"
            '{"reasoning": "brief fact check explanation", "is_faithful": true}\n'
            "or\n"
            '{"reasoning": "brief fact check explanation", "is_faithful": false}'
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            raw_text = str(response.content).strip()
            match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if match:
                data: dict[str, Any] = json.loads(match.group(0))
                if "is_faithful" in data:
                    return bool(data["is_faithful"])

            logger.warning(
                f"LLM returned invalid JSON for critic: {raw_text}. Falling back to faithful."
            )
        except Exception as e:
            logger.warning(f"Critic evaluation failed with exception: {e}")

        # fail-open. seethe cry but avoid blocking the pipeline
        return True

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

            is_faithful = await self._evaluate_with_fallback(critic_prompt)
            return {"is_faithful": is_faithful}

        def route_validation(state: GraphState) -> Literal["end", "retry"]:
            if state["is_faithful"] or state["retry_count"] > settings.max_hallucination_retries:
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

        async for event in self._graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node")

            if node_name == "validate" and kind == "on_chain_end":
                seen_validate = True
                continue

            if node_name == "generate" and kind == "on_chat_model_stream":
                if seen_validate and not is_retrying:
                    yield settings.hallucination_marker
                    is_retrying = True

                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
                    yield chunk.content

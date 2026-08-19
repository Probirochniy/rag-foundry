import typing
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from src.core.config import Settings, settings
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.embeddings import EmbeddingsProtocol
from src.core.protocols.llm import LLMClientProtocol
from src.core.protocols.vector_store import VectorStoreProtocol
from src.core.services.rag_service import RAGService
from src.infrastructure.embeddings.fastembed_adapter import FastEmbedAdapter
from src.infrastructure.embeddings.tei_embed_adapter import TEIEmbedAdapter
from src.infrastructure.llm.factory import create_llm
from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter
from src.infrastructure.llm.langgraph_adapter import LangGraphLLMAdapter


@lru_cache
def get_settings() -> Settings:
    return settings


@lru_cache
def get_fastembed_adapter() -> EmbeddingsProtocol:
    return FastEmbedAdapter()


def get_embeddings_adapter() -> EmbeddingsProtocol:
    return TEIEmbedAdapter(
        base_url=settings.tei_url,
        batch_size=settings.tei_batch_size,
    )


def get_cache_repository(request: Request) -> CacheStoreProtocol:
    return typing.cast(CacheStoreProtocol, request.app.state.redis)


def get_vector_repository(request: Request) -> VectorStoreProtocol:
    return typing.cast(VectorStoreProtocol, request.app.state.vector_repo)


@lru_cache
def get_langchain_client() -> LLMClientProtocol:
    llm = create_llm()
    return LangChainLLMAdapter(llm)


@lru_cache
def get_langgraph_client() -> LLMClientProtocol:
    llm = create_llm()
    return LangGraphLLMAdapter(llm)


def get_rag_service(
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_repository)],
    cache_store: Annotated[CacheStoreProtocol, Depends(get_cache_repository)],
    llm_client: Annotated[LLMClientProtocol, Depends(get_langgraph_client)],
) -> RAGService:
    return RAGService(
        vector_store=vector_store,
        cache_store=cache_store,
        llm_client=llm_client,
        cache_ttl_seconds=settings.redis_cache_ttl_seconds,
    )

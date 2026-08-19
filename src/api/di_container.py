from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.core.config import Settings, settings
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.embeddings import EmbeddingsProtocol
from src.core.protocols.llm import LLMClientProtocol
from src.core.protocols.vector_store import VectorStoreProtocol
from src.core.services.rag_service import RAGService
from src.infrastructure.cache.redis_repository import RedisCacheRepository
from src.infrastructure.embeddings.fastembed_adapter import FastEmbedAdapter
from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter
from src.infrastructure.llm.langgraph_adapter import LangGraphLLMAdapter
from src.infrastructure.vector_store.qdrant_repository import QdrantRepository


@lru_cache
def get_settings() -> Settings:
    return settings


@lru_cache
def get_embeddings_adapter() -> EmbeddingsProtocol:
    return FastEmbedAdapter()


@lru_cache
def get_cache_repository() -> CacheStoreProtocol:
    return RedisCacheRepository(redis_url=settings.redis_url)


@lru_cache
def get_vector_repository() -> VectorStoreProtocol:
    return QdrantRepository(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        embeddings=get_embeddings_adapter(),
        vector_size=settings.embedding_dimension,
    )


@lru_cache
def get_langchain_client() -> LLMClientProtocol:
    return LangChainLLMAdapter(
        api_key=settings.openai_api_key,
        model_name=settings.openai_model,
    )


@lru_cache
def get_langgraph_client() -> LLMClientProtocol:
    return LangGraphLLMAdapter(
        api_key=settings.openai_api_key,
        model_name=settings.openai_model,
    )


def get_rag_service(
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_repository)],
    cache_store: Annotated[CacheStoreProtocol, Depends(get_cache_repository)],
    llm_client: Annotated[LLMClientProtocol, Depends(get_langgraph_client)],
) -> RAGService:
    return RAGService(
        vector_store=vector_store,
        cache_store=cache_store,
        llm_client=llm_client,
    )

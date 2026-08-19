import typing
from typing import Annotated

from fastapi import Depends, Request

from src.core.config import settings
from src.core.protocols.cache import CacheStoreProtocol
from src.core.protocols.embeddings import EmbeddingsProtocol
from src.core.protocols.llm import LLMClientProtocol
from src.core.protocols.splitter import TextSplitterProtocol
from src.core.protocols.vector_store import VectorStoreProtocol
from src.core.services.rag_service import RAGService


def get_embeddings_adapter(request: Request) -> EmbeddingsProtocol:
    return typing.cast(EmbeddingsProtocol, request.app.state.embeddings_repo)


def get_cache_repository(request: Request) -> CacheStoreProtocol:
    return typing.cast(CacheStoreProtocol, request.app.state.cache_repo)


def get_vector_repository(request: Request) -> VectorStoreProtocol:
    return typing.cast(VectorStoreProtocol, request.app.state.vector_repo)


def get_llm_client(request: Request) -> LLMClientProtocol:
    return typing.cast(LLMClientProtocol, request.app.state.llm_client)


def get_text_splitter(request: Request) -> TextSplitterProtocol:
    return typing.cast(TextSplitterProtocol, request.app.state.text_splitter)


def get_rag_service(
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_repository)],
    cache_store: Annotated[CacheStoreProtocol, Depends(get_cache_repository)],
    llm_client: Annotated[LLMClientProtocol, Depends(get_llm_client)],
    text_splitter: Annotated[TextSplitterProtocol, Depends(get_text_splitter)],
) -> RAGService:
    return RAGService(
        vector_store=vector_store,
        cache_store=cache_store,
        llm_client=llm_client,
        text_splitter=text_splitter,
        cache_ttl_seconds=settings.redis_cache_ttl_seconds,
    )

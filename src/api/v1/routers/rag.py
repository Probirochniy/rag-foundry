import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.di_container import get_rag_service
from src.api.v1.models.ingest import IngestRequest, IngestResponse
from src.api.v1.models.query import QueryRequest, QueryResponse
from src.core.config import settings
from src.core.entities.rag import GeneratedAnswer
from src.core.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=200,
    summary="Execute RAG query",
)
async def query_rag(
    request: QueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> QueryResponse:
    answer: GeneratedAnswer = await rag_service.ask(query=request.query, top_k=request.top_k)

    return QueryResponse.from_generated_answer(answer)


@router.post(
    "/stream",
    status_code=200,
    summary="Stream RAG query response (SSE)",
)
async def stream_rag(
    request: QueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in rag_service.ask_stream(query=request.query, top_k=request.top_k):
            if chunk == settings.hallucination_marker:
                yield 'event: reset\ndata: {"reason": "hallucination_detected"}\n\n'
            else:
                yield f"event: delta\ndata: {json.dumps({'content': chunk})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=201,
    summary="Ingest document chunks into vector store",
)
async def ingest_documents(
    request: IngestRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> IngestResponse:
    chunks = IngestRequest.to_document_chunks(request.documents)

    await rag_service.ingest_documents(chunks)

    return IngestResponse(status="success", ingested_count=len(chunks))

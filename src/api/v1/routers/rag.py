from fastapi import APIRouter, status

from src.api.v1.models.requests.query.query_request import QueryRequest
from src.api.v1.models.responses.query.query_response import QueryResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute RAG query",
)
async def execute_query(request: QueryRequest) -> QueryResponse:
    return QueryResponse(
        answer=f"Echo: {request.query}",
        sources=["mock_doc_1.pdf"],
        cached=False,
    )

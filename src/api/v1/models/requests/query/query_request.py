from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, max_length=2000, description="User prompt or question")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to retrieve")

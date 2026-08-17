from pydantic import BaseModel, ConfigDict, Field


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="LLM generated answer")
    sources: list[str] = Field(default_factory=list, description="Source documents or chunks")
    cached: bool = Field(default=False, description="Was response served from semantic cache")

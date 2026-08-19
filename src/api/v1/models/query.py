import typing

from pydantic import BaseModel, ConfigDict, Field

from src.core.entities.rag import GeneratedAnswer


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="LLM generated answer")
    sources: list[str] = Field(default_factory=list, description="Source documents or chunks")
    cached: bool = Field(default=False, description="Was response served from semantic cache")

    @classmethod
    def from_generated_answer(cls, generated_answer: GeneratedAnswer) -> typing.Self:
        return cls(
            answer=generated_answer.answer,
            sources=generated_answer.sources,
            cached=generated_answer.cached,
        )


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, max_length=2000, description="User prompt or question")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to retrieve")

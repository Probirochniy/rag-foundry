from pydantic import BaseModel, ConfigDict, Field


class IngestDocumentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="file name or uri")
    content: str = Field(..., min_length=1, description="raw text")


class IngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = Field(default="success", description="Ingestion status message")
    ingested_count: int = Field(
        ..., description="Number of document chunks ingested into the vector store"
    )

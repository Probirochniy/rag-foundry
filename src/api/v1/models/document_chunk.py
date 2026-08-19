from pydantic import BaseModel, ConfigDict, Field


class DocumentChunkDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., min_length=1, description="Text chunk content")
    source_id: str = Field(..., description="Source document name or URL")

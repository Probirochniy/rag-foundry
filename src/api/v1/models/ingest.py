from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.models.document_chunk import DocumentChunkDTO
from src.core.entities.rag import DocumentChunk


class IngestRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: list[DocumentChunkDTO] = Field(
        ..., min_length=1, description="List of document chunks to ingest into the vector store"
    )

    @classmethod
    def to_document_chunks(cls, documents: list[DocumentChunkDTO]) -> list[DocumentChunk]:
        return [
            DocumentChunk(id=doc.id, content=doc.content, metadata={"source_id": doc.source_id})
            for doc in documents
        ]


class IngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = Field(default="success", description="Ingestion status message")
    ingested_count: int = Field(
        ..., description="Number of document chunks ingested into the vector store"
    )

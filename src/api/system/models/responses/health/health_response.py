from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok", description="Service status")
    version: str = Field(default="0.1.0", description="API version")

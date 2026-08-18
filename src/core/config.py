import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Foundry"
    environment: str = "local"

    redis_url: str = os.environ.get("REDIS_URL", "")
    redis_cache_ttl_seconds: int = int(os.environ.get("REDIS_CACHE_TTL_SECONDS", 3600))

    qdrant_url: str = os.environ.get("QDRANT_URL", "")
    qdrant_collection_name: str = os.environ.get("QDRANT_COLLECTION_NAME", "knowledge_base")
    embedding_dimension: int = int(os.environ.get("EMBEDDING_DIMENSION", 1536))


settings = Settings()

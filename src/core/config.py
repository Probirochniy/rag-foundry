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

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-5.6-sol")
    openai_temperature: float = float(os.environ.get("OPENAI_TEMPERATURE", 0.1))

    rag_system_prompt: str = (
        "You are a high-class RAG Foundry technical assistant"
        " that answers questions based on the provided context."
        " If the context is insufficient, you should say that you don't have enough information."
        "\n\nContext:\n{context}"
    )

    hallucination_critic_prompt: str = (
        "You are a hallucination critic for RAG Foundry."
        " Your task is to evaluate the answer provided by the RAG system"
        " based on the context given. If the answer is not supported by the context,"
        " you should indicate that the answer is hallucinated."
        "\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    )


settings = Settings()

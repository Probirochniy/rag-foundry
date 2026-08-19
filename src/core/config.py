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

    hallucination_warning_prompt: str = (
        "\n\nWARNING: The previous answer was flagged for hallucination! "
        "Please provide a more accurate response based on the context."
    )

    hallucination_critic_prompt: str = (
        "You are a strict fact-checker."
        " Assess whether the answer is based EXCLUSIVELY on facts from the context.\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Reply with only one word: YES (if the facts align) or NO (if there is fabrication)."
    )


settings = Settings()

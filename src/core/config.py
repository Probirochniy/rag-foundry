from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RAG Foundry"
    environment: str = "local"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 3600

    # Qdrant Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "knowledge_base"
    qdrant_upsert_batch_size: int = 128

    # Embeddings (TEI)
    tei_url: str = "http://localhost:8080"
    tei_batch_size: int = 32
    embedding_dimension: int = 384

    chunk_size: int = 500
    chunk_overlap: int = 50

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-sol"
    openai_temperature: float = 0.1

    # vLLM
    vllm_base_url: str = ""
    vllm_model: str = "gemma-4-26B"
    vllm_temperature: float = 0.1

    llm_provider: str = "openai"

    # Docker Profiles & Feature Toggles
    compose_profiles: str = ""

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    # Hallucination detection
    max_hallucination_retries: int = 1

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
        "Answer:\n{answer}"
    )

    hallucination_marker: str = "[RESET]"

    @property
    def is_langfuse_enabled(self) -> bool:
        return "langfuse" in self.compose_profiles.lower() and bool(self.langfuse_public_key)


settings = Settings()

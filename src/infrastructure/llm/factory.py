from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.core.config import settings


def create_llm() -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        return ChatOpenAI(
            api_key=SecretStr(settings.openai_api_key),
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            streaming=True,
        )

    elif provider == "vllm":
        return ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=None,
            model=settings.vllm_model,
            temperature=settings.vllm_temperature,
            streaming=True,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")

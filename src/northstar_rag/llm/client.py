"""Create the chat model used everywhere in the project."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from northstar_rag.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """One shared chat model (gpt-4o-mini, temperature 0 by default)."""
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

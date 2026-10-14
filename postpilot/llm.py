"""Provider-agnostic model factories, selected by LLM_PROVIDER / EMBEDDINGS_PROVIDER."""

from __future__ import annotations

from typing import TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from postpilot.config import get_settings

T = TypeVar("T", bound=BaseModel)


def get_chat_model(temperature: float | None = None) -> BaseChatModel:
    s = get_settings()
    kwargs: dict = {}
    if s.llm_provider == "ollama":
        kwargs = {
            "base_url": s.ollama_base_url,
            "num_ctx": s.ollama_num_ctx,
            "reasoning": s.ollama_reasoning,
        }
    return init_chat_model(
        s.llm_model,
        model_provider=s.llm_provider,
        temperature=s.llm_temperature if temperature is None else temperature,
        **kwargs,
    )


def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embeddings_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.embeddings_model, base_url=s.ollama_base_url)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=s.embeddings_model, dimensions=s.embedding_dim)


def ask_structured(schema: type[T], system: str, user: str, temperature: float | None = None) -> T:
    """One LLM call that must return `schema`. Raises so node retry policies can kick in."""
    model = get_chat_model(temperature).with_structured_output(schema)
    result = model.invoke([SystemMessage(system), HumanMessage(user)])
    if result is None:
        raise ValueError(f"model returned no {schema.__name__}")
    if isinstance(result, dict):
        result = schema.model_validate(result)
    return result

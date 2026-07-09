"""LLM provider factory — forked from linki-agent-i, extended with a judge model.

OpenAI-compatible only (OpenAI / DeepSeek). Embeddings are handled separately and
always local (see ``linki.ingestion.indexer``); this module is LLM-only.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from linki.config import Settings

_ENV_KEY = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to .env or export it in your shell.")
    return value


def required_env_for_provider(provider: str) -> str:
    try:
        return _ENV_KEY[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}") from None


def validate_provider_config(provider: str) -> None:
    """Fail fast when the selected provider cannot be configured."""
    load_dotenv()
    _required_env(required_env_for_provider(provider))


def create_model(provider: str = "openai", model: str | None = None, *, temperature: float = 0.0):
    """Create an OpenAI-compatible chat model after loading environment variables."""
    from langchain_openai import ChatOpenAI

    validate_provider_config(provider)

    if provider == "openai":
        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=_required_env("OPENAI_API_KEY"),
            temperature=temperature,
        )
    if provider == "deepseek":
        return ChatOpenAI(
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=_required_env("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=temperature,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def create_main_model(settings: "Settings", *, temperature: float = 0.0) -> "ChatOpenAI":
    return create_model(settings.provider, settings.llm_model, temperature=temperature)


def create_judge_model(settings: "Settings", *, temperature: float = 0.0) -> "ChatOpenAI":
    """Verifier / grader-as-judge model. Falls back to the main model when no
    separate judge model is configured, but a distinct model is recommended."""
    model = settings.judge_model or settings.llm_model
    return create_model(settings.judge_provider, model, temperature=temperature)

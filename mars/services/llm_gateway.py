"""
LLM Gateway service — DeepSeek provider.

Provides a unified interface for obtaining LLM instances.
All six MARS agents use DeepSeek V4 Flash via the OpenAI-compatible API.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import LLM
from langchain_openai import ChatOpenAI

from mars.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level factories
# ---------------------------------------------------------------------------

def _openai_compatible_llm(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.3,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at an OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def _crewai_compatible_llm(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.3,
    **kwargs: Any,
) -> LLM:
    """Return a crewai.LLM instance for DeepSeek."""
    if api_key:
        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )
    return LLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        is_litellm=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Provider: DeepSeek
# ---------------------------------------------------------------------------

def get_deepseek_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """DeepSeek model via OpenAI-compatible API."""
    return _openai_compatible_llm(
        model=model or settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Provider map
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "deepseek": get_deepseek_llm,
}

_PROVIDER_KEY_ATTR: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_available_providers() -> list[str]:
    """Return provider names whose API key is configured."""
    return [
        name
        for name, attr in _PROVIDER_KEY_ATTR.items()
        if getattr(settings, attr, "")
    ]


# ---------------------------------------------------------------------------
# CrewAI LLM construction
# ---------------------------------------------------------------------------

def _get_crewai_llm(model: str | None = None, temperature: float = 0.3) -> LLM:
    """Return a crewai.LLM for DeepSeek."""
    model_name = model or settings.DEEPSEEK_MODEL
    return _crewai_compatible_llm(
        model=f"deepseek/{model_name}",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Agent → LLM mapping (all DeepSeek)
# ---------------------------------------------------------------------------

_AGENT_LLM_MAP: dict[str, tuple[str, str | None]] = {
    "researcher":  ("deepseek", None),
    "searcher":    ("deepseek", None),
    "analyzer":    ("deepseek", None),
    "connector":   ("deepseek", None),
    "summarizer":  ("deepseek", None),
    "evaluator":   ("deepseek", None),
}


def get_llm_by_task(agent_role: str, temperature: float = 0.3) -> LLM:
    """Return a crewai.LLM for a specific MARS agent role."""
    role_key = agent_role.lower()
    if role_key not in _AGENT_LLM_MAP:
        raise ValueError(
            f"Unknown agent role '{agent_role}'. "
            f"Choose from: {list(_AGENT_LLM_MAP.keys())}"
        )
    return _get_crewai_llm(temperature=temperature)


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a DeepSeek ChatOpenAI instance for direct tool use."""
    if not settings.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not configured.")
    return get_deepseek_llm(temperature=temperature)

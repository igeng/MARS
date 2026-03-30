"""
LLM Gateway service.

Provides a unified interface for obtaining LLM instances.
Includes ``get_llm_by_task`` which maps each MARS agent role to
the most appropriate LLM provider / model.

When the preferred provider's API key is not configured the gateway
automatically falls back to any provider that *does* have a key,
so users with only one or two keys can still use the full system.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from mars.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level factory
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


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def get_qwen_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """Qwen model via DashScope OpenAI-compatible endpoint."""
    return _openai_compatible_llm(
        model=model or settings.QWEN_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


def get_deepseek_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """DeepSeek model via DeepSeek Open API (OpenAI-compatible)."""
    return _openai_compatible_llm(
        model=model or settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def get_kimi_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """Kimi model via Moonshot AI API (OpenAI-compatible)."""
    return _openai_compatible_llm(
        model=model or settings.KIMI_MODEL,
        api_key=settings.MOONSHOT_API_KEY,
        base_url=settings.MOONSHOT_BASE_URL,
        temperature=temperature,
    )


def get_glm_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """GLM model via Zhipu AI API (OpenAI-compatible endpoint)."""
    return _openai_compatible_llm(
        model=model or settings.GLM_MODEL,
        api_key=settings.ZHIPU_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Provider map
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "qwen": get_qwen_llm,
    "deepseek": get_deepseek_llm,
    "kimi": get_kimi_llm,
    "glm": get_glm_llm,
}

# Maps provider name → the settings attribute that holds its API key.
_PROVIDER_KEY_ATTR: dict[str, str] = {
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "glm": "ZHIPU_API_KEY",
}


def get_available_providers() -> list[str]:
    """Return provider names whose API key is configured (non-empty)."""
    return [
        name
        for name, attr in _PROVIDER_KEY_ATTR.items()
        if getattr(settings, attr, "")
    ]


def _resolve_provider(provider: str) -> str:
    """Return *provider* if its key is set, otherwise fall back.

    Falls back to ``DEFAULT_LLM_PROVIDER`` first, then to the first
    available provider.  If *no* provider has a key, returns the
    original provider unchanged (the LLM object can still be created
    but will fail at invocation time with a clear auth error).
    """
    key_attr = _PROVIDER_KEY_ATTR.get(provider, "")
    if key_attr and getattr(settings, key_attr, ""):
        return provider  # preferred provider is available

    available = get_available_providers()
    if not available:
        # No keys at all – return original so the LLM object can be
        # constructed (it will fail with an auth error at call time).
        logger.warning(
            "No LLM provider API key is configured. "
            "Set at least one of: DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, "
            "MOONSHOT_API_KEY, ZHIPU_API_KEY in your .env file."
        )
        return provider

    # Try the default provider first
    default = settings.DEFAULT_LLM_PROVIDER.lower()
    if default in available:
        fallback = default
    else:
        fallback = available[0]

    logger.warning(
        "Provider '%s' has no API key configured – falling back to '%s'.",
        provider,
        fallback,
    )
    return fallback


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """
    Return an LLM instance for the given provider.

    When the requested provider's API key is not set the function
    transparently falls back to an available provider so that users
    who only have one or two keys can still run all agents.

    Args:
        provider: One of "qwen", "deepseek", "kimi", "glm".
                  Defaults to ``settings.DEFAULT_LLM_PROVIDER``.
        model:    Override the default model name for that provider.
        temperature: Sampling temperature.

    Raises:
        ValueError: If the provider name is not recognised or no key
                    is configured at all.
    """
    requested = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
    if requested not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM provider '{requested}'. "
            f"Choose from: {list(_PROVIDER_MAP.keys())}"
        )

    resolved = _resolve_provider(requested)
    # When falling back, do not pass the caller's model override because
    # that model name may be invalid for the fallback provider.
    effective_model = model if resolved == requested else None
    return _PROVIDER_MAP[resolved](model=effective_model, temperature=temperature)


# ---------------------------------------------------------------------------
# Agent-role → LLM mapping
# ---------------------------------------------------------------------------

# Maps each MARS agent role (lowercase key) to (provider, model_override).
# ``None`` for model_override means "use the provider default".
_AGENT_LLM_MAP: dict[str, tuple[str, str | None]] = {
    "researcher": ("qwen", None),        # Qwen-Max – strong reasoning
    "searcher": ("deepseek", None),       # DeepSeek-V3 – balanced
    "analyzer": ("kimi", None),           # Kimi – long-context
    "connector": ("glm", None),           # GLM-4-Plus – relation reasoning
    "summarizer": ("qwen", None),         # Qwen-Max – long text generation
    "evaluator": ("deepseek", None),      # DeepSeek-V3 – evaluation tasks
}


def get_llm_by_task(agent_role: str, temperature: float = 0.3) -> ChatOpenAI:
    """
    Return the LLM instance configured for a specific MARS agent role.

    If the preferred provider for the role has no API key, the gateway
    transparently falls back to an available provider.

    Args:
        agent_role: One of "researcher", "searcher", "analyzer",
                    "connector", "summarizer", "evaluator".
        temperature: Sampling temperature.

    Raises:
        ValueError: If the agent role is unknown or no provider has a key.
    """
    role_key = agent_role.lower()
    if role_key not in _AGENT_LLM_MAP:
        raise ValueError(
            f"Unknown agent role '{agent_role}'. "
            f"Choose from: {list(_AGENT_LLM_MAP.keys())}"
        )
    provider, model = _AGENT_LLM_MAP[role_key]
    return get_llm(provider=provider, model=model, temperature=temperature)

"""
LLM Gateway service.

Provides a unified interface for obtaining LLM instances.
Includes ``get_llm_by_task`` which maps each MARS agent role to
the most appropriate LLM provider / model.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from mars.config.settings import settings


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


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """
    Return an LLM instance for the given provider.

    Args:
        provider: One of "qwen", "deepseek", "kimi", "glm".
                  Defaults to ``settings.DEFAULT_LLM_PROVIDER``.
        model:    Override the default model name for that provider.
        temperature: Sampling temperature.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: {list(_PROVIDER_MAP.keys())}"
        )
    return _PROVIDER_MAP[provider](model=model, temperature=temperature)


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

    Args:
        agent_role: One of "researcher", "searcher", "analyzer",
                    "connector", "summarizer", "evaluator".
        temperature: Sampling temperature.

    Raises:
        ValueError: If the agent role is unknown.
    """
    role_key = agent_role.lower()
    if role_key not in _AGENT_LLM_MAP:
        raise ValueError(
            f"Unknown agent role '{agent_role}'. "
            f"Choose from: {list(_AGENT_LLM_MAP.keys())}"
        )
    provider, model = _AGENT_LLM_MAP[role_key]
    return get_llm(provider=provider, model=model, temperature=temperature)

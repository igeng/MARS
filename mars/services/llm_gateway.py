"""
LLM Gateway service.

Provides a unified interface for obtaining LLM instances.
Includes ``get_llm_by_task`` which maps each MARS agent role to
the most appropriate LLM provider / model.

When the preferred provider's API key is not configured the gateway
automatically falls back to any provider that *does* have a key,
so users with only one or two keys can still use the full system.

Two types of LLM objects are returned:

* :func:`get_llm` and the per-provider helpers (``get_qwen_llm`` etc.)
  return a :class:`langchain_openai.ChatOpenAI` instance for use in
  tools that call the LLM directly (e.g. ``KeywordExpanderTool``).

* :func:`get_llm_by_task` returns a :class:`crewai.LLM` instance for
  use in CrewAI ``Agent`` objects.  CrewAI ≥0.80 uses either a native
  provider SDK or LiteLLM internally; both require the model string to
  carry a provider prefix.  All OpenAI-compatible endpoints (DashScope,
  Moonshot/Kimi, Zhipu AI) use the ``openai/<model>`` prefix with a
  custom ``base_url`` — the reliable LiteLLM path for any custom
  OpenAI-compatible API.  Provider-specific prefixes such as
  ``dashscope/`` or ``moonshot/`` can cause the installed provider SDK
  to intercept the call and strip the prefix, resulting in a
  "LLM Provider NOT provided" error.  Passing a raw ``ChatOpenAI`` object
  also fails in the same way.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

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
    """Return a ChatOpenAI instance pointed at an OpenAI-compatible endpoint.

    Used for direct LLM calls inside tools (e.g. ``KeywordExpanderTool``).
    """
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
) -> LLM:
    """Return a crewai.LLM instance pointed at an OpenAI-compatible endpoint.

    Used when assigning an LLM to a CrewAI ``Agent``.  CrewAI ≥0.80 passes
    LLM calls through either a native provider SDK or LiteLLM.  The model
    string must carry the correct provider prefix so that CrewAI/LiteLLM
    routes the call to the right backend.  For all OpenAI-compatible
    endpoints (DashScope, Moonshot/Kimi, Zhipu AI) use ``openai/<model>``
    together with a custom ``base_url``; this is the standard LiteLLM
    pattern for custom OpenAI-compatible APIs.  Using provider-specific
    prefixes such as ``dashscope/<model>`` or ``moonshot/<model>`` can cause
    LiteLLM to strip the prefix before calling the fallback path, resulting
    in a "LLM Provider NOT provided" error.  Only ``deepseek/`` is kept as a
    native LiteLLM provider because DeepSeek is fully supported without a
    custom base_url.

    When *api_key* is falsy (either ``None`` or an empty string ``""``, which
    both occur in test environments where no key is configured) the function
    bypasses native-SDK API-key validation by requesting the LiteLLM path so
    that the LLM object can always be constructed.  It will still raise an auth
    error if an actual API call is made without a valid key.
    """
    # Native provider SDKs validate the API key at construction time.
    # When no key is available yet (api_key is None or ""), fall back to the
    # LiteLLM code path, which defers validation to call time.
    if api_key:
        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
    return LLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        is_litellm=True,
    )


# ---------------------------------------------------------------------------
# Provider helpers  (return ChatOpenAI – for direct tool invocation)
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


def get_kimi_llm(model: str | None = None, temperature: float = 1.0) -> ChatOpenAI:
    """Kimi model via Moonshot AI API (OpenAI-compatible).

    The default ``kimi-k2.5`` model only accepts ``temperature=1``; the
    default value is therefore set to ``1.0`` here.  If you switch to a
    different Kimi model that supports a wider range, pass *temperature*
    explicitly.
    """
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
# Provider maps
# ---------------------------------------------------------------------------

# ChatOpenAI factories – used by get_llm() for direct tool invocation.
_PROVIDER_MAP = {
    "qwen": get_qwen_llm,
    "deepseek": get_deepseek_llm,
    "kimi": get_kimi_llm,
    "glm": get_glm_llm,
}

# crewai.LLM factories – used by get_llm_by_task() for agent construction.
# All OpenAI-compatible providers (DashScope, Moonshot, Zhipu) use the
# "openai" model prefix with a custom base_url.  This is the reliable
# LiteLLM path for any OpenAI-compatible endpoint.  DeepSeek uses the
# "deepseek" prefix because it is a natively supported LiteLLM provider.


class _CrewAIProviderCfg(NamedTuple):
    """Configuration needed to construct a crewai.LLM for a given provider."""

    model_prefix: str             # LiteLLM provider prefix (e.g. "openai", "deepseek")
    model_attr: str               # Settings attribute name holding the default model name
    key_attr: str                 # Settings attribute name holding the API key
    base_url: str                 # Endpoint URL for the provider
    temperature_override: float | None = None  # When set, always use this temperature


_CREWAI_PROVIDER_CFG: dict[str, _CrewAIProviderCfg] = {
    # Use the "openai" prefix for all OpenAI-compatible endpoints so that
    # LiteLLM routes the call through the standard OpenAI SDK with a custom
    # base_url.  Provider-specific prefixes (e.g. "dashscope", "moonshot")
    # are not reliably handled by LiteLLM and cause "LLM Provider NOT
    # provided" errors when the installed provider SDK intercepts the call
    # and strips the prefix before the LiteLLM fallback path runs.
    # kimi-k2.5 only accepts temperature=1; the override enforces this.
    "qwen":     _CrewAIProviderCfg("openai",   "QWEN_MODEL",    "DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "deepseek": _CrewAIProviderCfg("deepseek", "DEEPSEEK_MODEL", "DEEPSEEK_API_KEY",  "https://api.deepseek.com/v1"),
    "kimi":     _CrewAIProviderCfg("openai",   "KIMI_MODEL",     "MOONSHOT_API_KEY",  "https://api.moonshot.cn/v1", 1.0),
    "glm":      _CrewAIProviderCfg("openai",   "GLM_MODEL",      "ZHIPU_API_KEY",     "https://open.bigmodel.cn/api/paas/v4"),
}


def _get_crewai_llm(provider: str, model: str | None = None, temperature: float = 0.3) -> LLM:
    """Return a crewai.LLM for *provider*, optionally overriding the model name."""
    cfg = _CREWAI_PROVIDER_CFG[provider]
    model_name = model or getattr(settings, cfg.model_attr)
    effective_temperature = (
        cfg.temperature_override if cfg.temperature_override is not None else temperature
    )
    return _crewai_compatible_llm(
        model=f"{cfg.model_prefix}/{model_name}",
        api_key=getattr(settings, cfg.key_attr),
        base_url=cfg.base_url,
        temperature=effective_temperature,
    )


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

    # Try the default provider first (but keep glm as last resort)
    default = settings.DEFAULT_LLM_PROVIDER.lower()
    if default in available and default != "glm":
        fallback = default
    else:
        non_glm = [p for p in available if p != "glm"]
        fallback = non_glm[0] if non_glm else "glm"

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
# Preferred providers are qwen and kimi – both work without extra keys.
# deepseek/glm entries fall back automatically when those keys are absent.
_AGENT_LLM_MAP: dict[str, tuple[str, str | None]] = {
    "researcher": ("qwen", None),     # Qwen – strong reasoning
    "searcher": ("kimi", None),       # Kimi – long-context search synthesis
    "analyzer": ("kimi", None),       # Kimi – long-context deep analysis
    "connector": ("qwen", None),      # Qwen – relation reasoning
    "summarizer": ("qwen", None),     # Qwen – long text generation
    "evaluator": ("kimi", None),      # Kimi – evaluation tasks
}


def get_llm_by_task(agent_role: str, temperature: float = 0.3) -> LLM:
    """
    Return a :class:`crewai.LLM` instance configured for a specific MARS agent role.

    Returns a ``crewai.LLM`` object rather than a plain ``ChatOpenAI``.
    CrewAI ≥0.80 routes agent calls through its native provider SDKs or
    LiteLLM.  Both require the model string to include the correct provider
    prefix (e.g. ``dashscope/qwen3.5-flash``) and a ``base_url`` pointing
    at the custom endpoint.  Using ``crewai.LLM`` ensures these are set
    correctly for every provider.

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
    resolved = _resolve_provider(provider)
    # When falling back, do not pass the caller's model override because
    # that model name may be invalid for the fallback provider.
    effective_model = model if resolved == provider else None
    return _get_crewai_llm(provider=resolved, model=effective_model, temperature=temperature)

"""
LLM factory - creates LangChain-compatible LLM instances
for each supported provider (Qwen, DeepSeek, Kimi, GLM).

CrewAI uses LangChain under the hood, so we expose ChatOpenAI /
provider-specific wrappers that implement the ChatModel interface.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from mars.config import settings


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


def get_qwen_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """Qwen model via DashScope OpenAI-compatible endpoint."""
    model = model or settings.QWEN_MODEL
    return _openai_compatible_llm(
        model=model,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


def get_deepseek_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """DeepSeek model via DeepSeek Open API (OpenAI-compatible)."""
    model = model or settings.DEEPSEEK_MODEL
    return _openai_compatible_llm(
        model=model,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def get_kimi_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """Kimi model via Moonshot AI API (OpenAI-compatible)."""
    model = model or settings.KIMI_MODEL
    return _openai_compatible_llm(
        model=model,
        api_key=settings.MOONSHOT_API_KEY,
        base_url=settings.MOONSHOT_BASE_URL,
        temperature=temperature,
    )


def get_glm_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """GLM model via Zhipu AI API (OpenAI-compatible endpoint)."""
    model = model or settings.GLM_MODEL
    return _openai_compatible_llm(
        model=model,
        api_key=settings.ZHIPU_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=temperature,
    )


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
    """
    provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: {list(_PROVIDER_MAP.keys())}"
        )
    return _PROVIDER_MAP[provider](model=model, temperature=temperature)

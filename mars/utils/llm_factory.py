"""
LLM factory - backwards-compatible re-exports from :mod:`mars.services.llm_gateway`.

All new code should prefer importing from ``mars.services.llm_gateway``
directly.  This module exists so that existing call-sites (agents, tools,
tests) continue to work without changes.
"""

from mars.services.llm_gateway import (  # noqa: F401 – re-export
    get_available_providers,
    get_deepseek_llm,
    get_glm_llm,
    get_kimi_llm,
    get_llm,
    get_llm_by_task,
    get_qwen_llm,
)

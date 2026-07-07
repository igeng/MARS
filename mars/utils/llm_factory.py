"""
LLM factory — backwards-compatible re-exports from :mod:`mars.services.llm_gateway`.

.. deprecated:: 0.2.0
    Import from ``mars.services.llm_gateway`` directly.
"""

from __future__ import annotations

from mars.services.llm_gateway import (  # noqa: F401
    get_available_providers,
    get_deepseek_llm,
    get_llm,
    get_llm_by_task,
)

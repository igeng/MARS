"""
LLM factory — **DEPRECATED** backwards-compatibility re-exports.

.. deprecated:: 0.2.0
    This module is a thin compatibility shim over
    :mod:`mars.services.llm_gateway`.  All new code should import from
    ``mars.services.llm_gateway`` directly.  This module will be removed
    in version 0.3.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "mars.utils.llm_factory is deprecated; "
    "use mars.services.llm_gateway instead. "
    "This module will be removed in v0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mars.services.llm_gateway import (  # noqa: F401, E402 – re-export
    get_available_providers,
    get_deepseek_llm,
    get_glm_llm,
    get_kimi_llm,
    get_llm,
    get_llm_by_task,
    get_qwen_llm,
)

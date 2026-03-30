"""
Centralised logging configuration for MARS.

Call ``setup_logging()`` once at application startup (CLI ``main()`` or
FastAPI ``lifespan``) and then use the standard library ``logging`` module
throughout the codebase::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("hello %s", "world")
"""

from __future__ import annotations

import logging
import sys

from mars.config.settings import settings

_CONFIGURED = False


def setup_logging() -> None:
    """Configure the root logger based on ``settings.LOG_LEVEL``."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Silence overly chatty libraries at INFO level
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True

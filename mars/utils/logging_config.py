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

import datetime
import logging
import sys
from typing import IO, Optional

from mars.config.settings import settings

_CONFIGURED = False


def setup_logging(log_stream: Optional[IO[str]] = None) -> None:
    """Configure the root logger based on ``settings.LOG_LEVEL``.

    Args:
        log_stream: Optional writable text stream to direct file log output to
                    (e.g. the run-specific ``run.log`` file handle).  When
                    omitted a timestamped ``.log`` file is created in
                    ``settings.OUTPUT_DIR`` as before.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # File handler — write to the supplied stream or a new timestamped file
    if log_stream is not None:
        file_handler = logging.StreamHandler(log_stream)
    else:
        log_dir = settings.OUTPUT_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"mars_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence overly chatty libraries at INFO level
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True

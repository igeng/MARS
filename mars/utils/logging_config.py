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


# ---------------------------------------------------------------------------
# Log filter that injects a per-run identifier into every log record.
# ---------------------------------------------------------------------------

class _RunIdFilter(logging.Filter):
    """Logging filter that attaches ``run_id`` to every record.

    The filter is a no-op when ``run_id`` is ``None`` (e.g. before a run
    starts or in non-run contexts such as ``mars check``).
    """

    run_id: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id or "-"  # type: ignore[attr-defined]
        return True


# Singleton module-level filter so that callers can set the id without
# digging into the logging handler chain.
_run_id_filter = _RunIdFilter()

# Sentinel to prevent double-configuration of the root logger.
_CONFIGURED = False


def set_run_id(run_id: str | None) -> None:
    """Set the run identifier to be injected into every subsequent log line.

    Pass ``None`` to clear (e.g. at the end of a run).
    """
    _run_id_filter.run_id = run_id


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

    from mars.config.settings import settings

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addFilter(_run_id_filter)

    # Console handler (stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # File handler — write to the supplied stream or a new timestamped file
    if log_stream is not None:
        file_handler = logging.StreamHandler(log_stream)
    else:
        log_dir = settings.OUTPUT_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"mars_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root.addHandler(handler)
    root.addHandler(file_handler)

    # Silence overly chatty libraries at INFO level
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True

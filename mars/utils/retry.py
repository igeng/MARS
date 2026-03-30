"""
Retry helper with exponential back-off for external HTTP calls.

Usage::

    from mars.utils.retry import retry_on_network_error

    @retry_on_network_error(max_retries=3)
    def _call_api():
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_on_network_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """Decorator that retries a function on ``requests.RequestException``.

    The delay between retries grows exponentially:
    ``base_delay * backoff_factor ** attempt``.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs – %s",
                            attempt + 1,
                            max_retries,
                            func.__qualname__,
                            delay,
                            exc,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator

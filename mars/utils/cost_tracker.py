"""
Token cost tracking for MARS runs.

Records per-agent token consumption and estimates API cost using
LiteLLM's callback hooks.  Write a ``cost_report.json`` alongside
each run's output files.

Usage (automatic — no code changes needed in agents/crews):

    from mars.utils.cost_tracker import CostTracker

    tracker = CostTracker.start()          # enables LiteLLM hooks
    # ... run MARS crew ...
    tracker.save("output/run_xxx/")
    tracker.stop()                         # disables hooks
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approximate pricing per 1M tokens (input / output), USD, updated 2025-2026.
# These are estimates — actual costs depend on provider pricing tiers.
# ---------------------------------------------------------------------------

_PRICE_PER_1M: Dict[str, tuple[float, float]] = {
    # (input_price, output_price) in USD
    "gpt-4o":           (2.50,  10.00),
    "gpt-4o-mini":      (0.15,   0.60),
    "gpt-4-turbo":      (10.00,  30.00),
    "claude-sonnet-4":  (3.00,   15.00),
    "claude-opus-4":    (15.00,  75.00),
    "claude-haiku-3.5": (0.80,   4.00),
    "qwen3.5-flash":    (0.00,   0.00),    # free tier
    "qwen-max":         (0.60,   2.00),
    "qwen-plus":        (0.30,   1.00),
    "deepseek-chat":    (0.14,   0.28),
    "deepseek-coder":   (0.14,   0.28),
    "kimi-k2.5":        (0.60,   1.40),
    "moonshot-v1-8k":   (0.60,   1.40),
    "glm-4.7-flash":    (0.00,   0.00),    # free tier
    "glm-4-plus":       (1.00,   3.00),
    "glm-4-air":        (0.50,   1.50),
    "minimax-m2.1":     (0.30,   1.00),
}

_DEFAULT_PRICE = (1.00, 3.00)  # fallback for unknown models


def _lookup_price(model: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M) for a model string."""
    model_lower = model.lower()
    for key, price in _PRICE_PER_1M.items():
        if key in model_lower:
            return price
    return _DEFAULT_PRICE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AgentCost:
    """Per-agent token + cost breakdown."""
    agent: str = ""
    model: str = ""
    provider: str = ""
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class RunCost:
    """Per-run cost summary."""
    run_id: str = ""
    started_at: float = 0.0
    agents: Dict[str, AgentCost] = field(default_factory=dict)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0

    def record(self, agent: str, model: str, provider: str,
               tokens_in: int, tokens_out: int) -> None:
        """Record a single LLM call."""
        key = agent
        if key not in self.agents:
            self.agents[key] = AgentCost(agent=agent)
        ac = self.agents[key]
        ac.calls += 1
        ac.tokens_in += tokens_in
        ac.tokens_out += tokens_out
        if not ac.model:
            ac.model = model
            ac.provider = provider

        in_price, out_price = _lookup_price(model)
        ac.cost_usd += (tokens_in / 1_000_000) * in_price + (tokens_out / 1_000_000) * out_price

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd += ac.cost_usd

    def to_dict(self) -> Dict[str, Any]:
        elapsed = time.time() - self.started_at if self.started_at else 0
        return {
            "run_id": self.run_id,
            "elapsed_seconds": round(elapsed, 1),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "agents": {
                name: {
                    "calls": ac.calls,
                    "model": ac.model,
                    "provider": ac.provider,
                    "tokens_in": ac.tokens_in,
                    "tokens_out": ac.tokens_out,
                    "cost_usd": round(ac.cost_usd, 6),
                }
                for name, ac in self.agents.items()
            },
        }


# ---------------------------------------------------------------------------
# Singleton tracker
# ---------------------------------------------------------------------------

_tracker: Optional["CostTracker"] = None
_lock = threading.Lock()


class CostTracker:
    """Global singleton for tracking LLM costs across a MARS run."""

    def __init__(self, run_id: str = ""):
        self._run = RunCost(
            run_id=run_id or time.strftime("%Y%m%d_%H%M%S"),
            started_at=time.time(),
        )
        self._active = False
        self._original_success_handler = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start(cls, run_id: str = "") -> "CostTracker":
        """Create (or reset) the global tracker and enable hooks."""
        global _tracker
        with _lock:
            _tracker = cls(run_id)
            _tracker._install_hooks()
            _tracker._active = True
        logger.info("Cost tracker started for run '%s'.", _tracker._run.run_id)
        return _tracker

    @classmethod
    def current(cls) -> Optional["CostTracker"]:
        with _lock:
            return _tracker

    def stop(self) -> None:
        self._active = False
        self._remove_hooks()
        logger.info("Cost tracker stopped.")

    def save(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / "cost_report.json"
        path.write_text(
            json.dumps(self._run.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Cost report saved to %s (total: $%.4f)", path, self._run.total_cost_usd)
        return path

    @property
    def run(self) -> RunCost:
        return self._run

    # ------------------------------------------------------------------
    # LiteLLM hook
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Patch into LiteLLM's success callback to intercept token usage."""
        try:
            import litellm
            self._original_success_handler = getattr(litellm, "success_callback", [])
            target = getattr(litellm, "success_callback", None)
            if isinstance(target, list):
                target.append(self._on_llm_success)
            else:
                litellm.success_callback = [self._on_llm_success]
        except ImportError:
            pass

    def _remove_hooks(self) -> None:
        try:
            import litellm
            cb = getattr(litellm, "success_callback", [])
            if isinstance(cb, list) and self._on_llm_success in cb:
                cb.remove(self._on_llm_success)
        except ImportError:
            pass

    def _on_llm_success(self, kwargs, response, start_time, end_time) -> None:
        """LiteLLM callback — records token usage from each LLM call."""
        if not self._active:
            return
        try:
            model = kwargs.get("model", "unknown")
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
            if tokens_in == 0 and tokens_out == 0:
                return

            # Best-effort agent identification from custom metadata
            agent = kwargs.get("litellm_metadata", {})
            if isinstance(agent, dict):
                agent_name = agent.get("agent_name", "unknown")
            else:
                agent_name = "unknown"

            provider = model.split("/")[0] if "/" in model else "unknown"

            self._run.record(
                agent=agent_name,
                model=model,
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        except Exception:
            pass  # never let cost tracking crash the real work


def get_cost_report() -> Optional[Dict[str, Any]]:
    """Get the current run's cost summary, if tracking is active."""
    tracker = CostTracker.current()
    if tracker is None:
        return None
    return tracker.run.to_dict()


def start_cost_tracking(run_id: str = "") -> CostTracker:
    """Convenience: start global cost tracking."""
    return CostTracker.start(run_id)

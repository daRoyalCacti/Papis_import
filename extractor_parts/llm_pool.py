"""Simple round-robin model pool with rate-limit cooling for LLM API calls."""
from __future__ import annotations

import time
from typing import Optional


class ModelPool:
    """Rotate across a list of models, cooling a model when it hits a rate limit.

    Models are tried in preference order (index 0 = most preferred).  A model
    is marked "cooling" when ``mark_rate_limited`` is called; it becomes
    available again after the specified wait has elapsed.
    """

    def __init__(self, models: tuple[str, ...], switch_threshold_s: float) -> None:
        self.models = models
        self.switch_threshold_s = switch_threshold_s
        # available_at maps model → monotonic time when it's next usable
        self._available_at: dict[str, float] = {m: 0.0 for m in models}
        self._soft_failures: int = 0

    # ------------------------------------------------------------------
    # Per-call state
    # ------------------------------------------------------------------

    def reset_call_state(self) -> None:
        """Reset counters that track failures within a single LLM call."""
        self._soft_failures = 0

    def soft_failures_this_call(self) -> int:
        return self._soft_failures

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def pick(self) -> Optional[str]:
        """Return the first available model in preference order, or None."""
        now = time.monotonic()
        for m in self.models:
            if self._available_at.get(m, 0.0) <= now:
                return m
        return None

    def all_cooling(self) -> Optional[tuple[str, float]]:
        """Return (soonest_model, wait_s) when every model is cooling, else None."""
        now = time.monotonic()
        best: Optional[tuple[str, float]] = None
        for m in self.models:
            avail = self._available_at.get(m, 0.0)
            if avail <= now:
                return None  # at least one model is ready
            wait = avail - now
            if best is None or wait < best[1]:
                best = (m, wait)
        return best

    # ------------------------------------------------------------------
    # Failure recording
    # ------------------------------------------------------------------

    def mark_rate_limited(self, model: str, wait_s: float, reason: str = "") -> None:
        """Mark *model* as cooling for *wait_s* seconds."""
        if model in self._available_at:
            self._available_at[model] = time.monotonic() + wait_s

    def mark_soft_failure(self, model: str, reason: str = "") -> None:
        """Record a non-rate-limit failure (bad response shape, etc.)."""
        self._soft_failures += 1

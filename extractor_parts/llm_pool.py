"""Per-model rate-limit state and selection policy for text LLM cycling."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _ModelState:
    name: str
    cooldown_until: float = 0.0
    last_reason: str = ""
    consecutive_soft_failures: int = 0
    total_attempts: int = 0


class ModelPool:
    """Tracks per-model cooldown state and selects the best available model.

    Created once per run and held by TextLlmExtractor so cooldowns persist
    across PDFs (daily-token 429s from model A stay marked until the reset
    time passes, at which point pick() naturally returns to model A again).
    """

    def __init__(self, models: tuple[str, ...], switch_threshold_s: float) -> None:
        self._models = models
        self.switch_threshold_s = switch_threshold_s
        self._state: dict[str, _ModelState] = {m: _ModelState(name=m) for m in models}
        # Per-PDF-call tracking (reset at the top of each candidate() call)
        self._call_soft_failures: int = 0
        self._call_attempted_models: set[str] = set()

    def pick(self) -> str | None:
        """Return the highest-preference model not currently cooling down, or None."""
        now = time.monotonic()
        for m in self._models:
            if m not in self._call_attempted_models and self._state[m].cooldown_until <= now:
                return m
        return None

    def mark_rate_limited(self, model: str, wait_s: float, reason: str) -> None:
        """Mark a model as cooling due to a 429 with a known reset time."""
        state = self._state.get(model)
        if state is None:
            return
        state.cooldown_until = time.monotonic() + max(wait_s, 0.0)
        state.last_reason = reason
        state.total_attempts += 1

    def mark_soft_failure(self, model: str, reason: str) -> None:
        """Exclude model from the rest of this PDF's attempts due to bad/empty response.

        Soft failures do NOT set a persistent cooldown — they are per-PDF-call only.
        The exclusion is cleared by reset_call_state() at the start of the next PDF.
        This prevents a single unreadable PDF from degrading quality for subsequent PDFs.
        """
        state = self._state.get(model)
        if state is None:
            return
        state.last_reason = reason
        state.consecutive_soft_failures += 1
        state.total_attempts += 1
        self._call_soft_failures += 1
        self._call_attempted_models.add(model)

    def mark_success(self, model: str) -> None:
        """Record a successful response; resets the soft-failure counter."""
        state = self._state.get(model)
        if state is None:
            return
        state.consecutive_soft_failures = 0
        state.last_reason = ""
        state.total_attempts += 1

    def all_cooling(self) -> tuple[str, float] | None:
        """Return (model_name, seconds_until_ready) for the soonest-recovering model.

        Returns None only if there are no models at all.
        """
        now = time.monotonic()
        soonest: tuple[str, float] | None = None
        for m, s in self._state.items():
            wait = s.cooldown_until - now
            if wait > 0 and (soonest is None or wait < soonest[1]):
                soonest = (m, wait)
        return soonest

    def soft_failures_this_call(self) -> int:
        """Count of soft failures (empty/parse errors) accumulated for the current PDF."""
        return self._call_soft_failures

    def reset_call_state(self) -> None:
        """Clear per-PDF-call counters. Call at the start of each candidate() invocation."""
        self._call_soft_failures = 0
        self._call_attempted_models = set()

    def snapshot(self) -> dict[str, dict]:
        """Return a compact state dict for debug logging."""
        now = time.monotonic()
        return {
            m: {
                "cooldown_remaining_s": round(max(0.0, s.cooldown_until - now), 1),
                "last_reason": s.last_reason,
                "soft_failures": s.consecutive_soft_failures,
            }
            for m, s in self._state.items()
            if s.cooldown_until > now or s.last_reason
        }

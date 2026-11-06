"""Separate generation and answer-delivery deadlines within a shared time budget."""

from __future__ import annotations

import time

MIN_MAX_NEW = 300  # every problem is guaranteed a floor of analysis tokens
MIN_SCALE = 0.25


class Budget:
    """Tracks elapsed time against an internal deadline (85% of the hard limit)."""

    def __init__(self, limit_s: float, *, headroom: float = 0.85) -> None:
        self.t0 = time.time()
        self.deadline = self.t0 + limit_s * headroom
        self._batch_seconds: list[float] = []

    def elapsed(self) -> float:
        """Seconds since solver start."""
        return time.time() - self.t0

    def remaining(self) -> float:
        """Seconds until the internal deadline (may be negative)."""
        return self.deadline - time.time()

    def exhausted(self) -> bool:
        """Return True once the internal deadline has passed."""
        return time.time() > self.deadline

    def record_batch(self, seconds: float) -> None:
        """Record one generation batch's wall time (drives the estimates below)."""
        self._batch_seconds.append(seconds)

    @property
    def avg_batch_s(self) -> float:
        """Mean observed batch time; 0.0 before any measurement."""
        if not self._batch_seconds:
            return 0.0
        return sum(self._batch_seconds) / len(self._batch_seconds)

    def max_new_for(self, batches_left: int, base_max_new: int) -> int:
        """Scale the per-batch token cap to (time remaining / work remaining).

        Same policy as v1: unscaled until throughput is measured, then clamped
        to [MIN_SCALE, 1.0] of the base cap with an absolute floor.
        """
        if not self._batch_seconds:
            return base_max_new
        allowed_s = self.remaining() / max(1, batches_left)
        scale = max(MIN_SCALE, min(1.0, allowed_s / max(self.avg_batch_s, 1e-6)))
        return max(MIN_MAX_NEW, int(base_max_new * scale))

    def affords(self, est_batches: int, *, safety: float = 1.2) -> bool:
        """Return True when optional work (extra votes, refine) fits the remaining time."""
        if self.exhausted():
            return False
        if not self._batch_seconds:
            return True  # nothing measured yet: let the first pass establish rates
        return self.remaining() > est_batches * self.avg_batch_s * safety

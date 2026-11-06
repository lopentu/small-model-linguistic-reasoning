"""Separate generation and answer-delivery deadlines within a shared time budget."""

from __future__ import annotations

import time

MIN_MAX_NEW = 300  # every problem is guaranteed a floor of analysis tokens
MIN_SCALE = 0.25
DELIVERY_HEADROOM = 0.97  # delivery may use the gap between 85% and here; 3% pays the final flush


class Budget:
    """Tracks elapsed time against two deadlines: solve (85%) and delivery (97%)."""

    def __init__(self, limit_s: float, *, headroom: float = 0.85) -> None:
        self.t0 = time.time()
        self.deadline = self.t0 + limit_s * headroom
        self.delivery_deadline = self.t0 + limit_s * max(headroom, DELIVERY_HEADROOM)
        self._batch_seconds: list[float] = []
        import os  # lazy: tiny, avoids a module-level dep for one line

        os.environ["SOLVER_DEADLINE_TS"] = str(self.delivery_deadline)
        os.environ["SOLVER_SOFT_DEADLINE_TS"] = str(self.deadline)

    def elapsed(self) -> float:
        """Seconds since solver start."""
        return time.time() - self.t0

    def remaining(self) -> float:
        """Seconds until the SOFT deadline (may be negative once delivery has started).

        Deliberately soft-only: max_new_for() and affords() budget *new analysis*, and
        those must see zero once solving is over. Anything reasoning about the delivery
        window wants delivery_remaining() instead."""
        return self.deadline - time.time()

    def delivery_remaining(self) -> float:
        """Seconds until the DELIVERY deadline (may be negative)."""
        return self.delivery_deadline - time.time()

    def exhausted(self) -> bool:
        """Return True once the internal (soft) deadline has passed.

        Gates every pass that PRODUCES work: solve, validate, refine, extra votes.
        """
        return time.time() > self.deadline

    def delivery_exhausted(self) -> bool:
        """Return True once even the delivery deadline has passed (near the hard kill).

        Gates the finalize pass only. Between `exhausted()` and this, the solver stops
        generating new analysis but still converts analysis it already paid for into
        answer lines — the difference between a row scoring 0 and scoring normally.
        """
        return time.time() > self.delivery_deadline

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

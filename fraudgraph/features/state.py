"""Online feature state (D3, D4).

State is a set of per-key histories of step buckets: `[step, count, amount_sum,
amount_sum_sq]`. Keys are namespaced (`o:<account>` for outgoing activity of an origin,
later `d:<account>` for a destination). Writes go straight into the bucket of their
own step; reads return only buckets with `step < before_step`, so events in the same
step never see each other and no "step committed" event is needed.

Histories older than `horizon_steps` are dropped (the Redis store uses the same TTL),
so memory stays bounded by recent activity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

# One step bucket: [step, count, amount_sum, amount_sum_sq]. A list (not a tuple) so the
# in-memory store can update the newest bucket in place.
Bucket = list[float]


class OutOfOrderError(ValueError):
    """An event arrived for a step older than one already applied."""


class StateStore(Protocol):
    @property
    def latest_step(self) -> int | None: ...

    def history(self, key: str, before_step: int) -> list[Bucket]:
        """Buckets of `key` with `before_step - horizon <= step < before_step`, oldest first."""
        ...

    def add(self, key: str, step: int, amount: float) -> None:
        """Record one event of `amount` for `key` at `step`."""
        ...


class InMemoryStore:
    """Dict-backed store for backfill and tests."""

    def __init__(self, horizon_steps: int) -> None:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.horizon_steps = horizon_steps
        self._buckets: dict[str, list[Bucket]] = {}
        # Keys whose newest bucket is at a given step: candidates for eviction later.
        self._last_touched: defaultdict[int, list[str]] = defaultdict(list)
        self._latest: int | None = None
        self._evicted_through = 0

    @property
    def latest_step(self) -> int | None:
        return self._latest

    def __len__(self) -> int:
        return len(self._buckets)

    def history(self, key: str, before_step: int) -> list[Bucket]:
        buckets = self._buckets.get(key)
        if not buckets:
            return []
        oldest = before_step - self.horizon_steps
        return [b for b in buckets if oldest <= b[0] < before_step]

    def add(self, key: str, step: int, amount: float) -> None:
        if self._latest is not None and step < self._latest:
            raise OutOfOrderError(f"step {step} arrived after step {self._latest}")
        if self._latest is None or step > self._latest:
            self._latest = step
            self._evict(step)

        buckets = self._buckets.setdefault(key, [])
        if buckets and buckets[-1][0] == step:
            bucket = buckets[-1]
            bucket[1] += 1
            bucket[2] += amount
            bucket[3] += amount * amount
        else:
            buckets.append([float(step), 1.0, amount, amount * amount])
            oldest = step - self.horizon_steps
            while buckets[0][0] < oldest:
                buckets.pop(0)
        self._last_touched[step].append(key)

    def _evict(self, now: int) -> None:
        """Drop keys whose newest bucket is older than the horizon."""
        cutoff = now - self.horizon_steps  # buckets at step < cutoff are never read again
        for step in range(self._evicted_through, cutoff):
            for key in self._last_touched.pop(step, ()):
                buckets = self._buckets.get(key)
                if buckets and buckets[-1][0] == step:
                    del self._buckets[key]
        self._evicted_through = max(self._evicted_through, cutoff)

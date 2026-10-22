"""Online feature state (D3, D4).

State is a set of per-key histories of step buckets. Keys are namespaced: `o:<acct>`
(outgoing activity of an account; members = destinations it paid), `d:<acct>` (incoming
activity; members = senders), `f:<acct>` (confirmed fraud labels on incoming payments,
recorded when the label becomes known). Writes go straight into the bucket of their own
step; reads return only buckets with `step < before_step`, so events in the same step
never see each other and no "step committed" event is needed.

Histories older than `horizon_steps` are dropped (the Redis store uses the same TTL),
so memory stays bounded by recent activity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

# One write of an event: (key, step, amount, counterparty or None).
Write = tuple[str, int, float, str | None]


class Bucket:
    """One key's activity in one step. `members` (optional) lists counterparties."""

    __slots__ = ("members", "n", "step", "total", "total_sq")

    def __init__(self, step: int, amount: float, member: str | None) -> None:
        self.step = step
        self.n = 1
        self.total = amount
        self.total_sq = amount * amount
        self.members: list[str] | None = [member] if member is not None else None

    def add(self, amount: float, member: str | None) -> None:
        self.n += 1
        self.total += amount
        self.total_sq += amount * amount
        if member is not None:
            if self.members is None:
                self.members = [member]
            else:
                self.members.append(member)


class OutOfOrderError(ValueError):
    """An event arrived for a step older than one already applied."""


class StateStore(Protocol):
    @property
    def latest_step(self) -> int | None: ...

    def history(self, key: str, before_step: int) -> list[Bucket]:
        """Buckets of `key` with `before_step - horizon <= step < before_step`, oldest first."""
        ...

    def add(self, key: str, step: int, amount: float, member: str | None = None) -> None:
        """Record one event of `amount` (with an optional counterparty) for `key`."""
        ...

    def apply(self, marker: str | None, writes: Sequence[Write]) -> bool:
        """Apply `writes` as one unit. With a `marker` (e.g. the txn_id), a unit already
        applied under the same marker is skipped and False is returned (retries)."""
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
        return [b for b in buckets if oldest <= b.step < before_step]

    def add(self, key: str, step: int, amount: float, member: str | None = None) -> None:
        if self._latest is not None and step < self._latest:
            raise OutOfOrderError(f"step {step} arrived after step {self._latest}")
        if self._latest is None or step > self._latest:
            self._latest = step
            self._evict(step)

        buckets = self._buckets.setdefault(key, [])
        if buckets and buckets[-1].step == step:
            buckets[-1].add(amount, member)
        else:
            buckets.append(Bucket(step, amount, member))
            oldest = step - self.horizon_steps
            while buckets[0].step < oldest:
                buckets.pop(0)
        self._last_touched[step].append(key)

    def apply(self, marker: str | None, writes: Sequence[Write]) -> bool:
        # Backfill replays each event once, so markers are not kept (memory, D4).
        if writes and self._latest is not None and writes[0][1] < self._latest:
            raise OutOfOrderError(f"step {writes[0][1]} arrived after step {self._latest}")
        for key, step, amount, member in writes:
            self.add(key, step, amount, member)
        return True

    def _evict(self, now: int) -> None:
        """Drop keys whose newest bucket is older than the horizon."""
        cutoff = now - self.horizon_steps  # buckets at step < cutoff are never read again
        for step in range(self._evicted_through, cutoff):
            for key in self._last_touched.pop(step, ()):
                buckets = self._buckets.get(key)
                if buckets and buckets[-1].step == step:
                    del self._buckets[key]
        self._evicted_through = max(self._evicted_through, cutoff)

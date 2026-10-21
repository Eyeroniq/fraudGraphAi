"""Seeded synthetic data in the raw PaySim format, for tests and CI (D10).

The generator mirrors the structure found in the data audit (docs/data_audit.md):
- origins are single-use accounts; destinations are reused from a pool,
- legit volume follows a daily cycle, fraud is uniform over the day,
- the last days are sparse for legit traffic but not for fraud (late-period shift),
- "drain" fraud empties the sender (`amount == oldbalance_orig`, the PaySim artifact)
  and mostly goes to never-seen destinations,
- planted **mule rings**: brand-new destinations that each receive transfers from many
  new senders within a short burst. Ring transfers do *not* drain the sender, so only
  graph features can find them.

The same seed always produces the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from fraudgraph.data.schema import RAW_DTYPES

# Relative legit volume per hour of day (step % 24), shaped like PaySim's.
HOUR_WEIGHTS = np.array(
    [72, 27, 9, 2, 1, 2, 3, 9, 27, 284, 426, 446, 483, 468, 440, 417, 442, 440, 581, 648, 554,
     248, 195, 141],
    dtype=float,
)  # fmt: skip

LEGIT_TYPES = ("CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT")
LEGIT_TYPE_WEIGHTS = np.array([0.352, 0.338, 0.220, 0.084, 0.006])
# Median amount per legit type (lognormal, sigma 1.0).
LEGIT_MEDIAN_AMOUNT = {
    "CASH_OUT": 150_000.0,
    "PAYMENT": 9_000.0,
    "CASH_IN": 140_000.0,
    "TRANSFER": 480_000.0,
    "DEBIT": 5_000.0,
}
FRAUD_AMOUNT_CAP = 10_000_000.0


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 7
    days: int = 10
    legit_per_day: int = 2_450
    sparse_days: int = 2  # the last N days carry only `sparse_factor` of legit volume
    sparse_factor: float = 0.05
    drain_fraud_per_day: int = 10
    rings: int = 3
    ring_destinations: int = 3  # k new destinations per ring
    ring_senders: int = 12  # m distinct new senders into each ring destination
    ring_burst_steps: int = 6
    destination_pool: int = 3_000
    merchant_pool: int = 400
    legit_new_destination_share: float = 0.15
    fraud_new_destination_share: float = 0.63


@dataclass(frozen=True)
class SynthResult:
    frame: pl.DataFrame  # raw PaySim columns, sorted by step
    ring_destinations: tuple[str, ...]
    ring_bursts: dict[str, tuple[int, int]] = field(default_factory=dict)  # dest -> steps


class _Ids:
    """Account id factory: `C`/`M` + digits, never reused unless drawn from a pool."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._next = {"C": int(rng.integers(1_000_000_000, 1_100_000_000)), "M": 1_200_000_000}

    def new(self, prefix: str = "C") -> str:
        self._next[prefix] += 1
        return f"{prefix}{self._next[prefix]}"


def generate(config: SynthConfig | None = None) -> SynthResult:
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    ids = _Ids(rng)

    pool = [ids.new() for _ in range(cfg.destination_pool)]
    # Mildly skewed reuse: a few busy destinations, a long tail of quiet ones.
    pool_weights = 1.0 / (np.arange(cfg.destination_pool) + 50.0)
    pool_weights /= pool_weights.sum()
    merchants = [ids.new("M") for _ in range(cfg.merchant_pool)]

    rows: list[tuple[object, ...]] = []

    def destination(new_share: float) -> str:
        if rng.random() < new_share:
            return ids.new()
        return pool[int(rng.choice(cfg.destination_pool, p=pool_weights))]

    # ---- legit traffic
    for day in range(cfg.days):
        sparse = day >= cfg.days - cfg.sparse_days
        n = round(cfg.legit_per_day * (cfg.sparse_factor if sparse else 1.0))
        hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
        types = rng.choice(len(LEGIT_TYPES), size=n, p=LEGIT_TYPE_WEIGHTS)
        for hour, type_index in zip(hours, types, strict=True):
            type_ = LEGIT_TYPES[type_index]
            amount = _amount(rng, LEGIT_MEDIAN_AMOUNT[type_])
            old_orig = 0.0 if rng.random() < 0.47 else _amount(rng, 120_000.0)
            if old_orig == amount:  # legit never drains exactly (the PaySim artifact)
                old_orig = round(old_orig + 0.01, 2)
            if type_ == "PAYMENT":
                dest = merchants[int(rng.integers(cfg.merchant_pool))]
            else:
                dest = destination(cfg.legit_new_destination_share)
            rows.append(
                _row(rng, _step(day, int(hour)), type_, amount, ids.new(), old_orig, dest, False)
            )

    # ---- drain fraud: uniform over the day, mostly to never-seen destinations
    for day in range(cfg.days):
        for _ in range(cfg.drain_fraud_per_day):
            step = _step(day, int(rng.integers(24)))
            type_ = "TRANSFER" if rng.random() < 0.5 else "CASH_OUT"
            balance = _amount(rng, 400_000.0)
            amount = min(balance, FRAUD_AMOUNT_CAP)
            dest = destination(cfg.fraud_new_destination_share)
            rows.append(_row(rng, step, type_, amount, ids.new(), balance, dest, True))

    # ---- mule rings: k new destinations, each hit by m new senders within a burst
    ring_dests: list[str] = []
    bursts: dict[str, tuple[int, int]] = {}
    dense_last_step = (cfg.days - cfg.sparse_days) * 24
    for _ in range(cfg.rings):
        start = int(rng.integers(25, dense_last_step - cfg.ring_burst_steps))
        end = start + cfg.ring_burst_steps - 1
        for _ in range(cfg.ring_destinations):
            dest = ids.new()
            ring_dests.append(dest)
            bursts[dest] = (start, end)
            for _ in range(cfg.ring_senders):
                balance = _amount(rng, 60_000.0)
                amount = round(balance * float(rng.uniform(0.2, 0.8)), 2)
                step = int(rng.integers(start, end + 1))
                rows.append(_row(rng, step, "TRANSFER", amount, ids.new(), balance, dest, True))

    frame = pl.DataFrame(rows, schema=list(RAW_DTYPES), orient="row").cast(pl.Schema(RAW_DTYPES))
    order = rng.permutation(frame.height)  # mix rows within a step, then sort stably
    frame = frame[order].sort("step", maintain_order=True)
    return SynthResult(frame=frame, ring_destinations=tuple(ring_dests), ring_bursts=bursts)


def write_csv(result: SynthResult, path: Path) -> Path:
    """Write the raw-format CSV (the same layout as the Kaggle file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.write_csv(path, float_precision=2, line_terminator="\n")
    return path


def _step(day: int, hour: int) -> int:
    """PaySim step for a day (0-based) and an hour of day (= step % 24)."""
    return day * 24 + (hour if hour > 0 else 24)


def _amount(rng: np.random.Generator, median: float) -> float:
    return round(float(rng.lognormal(np.log(median), 1.0)), 2)


def _row(
    rng: np.random.Generator,
    step: int,
    type_: str,
    amount: float,
    orig: str,
    old_orig: float,
    dest: str,
    fraud: bool,
) -> tuple[object, ...]:
    incoming = type_ == "CASH_IN"
    new_orig = round(old_orig + amount if incoming else max(old_orig - amount, 0.0), 2)
    if dest.startswith("M") or (fraud and rng.random() < 0.5):
        old_dest = new_dest = 0.0  # merchants and half of fraud: the zero-zero artifact
    else:
        old_dest = 0.0 if rng.random() < 0.3 else _amount(rng, 200_000.0)
        new_dest = round(old_dest + (0.0 if incoming else amount), 2)
    flagged = fraud and type_ == "TRANSFER" and amount >= 2_000_000
    return (
        step, type_, amount, orig, old_orig, new_orig, dest, old_dest, new_dest,
        int(fraud), int(flagged),
    )  # fmt: skip

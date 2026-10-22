"""Drift scenario library (task 6.2, Juspay track J2).

Each scenario transforms *raw events* before the feature engine, so features see the drift
the way production would. A scenario has a start step, a mode (`sudden`, or `gradual`: the
share of affected payments ramps linearly over `ramp_steps`) and a seed, so the injection
point is known and detection delay can be measured. Raw fields stay consistent (amounts and
balances move together); injected scenarios create new accounts, events *and* their labels.

| ID | kind                | what changes                                                        |
|----|---------------------|---------------------------------------------------------------------|
| D0 | none                | nothing: the null stream (false-alarm rate)                          |
| D1 | small_ticket_fraud  | fraud amounts shrink into the legit range                            |
| D2 | channel_shift       | fraud TRANSFER -> CASH_OUT, plus 30 % of legit TRANSFERs             |
| D3 | mule_ring           | new destinations hit by many new senders in bursts                   |
| D4 | benign_inflation    | legit amounts x1.3 (covariate only: drift is not damage)             |
| D5 | volume_surge        | legit volume x3 (new senders, same receivers)                        |
| D6 | upi_mule_vpas       | many small P2P payments into new mule VPAs, which then cash out (J2) |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

KINDS = (
    "none",
    "small_ticket_fraud",
    "channel_shift",
    "mule_ring",
    "benign_inflation",
    "volume_surge",
    "upi_mule_vpas",
)
BALANCE_COLUMNS = ("oldbalance_orig", "newbalance_orig", "oldbalance_dest", "newbalance_dest")


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    start_step: int
    end_step: int
    mode: str = "sudden"
    ramp_steps: int = 24
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown scenario kind {self.kind!r}")
        if self.mode not in ("sudden", "gradual"):
            raise ValueError("mode must be sudden or gradual")

    def intensity(self, step: np.ndarray) -> np.ndarray:
        """Share of eligible payments affected at each step (0 before the start)."""
        after = (step >= self.start_step) & (step <= self.end_step)
        if self.mode == "sudden":
            return after.astype(float)
        ramp = np.clip((step - self.start_step + 1) / self.ramp_steps, 0.0, 1.0)
        return np.where(after, ramp, 0.0)


@dataclass
class Injected:
    events: pl.DataFrame
    injected_txn_ids: list[str]
    affected_txn_ids: list[str]  # existing payments a transform changed


def load_scenarios(path: Path) -> list[Scenario]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    return [Scenario(**{**defaults, **s}) for s in data["scenarios"]]


def apply(events: pl.DataFrame, scenario: Scenario) -> Injected:
    """Return the drifted event stream (step-sorted) for one scenario."""
    fn = {
        "none": _none,
        "small_ticket_fraud": _small_ticket,
        "channel_shift": _channel_shift,
        "mule_ring": _mule_ring,
        "benign_inflation": _inflation,
        "volume_surge": _volume_surge,
        "upi_mule_vpas": _upi_mules,
    }[scenario.kind]
    out = fn(events, scenario, np.random.default_rng(scenario.seed))
    out.events = out.events.sort("step", maintain_order=True)
    return out


# ----------------------------------------------------------------- transforms


def _none(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    return Injected(events, [], [])


def _chosen(
    events: pl.DataFrame, s: Scenario, rng: np.random.Generator, eligible: pl.Expr
) -> np.ndarray:
    mask = events.select(eligible).to_series().to_numpy().astype(bool)
    p = s.intensity(events["step"].to_numpy())
    return mask & (rng.random(events.height) < p)


def _scale_amounts(events: pl.DataFrame, chosen: np.ndarray, factor: np.ndarray) -> pl.DataFrame:
    f = pl.Series(np.where(chosen, factor, 1.0))
    return events.with_columns(
        [(pl.col("amount") * f).round(2).alias("amount")]
        + [(pl.col(c) * f).round(2).alias(c) for c in BALANCE_COLUMNS]
    )


def _small_ticket(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    chosen = _chosen(events, s, rng, pl.col("is_fraud"))
    low, high = s.params.get("factor", [0.02, 0.1])
    factor = rng.uniform(low, high, events.height)
    return Injected(_scale_amounts(events, chosen, factor), [], _ids(events, chosen))


def _channel_shift(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    fraud = _chosen(events, s, rng, pl.col("is_fraud") & (pl.col("type") == "TRANSFER"))
    legit_share = float(s.params.get("legit_share", 0.3))
    legit = _chosen(events, s, rng, ~pl.col("is_fraud") & (pl.col("type") == "TRANSFER"))
    legit &= rng.random(events.height) < legit_share
    chosen = fraud | legit
    out = events.with_columns(
        type=pl.when(pl.Series(chosen)).then(pl.lit("CASH_OUT")).otherwise(pl.col("type"))
    )
    return Injected(out, [], _ids(events, chosen))


def _inflation(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    chosen = _chosen(events, s, rng, ~pl.col("is_fraud"))
    factor = float(s.params.get("factor", 1.3))
    return Injected(
        _scale_amounts(events, chosen, np.full(events.height, factor)), [], _ids(events, chosen)
    )


def _volume_surge(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    copies = int(s.params.get("extra_copies", 2))
    chosen = _chosen(events, s, rng, ~pl.col("is_fraud"))
    base = events.filter(pl.Series(chosen))
    extra = []
    for c in range(copies):
        extra.append(
            base.with_columns(
                txn_id=pl.format("V{}{}", pl.lit(c), pl.col("txn_id")),
                name_orig=pl.format("C7{}{}", pl.lit(c), pl.col("name_orig").str.slice(1)),
            )
        )
    added = pl.concat(extra) if extra else events.clear()
    return Injected(pl.concat([events, added]), added["txn_id"].to_list(), [])


def _mule_ring(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    rows = []
    rings = int(s.params.get("rings", 40))
    dests = int(s.params.get("destinations_per_ring", 3))
    senders = int(s.params.get("senders_per_destination", 15))
    burst = int(s.params.get("burst_steps", 6))
    starts = _event_starts(s, rng, rings, burst)
    for r, start in enumerate(starts):
        for d in range(dests):
            dest = f"C8{s.seed:02d}{r:04d}{d:02d}00"
            for k in range(senders):
                balance = float(np.round(rng.lognormal(np.log(60_000), 0.8), 2))
                amount = float(np.round(balance * rng.uniform(0.2, 0.8), 2))
                rows.append(
                    _row(
                        f"R{s.seed:02d}{r:04d}{d:02d}{k:03d}",
                        int(rng.integers(start, start + burst)),
                        "TRANSFER",
                        amount,
                        f"C9{s.seed:02d}{r:04d}{d:02d}{k:03d}",
                        balance,
                        dest,
                    )
                )
    return _with_rows(events, rows)


def _upi_mules(events: pl.DataFrame, s: Scenario, rng: np.random.Generator) -> Injected:
    """J2: mule VPAs collect many small P2P payments (collect-request / scam victims), then
    move the money out with one CASH_OUT shortly after the burst."""
    rows = []
    mules = int(s.params.get("mules", 60))
    senders = int(s.params.get("senders_per_mule", 25))
    burst = int(s.params.get("burst_steps", 12))
    median = float(s.params.get("median_amount", 5_000.0))
    starts = _event_starts(s, rng, mules, burst + 3)
    for m, start in enumerate(starts):
        mule = f"C6{s.seed:02d}{m:05d}000"
        total = 0.0
        for k in range(senders):
            balance = float(np.round(rng.lognormal(np.log(40_000), 0.7), 2))
            amount = float(np.round(min(rng.lognormal(np.log(median), 0.6), balance * 0.9), 2))
            total += amount
            rows.append(
                _row(
                    f"U{s.seed:02d}{m:05d}{k:03d}",
                    int(rng.integers(start, start + burst)),
                    "TRANSFER",
                    amount,
                    f"C5{s.seed:02d}{m:05d}{k:03d}",
                    balance,
                    mule,
                )
            )
        rows.append(
            _row(
                f"U{s.seed:02d}{m:05d}999",
                int(start + burst + rng.integers(0, 3)),
                "CASH_OUT",
                round(total * 0.98, 2),
                mule,
                round(total, 2),
                f"C4{s.seed:02d}{m:05d}000",
            )
        )
    return _with_rows(events, rows)


# ----------------------------------------------------------------- helpers


def _event_starts(s: Scenario, rng: np.random.Generator, count: int, length: int) -> np.ndarray:
    """Start steps for injected bursts; gradual mode puts more of them later (ramp)."""
    last = max(s.start_step, s.end_step - length)
    if s.mode == "sudden":
        return np.sort(rng.integers(s.start_step, last + 1, count))
    u = np.sqrt(rng.random(count))  # density rising linearly over the window
    return np.sort((s.start_step + u * (last - s.start_step)).astype(int))


def _row(
    txn: str, step: int, typ: str, amount: float, orig: str, balance: float, dest: str
) -> dict[str, Any]:
    return {
        "txn_id": txn,
        "step": step,
        "type": typ,
        "amount": amount,
        "name_orig": orig,
        "oldbalance_orig": balance,
        "newbalance_orig": round(max(balance - amount, 0.0), 2),
        "name_dest": dest,
        "oldbalance_dest": 0.0,
        "newbalance_dest": amount,
        "is_fraud": True,
        "is_flagged_fraud": False,
    }


def _with_rows(events: pl.DataFrame, rows: list[dict[str, Any]]) -> Injected:
    if not rows:
        return Injected(events, [], [])
    added = pl.DataFrame(rows).cast({c: events.schema[c] for c in rows[0]}).select(events.columns)
    return Injected(pl.concat([events, added]), added["txn_id"].to_list(), [])


def _ids(events: pl.DataFrame, chosen: np.ndarray) -> list[str]:
    return events.filter(pl.Series(chosen))["txn_id"].to_list()

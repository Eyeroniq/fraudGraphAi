"""Time-based splits with a label-delay embargo (task 1.1).

All splits are contiguous step ranges, so a transaction is in at most one split and
every split is strictly later than the one before it. Inside the dense period
(steps 1..`dense_last_step`) boundaries are chosen by the share of modelled rows
(TRANSFER + CASH_OUT); everything after it is the late-period test.

    train_fit | emb | train_val | emb | calib_a | calib_b | test || late_test

Each split has one job (see ROADMAP 1.1):
- train_fit: fitting.  train_val: early stopping, tuning, model choice.
- calib_a: isotonic calibration, threshold and policy parameters.
- calib_b: conformal quantiles only.  test: final numbers.  late_test: natural shift.

Embargo: a model trained "at" the end of a training window cannot know the labels of
its last `L` steps yet (labels arrive late, D4), so those steps are dropped:
`max(train_fit.step) + L < min(train_val.step)` and
`max(train_val.step) + L < min(calib_a.step)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudgraph.config import SplitSettings

MODELLED_TYPES = ("TRANSFER", "CASH_OUT")
SPLITS = ("train_fit", "train_val", "calib_a", "calib_b", "test", "late_test")
STEPS_PER_DAY = 24
SHORT_BLOCK_STEPS = 6


class SplitError(ValueError):
    """The data cannot be split as configured (e.g. a split would be empty)."""


@dataclass(frozen=True)
class StepRange:
    start: int
    end: int  # inclusive

    @property
    def steps(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class SplitStats:
    rows: int
    modelled_rows: int
    fraud: int
    prevalence: float  # fraud / modelled_rows


@dataclass(frozen=True)
class SplitPlan:
    label_delay_steps: int
    ranges: dict[str, StepRange]
    embargoes: dict[str, StepRange | None]
    stats: dict[str, SplitStats]
    test_days: float
    bootstrap_block_steps: int

    def to_json(self, **extra: Any) -> str:
        return json.dumps({**asdict(self), **extra}, indent=2) + "\n"


def plan_splits(frame: pl.DataFrame, settings: SplitSettings, label_delay_steps: int) -> SplitPlan:
    """Choose the step boundaries for `frame` (a prepared PaySim table)."""
    if settings.train_share + settings.calib_share >= 1:
        raise SplitError("train_share + calib_share must leave room for the test split")
    dense_last = settings.dense_last_step
    max_step = int(frame["step"].max())  # type: ignore[arg-type]
    if max_step <= dense_last:
        raise SplitError(f"no late period: data ends at step {max_step} <= {dense_last}")

    counts = _modelled_rows_per_step(frame, dense_last)  # index i = step i + 1
    delay = label_delay_steps

    # Dense period -> train window | calib | test, by modelled-row share.
    train_end = _step_at_share(counts, 1, dense_last, settings.train_share)
    calib_end = _step_at_share(counts, 1, dense_last, settings.train_share + settings.calib_share)
    # Train window -> train_fit | train_val.
    val_start = _step_at_share(counts, 1, train_end, 1 - settings.train_val_share) + 1
    # Calib -> calib_a | calib_b.
    calib_a_end = _step_at_share(counts, train_end + 1, calib_end, settings.calib_a_share)

    ranges = {
        "train_fit": StepRange(1, val_start - 1 - delay),
        "train_val": StepRange(val_start, train_end - delay),
        "calib_a": StepRange(train_end + 1, calib_a_end),
        "calib_b": StepRange(calib_a_end + 1, calib_end),
        "test": StepRange(calib_end + 1, dense_last),
        "late_test": StepRange(dense_last + 1, max_step),
    }
    embargoes = {
        "before_train_val": StepRange(val_start - delay, val_start - 1) if delay else None,
        "before_calib_a": StepRange(train_end - delay + 1, train_end) if delay else None,
    }
    for name, step_range in ranges.items():
        if step_range.steps <= 0:
            raise SplitError(
                f"split {name} is empty ({step_range}); adjust shares or the label delay"
            )

    stats = _stats(frame, ranges)
    for name, split_stats in stats.items():
        if split_stats.modelled_rows == 0:
            raise SplitError(f"split {name} has no TRANSFER/CASH_OUT rows")

    test_days = ranges["test"].steps / STEPS_PER_DAY
    block = STEPS_PER_DAY if test_days >= settings.min_test_days else SHORT_BLOCK_STEPS
    return SplitPlan(
        label_delay_steps=delay,
        ranges=ranges,
        embargoes=embargoes,
        stats=stats,
        test_days=test_days,
        bootstrap_block_steps=block,
    )


def assign_split(frame: pl.DataFrame, plan: SplitPlan) -> pl.DataFrame:
    """Add a `split` column (null for embargoed steps)."""
    expr: pl.Expr = pl.lit(None, dtype=pl.String)
    for name in reversed(SPLITS):
        r = plan.ranges[name]
        expr = pl.when(pl.col("step").is_between(r.start, r.end)).then(pl.lit(name)).otherwise(expr)
    return frame.with_columns(split=expr)


def load_plan(path: Path) -> SplitPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SplitPlan(
        label_delay_steps=data["label_delay_steps"],
        ranges={k: StepRange(**v) for k, v in data["ranges"].items()},
        embargoes={k: StepRange(**v) if v else None for k, v in data["embargoes"].items()},
        stats={k: SplitStats(**v) for k, v in data["stats"].items()},
        test_days=data["test_days"],
        bootstrap_block_steps=data["bootstrap_block_steps"],
    )


def _modelled_rows_per_step(frame: pl.DataFrame, last_step: int) -> np.ndarray:
    per_step = (
        frame.filter(pl.col("type").is_in(MODELLED_TYPES) & (pl.col("step") <= last_step))
        .group_by("step")
        .len()
    )
    counts = np.zeros(last_step, dtype=np.int64)
    counts[per_step["step"].to_numpy() - 1] = per_step["len"].to_numpy()
    return counts


def _step_at_share(counts: np.ndarray, first: int, last: int, share: float) -> int:
    """First step in [first, last] where the cumulative modelled-row share reaches `share`."""
    window = counts[first - 1 : last]
    total = window.sum()
    if total == 0:
        raise SplitError(f"no modelled rows in steps {first}..{last}")
    cumulative = np.cumsum(window) / total
    return first + int(np.searchsorted(cumulative, share, side="left"))


def _stats(frame: pl.DataFrame, ranges: dict[str, StepRange]) -> dict[str, SplitStats]:
    stats = {}
    for name, r in ranges.items():
        part = frame.filter(pl.col("step").is_between(r.start, r.end))
        modelled = part.filter(pl.col("type").is_in(MODELLED_TYPES))
        fraud = int(modelled["is_fraud"].sum())
        stats[name] = SplitStats(
            rows=part.height,
            modelled_rows=modelled.height,
            fraud=fraud,
            prevalence=fraud / modelled.height if modelled.height else 0.0,
        )
    return stats

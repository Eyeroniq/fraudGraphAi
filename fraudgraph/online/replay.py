"""Stream replay with delayed, imperfect labels (task 6.1).

Every transaction gets a `label_step` (when its label becomes known) and an
`observed_fraud` (what the label says when it arrives):

- fraud: known after `delay_steps` +/- `jitter_steps`; a share `missing_fraud_share` is never
  reported (no chargeback) and so *looks legit* once the legit delay has passed;
- legit: implicitly confirmed after `legit_delay_steps` (no complaint arrived).

The same schedule drives the offline experiments (in process) and the replayer service
(`services/replayer`, which posts events to the API at a set speed).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class LabelConfig:
    delay_steps: int = 24
    jitter_steps: int = 6
    missing_fraud_share: float = 0.10
    legit_delay_steps: int = 24
    seed: int = 0


def label_schedule(frame: pl.DataFrame, config: LabelConfig) -> pl.DataFrame:
    """Add `label_step` and `observed_fraud` to a step-sorted event frame."""
    rng = np.random.default_rng(config.seed)
    n = frame.height
    fraud = frame["is_fraud"].to_numpy().astype(bool)
    step = frame["step"].to_numpy().astype(np.int64)
    jitter = rng.integers(-config.jitter_steps, config.jitter_steps + 1, n)
    reported = fraud & (rng.random(n) >= config.missing_fraud_share)
    label_step = np.where(
        reported, step + np.maximum(config.delay_steps + jitter, 1), step + config.legit_delay_steps
    )
    return frame.with_columns(
        label_step=pl.Series(label_step, dtype=pl.Int64), observed_fraud=pl.Series(reported)
    )


@dataclass(frozen=True)
class StepBatch:
    step: int
    events: pl.DataFrame  # events at this step, in file order
    labels: pl.DataFrame  # labels that become known at this step (txn_id, observed_fraud, ...)


def replay(frame: pl.DataFrame, *, first_step: int, last_step: int) -> Iterator[StepBatch]:
    """Yield one batch per step in [first_step, last_step]. `frame` needs `label_step`."""
    events = frame.filter(pl.col("step").is_between(first_step, last_step))
    labels = frame.filter(pl.col("label_step").is_between(first_step, last_step))
    by_step = events.partition_by("step", as_dict=True, maintain_order=True)
    labels_by_step = labels.partition_by("label_step", as_dict=True, maintain_order=True)
    empty_events, empty_labels = events.clear(), labels.clear()
    for step in range(first_step, last_step + 1):
        yield StepBatch(
            step=step,
            events=by_step.get((step,), empty_events),
            labels=labels_by_step.get((step,), empty_labels),
        )

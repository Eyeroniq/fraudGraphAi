"""Offline backfill: replay every event, in step order, through the online engine (D3).

Only TRANSFER and CASH_OUT rows get a feature row (they are what we model), but every
event updates state, because every event is part of an account's history.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.spec import Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore

MODELLED_TYPES = frozenset({"TRANSFER", "CASH_OUT"})
EVENT_COLUMNS = ("txn_id", "step", "type", "amount", "name_orig", "name_dest")
KEY_COLUMNS = ("txn_id", "step", "is_fraud")


@dataclass(frozen=True)
class BackfillReport:
    events: int
    feature_rows: int
    seconds: float
    peak_state_keys: int


def events_from(frame: pl.DataFrame) -> Iterable[Event]:
    for row in frame.select(EVENT_COLUMNS).iter_rows():
        yield Event(*row)


def backfill(frame: pl.DataFrame, spec: FeatureSpec) -> tuple[pl.DataFrame, BackfillReport]:
    """Features for every modelled row of a prepared, step-sorted PaySim frame."""
    store = InMemoryStore(spec.horizon_steps)
    engine = FeatureEngine(spec, store)
    names = engine.feature_names
    columns: dict[str, list[float]] = {name: [] for name in names}
    txn_ids: list[str] = []

    started = time.perf_counter()
    peak_keys = 0
    events = 0
    for ev in events_from(frame):
        if ev.type in MODELLED_TYPES:
            feats = engine.features(ev)
            txn_ids.append(ev.txn_id)
            for name in names:
                columns[name].append(feats[name])
        engine.apply(ev)
        events += 1
        if events % 200_000 == 0:
            peak_keys = max(peak_keys, len(store))
    peak_keys = max(peak_keys, len(store))

    features = pl.DataFrame(
        {"txn_id": txn_ids, **columns}, schema_overrides=dict.fromkeys(names, pl.Float32)
    )
    keys = frame.select(KEY_COLUMNS)
    table = keys.join(features, on="txn_id", how="inner", maintain_order="right")
    report = BackfillReport(
        events=events,
        feature_rows=table.height,
        seconds=round(time.perf_counter() - started, 1),
        peak_state_keys=peak_keys,
    )
    return table, report

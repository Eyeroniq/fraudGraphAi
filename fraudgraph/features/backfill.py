"""Offline backfill: replay every event, in step order, through the online engine (D3).

Only TRANSFER and CASH_OUT rows get a feature row (they are what we model), but every
event updates state, because every event is part of an account's history.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
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
    peak_memory_mb: float


class LabelQueue:
    """Simulated delayed labels (D4): fraud on a transaction at step s becomes known at
    s + L. A label is released at the first event at or after that step and recorded at
    that event's step, so features see it from the next step on (never earlier)."""

    def __init__(self, delay_steps: int) -> None:
        self.delay = delay_steps
        self._pending: deque[tuple[int, str]] = deque()

    def push(self, ev: Event) -> None:
        self._pending.append((ev.step, ev.name_dest))

    def release(self, engine: FeatureEngine, now: int) -> None:
        while self._pending and self._pending[0][0] + self.delay <= now:
            _, dest = self._pending.popleft()
            engine.apply_label(dest, True, now)


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (Windows and Unix)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a pseudo-handle (-1)
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return float(counters.PeakWorkingSetSize) / 1024 / 1024
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024  # KB on Linux


def events_from(frame: pl.DataFrame) -> Iterable[Event]:
    for row in frame.select(EVENT_COLUMNS).iter_rows():
        yield Event(*row)


def backfill(frame: pl.DataFrame, spec: FeatureSpec) -> tuple[pl.DataFrame, BackfillReport]:
    """Features for every modelled row of a prepared, step-sorted PaySim frame."""
    store = InMemoryStore(spec.horizon_steps)
    engine = FeatureEngine(spec, store)
    names = engine.feature_names
    # Preallocated float32 columns: Python lists of floats would cost ~8x the memory.
    n_rows = int(frame["type"].is_in(list(MODELLED_TYPES)).sum())
    columns = np.full((len(names), n_rows), np.nan, dtype=np.float32)
    txn_ids: list[str] = []

    fraud_rows = frame["is_fraud"].to_list()
    labels = LabelQueue(spec.label_delay_steps)

    started = time.perf_counter()
    peak_keys = 0
    events = 0
    for ev, is_fraud in zip(events_from(frame), fraud_rows, strict=True):
        labels.release(engine, ev.step)
        if ev.type in MODELLED_TYPES:
            feats = engine.features(ev)
            row = len(txn_ids)
            txn_ids.append(ev.txn_id)
            for i, name in enumerate(names):
                columns[i, row] = feats[name]
        engine.apply(ev)
        if is_fraud:
            labels.push(ev)
        events += 1
        if events % 200_000 == 0:
            peak_keys = max(peak_keys, len(store))
    peak_keys = max(peak_keys, len(store))

    features = pl.DataFrame(
        {"txn_id": txn_ids, **{name: columns[i] for i, name in enumerate(names)}}
    )
    keys = frame.select(KEY_COLUMNS)
    table = keys.join(features, on="txn_id", how="inner", maintain_order="right")
    report = BackfillReport(
        events=events,
        feature_rows=table.height,
        seconds=round(time.perf_counter() - started, 1),
        peak_state_keys=peak_keys,
        peak_memory_mb=peak_memory_mb(),
    )
    return table, report

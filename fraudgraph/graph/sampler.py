"""Strictly causal temporal neighbour sampling (task 3.4).

PyG's temporal `LinkNeighborLoader` needs the compiled `pyg-lib`, which has no Windows
wheels. On PaySim the neighbourhood that matters is simple anyway (docs/data_audit.md):
senders are single-use, so the useful context of a payment u -> v is the receiving
account's earlier incoming payments. For every target edge this module gathers the
last `k` edges into the same destination at strictly earlier steps within the horizon
(same-step edges never see each other, D4), with neighbour fraud labels visible only
once known (label known at s + L, visible from s + L + 1).

Everything is vectorised over the whole edge list: sort once by (dest, step), then a
target's neighbours are the k rows just before the first row of its own step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

TYPE_CODES = {"TRANSFER": 0, "CASH_OUT": 1}  # everything else -> 2
EDGE_FEATURES = ("log_amount", "is_transfer", "is_cash_out", "hour", "amount_at_cap")
NEIGHBOUR_FEATURES = (
    "log_amount",
    "is_transfer",
    "is_cash_out",
    "hour",
    "steps_ago",
    "known_fraud",
)
FRAUD_AMOUNT_CAP = 10_000_000.0


@dataclass(frozen=True)
class EdgeBatch:
    """Inputs for a set of target edges (row order = the order of `txn_ids`)."""

    txn_ids: np.ndarray  # [n]
    edge: np.ndarray  # [n, len(EDGE_FEATURES)] float32
    neighbours: np.ndarray  # [n, k, len(NEIGHBOUR_FEATURES)] float32 (zeros where masked)
    mask: np.ndarray  # [n, k] bool
    y: np.ndarray  # [n] float32

    def __len__(self) -> int:
        return len(self.txn_ids)

    def subset(self, rows: np.ndarray) -> EdgeBatch:
        return EdgeBatch(
            self.txn_ids[rows],
            self.edge[rows],
            self.neighbours[rows],
            self.mask[rows],
            self.y[rows],
        )


class TemporalNeighbours:
    """Precomputed per-edge arrays for the full event list; `batch()` builds inputs."""

    def __init__(self, events: pl.DataFrame, *, horizon_steps: int, label_delay_steps: int):
        frame = events.select("txn_id", "step", "type", "amount", "name_dest", "is_fraud")
        order = np.lexsort((frame["step"].to_numpy(), frame["name_dest"].to_numpy()))
        self.txn_id = frame["txn_id"].to_numpy()[order]
        self.step = frame["step"].to_numpy().astype(np.int64)[order]
        dest = frame["name_dest"].to_numpy()[order]
        type_code = (
            frame["type"].replace_strict(TYPE_CODES, default=2, return_dtype=pl.Int8).to_numpy()
        )[order]
        amount = frame["amount"].to_numpy()[order]
        self.is_fraud = frame["is_fraud"].to_numpy().astype(bool)[order]
        self.horizon = horizon_steps
        self.delay = label_delay_steps

        n = len(order)
        new_group = np.r_[True, dest[1:] != dest[:-1]]
        self.group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
        new_step = new_group | np.r_[True, self.step[1:] != self.step[:-1]]
        # First row of the same (dest, step): neighbours end just before it.
        self.step_start = np.maximum.accumulate(np.where(new_step, np.arange(n), 0))

        log_amount = np.log1p(amount).astype(np.float32)
        self.edge_table = np.stack(
            [
                log_amount,
                (type_code == 0).astype(np.float32),
                (type_code == 1).astype(np.float32),
                (self.step % 24).astype(np.float32) / 23.0,
                (amount == FRAUD_AMOUNT_CAP).astype(np.float32),
            ],
            axis=1,
        )
        # txn_id -> row via a sorted index (a dict over millions of ids costs ~600 MB).
        self._by_id = np.argsort(self.txn_id)
        self._sorted_ids = self.txn_id[self._by_id]

    def rows_of(self, txn_ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self._sorted_ids, txn_ids)
        if (pos >= len(self._sorted_ids)).any() or (self._sorted_ids[pos] != txn_ids).any():
            raise KeyError("unknown txn_id in batch")
        return np.asarray(self._by_id[pos], dtype=np.int64)

    def batch(self, txn_ids: np.ndarray, k: int) -> EdgeBatch:
        rows = self.rows_of(np.asarray(txn_ids))
        t = self.step[rows]
        candidates = self.step_start[rows, None] - np.arange(k, 0, -1)[None, :]  # [n, k]
        safe = np.clip(candidates, 0, None)
        mask = (candidates >= self.group_start[rows, None]) & (
            self.step[safe] >= (t - self.horizon)[:, None]
        )
        nb_steps = self.step[safe]
        known = self.is_fraud[safe] & (nb_steps + self.delay + 1 <= t[:, None])
        neighbours = np.concatenate(
            [
                self.edge_table[safe][..., :4],
                ((t[:, None] - nb_steps) / self.horizon).astype(np.float32)[..., None],
                known.astype(np.float32)[..., None],
            ],
            axis=2,
        )
        neighbours[~mask] = 0.0
        return EdgeBatch(
            txn_ids=np.asarray(txn_ids),
            edge=self.edge_table[rows],
            neighbours=neighbours.astype(np.float32),
            mask=mask,
            y=self.is_fraud[rows].astype(np.float32),
        )

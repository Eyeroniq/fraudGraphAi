"""Train the temporal edge GNN and export edge embeddings (tasks 3.4, 3.5).

CPU-friendly: all fraud edges plus a seeded sample of legit edges for training, early
stopping on train_val average precision, fixed seeds. Embeddings for the hybrid are
out-of-fold in time: a training row is only ever embedded by a model fitted on earlier
steps, so the hybrid never sees an embedding learned from its own label.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fraudgraph.graph.models import TemporalEdgeGNN
from fraudgraph.graph.sampler import EdgeBatch, TemporalNeighbours


@dataclass(frozen=True)
class GNNConfig:
    conv: str = "sage"
    k: int = 10
    hidden: int = 64
    embedding: int = 32
    dropout: float = 0.1
    epochs: int = 15
    patience: int = 3
    batch_size: int = 2_048
    lr: float = 1e-3
    weight_decay: float = 1e-5
    legit_per_fraud: int = 30
    seed: int = 0


@dataclass
class FitReport:
    config: dict[str, Any]
    train_rows: int
    best_epoch: int
    best_val_ap: float
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def _tensors(batch: EdgeBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.edge),
        torch.from_numpy(batch.neighbours),
        torch.from_numpy(batch.mask),
    )


def predict(
    model: TemporalEdgeGNN, batch: EdgeBatch, chunk: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    """(probability, embedding) for every row of `batch`."""
    model.eval()
    probs, embs = [], []
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            part = batch.subset(np.arange(start, min(start + chunk, len(batch))))
            emb = model.embed(*_tensors(part))
            probs.append(torch.sigmoid(model.head(emb).squeeze(-1)).numpy())
            embs.append(emb.numpy())
    if not probs:
        return np.zeros(0, dtype=np.float32), np.zeros((0, model.head.in_features), np.float32)
    return np.concatenate(probs), np.concatenate(embs)


def fit(
    neighbours: TemporalNeighbours,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    config: GNNConfig,
) -> tuple[TemporalEdgeGNN, FitReport]:
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    rng = np.random.default_rng(config.seed)

    # Pick the training sample from the labels first, then build neighbourhoods only for
    # it (building them for every row first would cost ~15x the memory for nothing).
    labels = neighbours.is_fraud[neighbours.rows_of(np.asarray(train_ids))]
    fraud_rows = np.flatnonzero(labels)
    legit_rows = np.flatnonzero(~labels)
    n_legit = min(len(legit_rows), config.legit_per_fraud * max(len(fraud_rows), 1))
    keep = np.sort(np.r_[fraud_rows, rng.choice(legit_rows, n_legit, replace=False)])
    train = neighbours.batch(np.asarray(train_ids)[keep], config.k)
    val = neighbours.batch(val_ids, config.k)

    model = TemporalEdgeGNN(config.conv, config.hidden, config.embedding, config.dropout)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    pos = max(float(train.y.sum()), 1.0)
    pos_weight = torch.tensor(np.sqrt((len(train) - pos) / pos), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    edge, nbs, mask = _tensors(train)
    y = torch.from_numpy(train.y)

    started = time.perf_counter()
    best: tuple[float, int, dict[str, Any]] = (-1.0, -1, {})
    history = []
    stale = 0
    for epoch in range(config.epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(train)))
        total = 0.0
        for start in range(0, len(train), config.batch_size):
            idx = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(edge[idx], nbs[idx], mask[idx]), y[idx])
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        val_prob, _ = predict(model, val)
        val_ap = float(average_precision_score(val.y, val_prob))
        history.append({"epoch": epoch, "loss": total / len(train), "val_ap": val_ap})
        if val_ap > best[0]:
            best = (val_ap, epoch, copy.deepcopy(model.state_dict()))
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best[2])
    report = FitReport(
        config=asdict(config),
        train_rows=len(train),
        best_epoch=best[1],
        best_val_ap=best[0],
        seconds=round(time.perf_counter() - started, 1),
        history=history,
    )
    return model, report


def out_of_fold_embeddings(
    neighbours: TemporalNeighbours,
    train_ids: np.ndarray,
    train_steps: np.ndarray,
    val_ids: np.ndarray,
    config: GNNConfig,
    folds: int = 4,
    embargo_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings for `train_ids`, each from a model fitted on strictly earlier folds,
    leaving out the last `embargo_steps` before the fold (their labels are not known yet
    when the fold starts, D4).

    Returns (embeddings [n, d], has_embedding [n] bool): rows of the first fold have no
    earlier data, so they get no embedding and are left out of the hybrid's training.
    """
    edges = np.quantile(train_steps, np.linspace(0, 1, folds + 1))
    fold = np.clip(np.searchsorted(edges, train_steps, side="right") - 1, 0, folds - 1)
    out = np.zeros((len(train_ids), config.embedding), dtype=np.float32)
    has = np.zeros(len(train_ids), dtype=bool)
    for j in range(1, folds):
        target = fold == j
        if not target.any():
            continue
        fold_start = int(train_steps[target].min())
        earlier = (fold < j) & (train_steps < fold_start - embargo_steps)
        if not earlier.any():
            continue
        model, _ = fit(neighbours, train_ids[earlier], val_ids, config)
        _, emb = predict(model, neighbours.batch(train_ids[target], config.k))
        out[target] = emb
        has[target] = True
    return out, has

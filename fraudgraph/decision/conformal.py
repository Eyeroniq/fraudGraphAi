"""Split conformal prediction sets for fraud / legit (task 4.1).

Given a calibrated probability p = P(fraud), the model's score for each class is
p_fraud = p and p_legit = 1 - p. A nonconformity score measures how unusual the true
class looks:

- LAC: s(x, y) = 1 - p_y.
- APS (non-randomised, binary): s = p_y if y is the most likely class, else 1.

Conformalised on `calib_b` (never test, never the isotonic data). With quantile
q_hat at level 1 - alpha, the set is {y : s(x, y) <= q_hat}:

- marginal: one q_hat over all calibration rows (coverage over all rows);
- class-conditional (Mondrian, D7): one q_hat per class, from that class's rows, so
  fraud coverage holds on its own instead of being swamped by 99.8 % legit rows.

Sets are {legit}, {fraud}, {legit, fraud} ("uncertain") or {} (empty). Serving needs
only the stored quantiles (conformal.json): no MAPIE import.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CLASSES = ("legit", "fraud")
SCORES = ("lac", "aps")


def class_probabilities(p_fraud: np.ndarray) -> np.ndarray:
    """[n, 2] matrix of (p_legit, p_fraud)."""
    p = np.clip(np.asarray(p_fraud, dtype=np.float64), 0.0, 1.0)
    return np.stack([1.0 - p, p], axis=1)


def nonconformity(probs: np.ndarray, score: str) -> np.ndarray:
    """Score of every class for every row, shape [n, 2]."""
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        top = probs.argmax(axis=1)
        out = np.ones_like(probs)
        rows = np.arange(len(probs))
        out[rows, top] = probs[rows, top]
        return out
    raise ValueError(f"unknown score {score!r}")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")  # too few calibration rows: every label is always included
    return float(np.sort(scores)[rank - 1])


@dataclass(frozen=True)
class ConformalModel:
    alpha: float
    score: str
    mondrian: bool
    q_hat: tuple[float, float]  # per class (legit, fraud); equal when marginal
    n_calib: tuple[int, int]

    def sets(self, p_fraud: np.ndarray) -> np.ndarray:
        """Boolean [n, 2] membership: column 0 legit, column 1 fraud."""
        s = nonconformity(class_probabilities(p_fraud), self.score)
        return s <= np.asarray(self.q_hat)[None, :]

    def to_json(self) -> str:
        data = asdict(self)
        data["q_hat"] = [_finite(q) for q in self.q_hat]
        return json.dumps(data, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ConformalModel:
        data = json.loads(text)
        q = tuple(float("inf") if v is None else float(v) for v in data["q_hat"])
        return cls(
            alpha=data["alpha"],
            score=data["score"],
            mondrian=data["mondrian"],
            q_hat=(q[0], q[1]),
            n_calib=(int(data["n_calib"][0]), int(data["n_calib"][1])),
        )


def fit(
    p_fraud: np.ndarray, y: np.ndarray, *, alpha: float, score: str = "lac", mondrian: bool = True
) -> ConformalModel:
    """Conformalise on calibration rows (calibrated probabilities and true labels)."""
    s = nonconformity(class_probabilities(p_fraud), score)
    y = np.asarray(y).astype(int)
    true_scores = s[np.arange(len(y)), y]
    counts = (int((y == 0).sum()), int((y == 1).sum()))
    if mondrian:
        q = tuple(conformal_quantile(true_scores[y == c], alpha) for c in (0, 1))
    else:
        qm = conformal_quantile(true_scores, alpha)
        q = (qm, qm)
    return ConformalModel(alpha, score, mondrian, (q[0], q[1]), counts)


def set_labels(sets: np.ndarray) -> np.ndarray:
    """'legit', 'fraud', 'uncertain' (both) or 'empty' per row."""
    out = np.full(len(sets), "empty", dtype=object)
    out[sets[:, 0] & ~sets[:, 1]] = "legit"
    out[~sets[:, 0] & sets[:, 1]] = "fraud"
    out[sets[:, 0] & sets[:, 1]] = "uncertain"
    return out


def coverage(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Share of rows whose true label is in the set: overall and per class."""
    y = np.asarray(y).astype(int)
    hit = sets[np.arange(len(y)), y]
    return {
        "overall": float(hit.mean()) if len(y) else float("nan"),
        "legit": float(hit[y == 0].mean()) if (y == 0).any() else float("nan"),
        "fraud": float(hit[y == 1].mean()) if (y == 1).any() else float("nan"),
    }


def save(model: ConformalModel, path: Path) -> None:
    path.write_text(model.to_json(), encoding="utf-8")


def load(path: Path) -> ConformalModel:
    return ConformalModel.from_json(path.read_text(encoding="utf-8"))


def _finite(value: float) -> float | None:
    return None if math.isinf(value) else value

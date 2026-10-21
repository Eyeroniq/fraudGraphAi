"""Probability calibration (task 1.5): isotonic on calib_a, Platt as the comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    """Maps a raw model score to P(fraud). `method` is "isotonic" or "platt"."""

    method: str
    model: IsotonicRegression | LogisticRegression

    def __call__(self, score: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(self.model.predict(score), dtype=np.float64)
        return np.asarray(self.model.predict_proba(_logit(score)[:, None])[:, 1])


def fit_calibrator(score: np.ndarray, y: np.ndarray, method: str = "isotonic") -> Calibrator:
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(score, y)
        return Calibrator("isotonic", iso)
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1_000)
        lr.fit(_logit(score)[:, None], y)
        return Calibrator("platt", lr)
    raise ValueError(f"unknown calibration method {method!r}")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE over equal-count bins (equal-width bins would put almost everything in bin 0)."""
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return total / len(p)


def _logit(score: np.ndarray) -> np.ndarray:
    s = np.clip(score, 1e-9, 1 - 1e-9)
    return np.asarray(np.log(s / (1 - s)))

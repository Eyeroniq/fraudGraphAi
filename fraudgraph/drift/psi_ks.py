"""Input and prediction drift statistics (task 6.4).

- PSI per feature, with bins fixed by the bundle's `reference_stats.json` (train_fit
  quantiles; NaN is its own bin). PSI = sum((a - e) * ln(a / e)); 0.1 warn, 0.2 alarm.
- KS (continuous) and chi-squared (binary / categorical) tests against a reference sample,
  with Benjamini-Hochberg across features. With thousands of rows per window these tests
  reject almost anything, so alarms need both a significant test *and* a material PSI.
- Prediction drift: PSI of the score distribution, and a z-score of the alert rate (share of
  payments not approved) against its spread over reference windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, ks_2samp

from fraudgraph.models.bundle import reference_stats

EPS = 1e-4


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Shares per bin [len(edges)+1 finite bins, then NaN], matching reference_stats."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = np.bincount(
        np.searchsorted(edges, values[finite], side="right"), minlength=len(edges) + 1
    )
    out = np.r_[counts, (~finite).sum()].astype(np.float64)
    shares: np.ndarray = out / max(len(values), 1)
    return shares


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=np.float64), EPS, None)
    a = np.clip(np.asarray(actual, dtype=np.float64), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of discoveries at false-discovery rate q."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out


def matrix_stats(x: np.ndarray, features: list[str]) -> dict[str, Any]:
    """Reference bins on the *model matrix* (NaNs already filled), as the monitors see it."""
    return reference_stats(pl.DataFrame({f: x[:, i] for i, f in enumerate(features)}), features)


def reference_alert_rates(
    scores: np.ndarray, step: np.ndarray, first: int, last: int, cut: float
) -> list[float]:
    """Alert rates over day-long windows sliding by 6 steps inside the reference period."""
    return [
        float(np.mean(scores[(step >= a) & (step < a + 24)] >= cut))
        for a in range(first, last - 22, 6)
    ]


@dataclass
class ReferenceFeature:
    name: str
    edges: np.ndarray
    shares: np.ndarray  # finite bins + NaN bin
    sample: np.ndarray
    categorical: bool


class InputDriftMonitor:
    def __init__(
        self,
        reference_stats: dict[str, Any],
        reference_sample: dict[str, np.ndarray],
        *,
        psi_warn: float = 0.1,
        psi_alarm: float = 0.2,
        fdr: float = 0.01,
    ) -> None:
        self.features = []
        for name, stats in reference_stats.items():
            sample = np.asarray(reference_sample[name], dtype=np.float64)
            finite = sample[np.isfinite(sample)]
            self.features.append(
                ReferenceFeature(
                    name=name,
                    edges=np.asarray(stats["edges"], dtype=np.float64),
                    shares=np.r_[stats["shares"], stats["nan_share"]],
                    sample=finite,
                    categorical=len(np.unique(finite)) <= 2,
                )
            )
        self.psi_warn, self.psi_alarm, self.fdr = psi_warn, psi_alarm, fdr

    def evaluate(self, window: dict[str, np.ndarray]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        pvalues = []
        for f in self.features:
            values = np.asarray(window[f.name], dtype=np.float64)
            value_psi = psi(f.shares, bin_shares(values, f.edges))
            finite = values[np.isfinite(values)]
            if f.categorical:
                levels = np.unique(np.r_[f.sample, finite])
                table = np.array(
                    [[np.sum(f.sample == v) for v in levels], [np.sum(finite == v) for v in levels]]
                )
                table = table[:, table.sum(axis=0) > 0]
                stat, p = (0.0, 1.0) if table.shape[1] < 2 else chi2_contingency(table)[:2]
                test = "chi2"
            else:
                stat, p = ks_2samp(f.sample, finite) if len(finite) else (0.0, 1.0)
                test = "ks"
            rows.append(
                {
                    "feature": f.name,
                    "psi": value_psi,
                    "test": test,
                    "stat": float(stat),
                    "p": float(p),
                }
            )
            pvalues.append(float(p))
        significant = benjamini_hochberg(np.array(pvalues), self.fdr)
        for row, sig in zip(rows, significant, strict=True):
            row["significant"] = bool(sig)
            row["alarm"] = bool(sig and row["psi"] > self.psi_alarm)
            row["warn"] = bool(sig and row["psi"] > self.psi_warn)
        top = max(rows, key=lambda r: r["psi"])
        return {
            "features": rows,
            "max_psi": top["psi"],
            "max_psi_feature": top["feature"],
            "alarm": any(r["alarm"] for r in rows),
            "alarm_features": [r["feature"] for r in rows if r["alarm"]],
        }


class PredictionDriftMonitor:
    """Score-distribution PSI (quantile bins of reference scores) and alert-rate z-score."""

    def __init__(
        self,
        reference_scores: np.ndarray,
        reference_alert_rates: np.ndarray,
        *,
        bins: int = 10,
        psi_alarm: float = 0.2,
        z_alarm: float = 4.0,
    ) -> None:
        self.edges = np.unique(np.quantile(reference_scores, np.linspace(0, 1, bins + 1))[1:-1])
        self.reference = bin_shares(reference_scores, self.edges)
        self.rate_mean = float(np.mean(reference_alert_rates))
        self.rate_std = float(max(np.std(reference_alert_rates), 1e-6))
        self.psi_alarm, self.z_alarm = psi_alarm, z_alarm

    def evaluate(self, scores: np.ndarray, alert_rate: float) -> dict[str, Any]:
        value = psi(self.reference, bin_shares(scores, self.edges))
        z = (alert_rate - self.rate_mean) / self.rate_std
        return {
            "score_psi": value,
            "alert_rate": alert_rate,
            "alert_rate_z": float(z),
            "alarm": bool(value > self.psi_alarm or abs(z) > self.z_alarm),
        }

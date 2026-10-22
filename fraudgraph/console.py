"""Data for the web console's report pages (Phase 8): `fgx console-report`.

One JSON file, `reports/console/lab.json`, served by `GET /v1/models/lab`. It condenses
what the console shows but the API does not compute live:

- `models`: test / late-test metrics per model (Phase 1 and 3);
- `pr_curves`: precision-recall on the test split, downsampled for drawing;
- `sample`: the champion's calibrated P(fraud), amount and label on test (every fraud,
  a random legit sample with weights) so cost sliders can redraw the cost curve in the
  browser (8.4);
- `policy`, `explain`, `drift`: the conformal/policy study (Phase 4), explanation
  quality (Phase 5) and the drift study (Phase 6), trimmed to what the pages draw.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

SCORE_PREFIX = "score_"
CURVE_POINTS = 120
LEGIT_SAMPLE = 4_000


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _round(values: np.ndarray, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values]


def pr_curve(y: np.ndarray, score: np.ndarray, points: int = CURVE_POINTS) -> dict[str, Any]:
    """Precision-recall downsampled to evenly spaced recall levels (monotone envelope)."""
    precision, recall, _ = precision_recall_curve(y, score)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # interpolated precision
    grid = np.linspace(0.0, 1.0, points)
    order = np.argsort(recall)
    p_grid = np.interp(grid, recall[order], precision[order])
    return {
        "ap": round(float(average_precision_score(y, score)), 5),
        "prevalence": round(float(np.mean(y)), 6),
        "recall": _round(grid, 4),
        "precision": _round(p_grid, 4),
    }


def build(
    predictions: Path,
    events: Path,
    calibrator: Path,
    reports: Path,
    champion: str,
    seed: int = 0,
) -> dict[str, Any]:
    preds = pl.read_parquet(predictions).filter(pl.col("split") == "test")
    y = preds["is_fraud"].to_numpy().astype(int)
    models = [c[len(SCORE_PREFIX) :] for c in preds.columns if c.startswith(SCORE_PREFIX)]
    curves = {m: pr_curve(y, preds[f"{SCORE_PREFIX}{m}"].to_numpy()) for m in models}

    # Champion sample for the client-side cost curve: calibrated p, amount, label, weight.
    cal = joblib.load(calibrator)
    amounts = pl.read_parquet(events, columns=["txn_id", "amount", "type"])
    test = preds.select("txn_id", "is_fraud", f"{SCORE_PREFIX}{champion}").join(
        amounts, on="txn_id", how="left"
    )
    rng = np.random.default_rng(seed)
    fraud = test["is_fraud"].to_numpy().astype(bool)
    legit_idx = np.flatnonzero(~fraud)
    pick = rng.choice(legit_idx, min(LEGIT_SAMPLE, len(legit_idx)), replace=False)
    keep = np.sort(np.r_[np.flatnonzero(fraud), pick])
    weight = np.where(fraud[keep], 1.0, len(legit_idx) / len(pick))
    p = cal(test[f"{SCORE_PREFIX}{champion}"].to_numpy()[keep])
    sample = {
        "p": _round(p, 6),
        "amount": _round(test["amount"].to_numpy()[keep], 2),
        "y": fraud[keep].astype(int).tolist(),
        "w": _round(weight, 3),
        "rows": len(test),
        "fraud": int(fraud.sum()),
    }

    metrics = _load(reports / "eval" / "baselines" / "metrics.json") or {}
    conformal = _load(reports / "eval" / f"conformal_{champion}" / "report.json") or {}
    explain = _load(reports / "explain" / champion / "report.json") or {}
    study = _load(reports / "drift" / "study.json") or {}
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "champion": champion,
        "models": {
            split: data.get("models", {}) for split, data in metrics.get("splits", {}).items()
        },
        "pr_curves": curves,
        "sample": sample,
        "policy": {
            k: conformal.get(k)
            for k in (
                "costs",
                "chosen",
                "policies_test",
                "coverage_grid",
                "review_load_vs_alpha",
                "coverage_per_day",
                "done_when",
                "fraud_rate_by_action",
            )
        },
        "explain": {
            k: explain.get(k)
            for k in (
                "studied",
                "global_mean_abs_group_shap",
                "flagged_mean_group_shap",
                "seed_stability",
                "stability_threshold",
                "perturbation",
                "background",
                "quality",
                "examples",
            )
        },
        "drift": _drift(study),
    }


def _drift(study: dict[str, Any]) -> dict[str, Any]:
    if not study:
        return {}
    scenarios = [
        {
            "id": s["scenario"],
            "kind": s["kind"],
            "mode": s["mode"],
            "start_step": s["start_step"],
            "detection": s["detection"],
            "policies": {
                name: {
                    "ap_post": p["ap_post"],
                    "cost_post": p["cost_post"],
                    "rolling_ap": p["rolling_ap"],
                    "time_to_recover_steps": p["time_to_recover_steps"],
                }
                for name, p in s["policies"].items()
            },
            "windows": [
                {"start": w["start"], "max_psi": w["max_psi"], "score_psi": w["score_psi"]}
                for w in s["windows"]
            ],
            "alarm_steps": s["alarm_steps"],
        }
        for s in study.get("scenarios", [])
    ]
    late = [
        {k: w[k] for k in ("start", "fraud_rate", "max_psi", "score_psi", "static_ap")}
        for w in study.get("natural_late_period", [])
    ]
    return {"thresholds": study.get("thresholds"), "scenarios": scenarios, "late_period": late}


def clean(value: Any) -> Any:
    """NaN / inf -> null: browsers reject them in JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    return value


def write(report: dict[str, Any], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(report), separators=(",", ":"), allow_nan=False)
    out.write_text(text, encoding="utf-8")
    return out

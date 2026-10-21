"""Model bundle (A3): the only thing the API loads.

    artifacts/bundles/<version>/
      manifest.json         version, git sha, data hash, features in order, metrics, files
      model.txt             LightGBM booster
      calibrator.joblib     isotonic calibrator
      policy.json           costs, rule, global threshold
      feature_spec.json     engine version, windows, horizon, label delay
      reference_stats.json  per-feature bins + shares on train_fit (drift reference, 6.4)

Later phases add conformal.json, reason_codes.yaml and shap_background.npy; the
manifest lists exactly the files present.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

MANIFEST = "manifest.json"
REFERENCE_BINS = 10


def bundle_version(model: str, git_sha: str, today: dt.date | None = None) -> str:
    day = (today or dt.datetime.now(dt.UTC).date()).strftime("%Y.%m.%d")
    return f"{model}-{day}-{git_sha.split('-')[0][:7]}"


def reference_stats(train: pl.DataFrame, features: list[str]) -> dict[str, Any]:
    """Quantile bin edges and the share of rows per bin, per feature (NaN is its own bin)."""
    stats: dict[str, Any] = {}
    for name in features:
        values = train[name].cast(pl.Float64).to_numpy()
        finite = values[np.isfinite(values)]
        edges = (
            np.unique(np.quantile(finite, np.linspace(0, 1, REFERENCE_BINS + 1)))
            if len(finite)
            else np.array([])
        )
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, finite, side="right"), minlength=len(inner) + 1)
        stats[name] = {
            "edges": inner.tolist(),
            "shares": (counts / max(len(values), 1)).tolist(),
            "nan_share": float(np.mean(~np.isfinite(values))) if len(values) else 0.0,
        }
    return stats


def write_bundle(
    out_root: Path,
    *,
    model: str,
    booster_path: Path,
    calibrator_path: Path,
    policy_path: Path,
    feature_spec: dict[str, Any],
    train_features: pl.DataFrame,
    conformal_path: Path | None = None,
    explain_dir: Path | None = None,
    metrics: dict[str, Any],
    data_sha256: str,
    git_sha: str,
    today: dt.date | None = None,
) -> Path:
    version = bundle_version(model, git_sha, today)
    target = out_root / version
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    booster = lgb.Booster(model_file=str(booster_path))
    features = list(booster.feature_name())
    shutil.copyfile(booster_path, target / "model.txt")
    shutil.copyfile(calibrator_path, target / "calibrator.joblib")
    shutil.copyfile(policy_path, target / "policy.json")
    if conformal_path is not None and conformal_path.is_file():
        shutil.copyfile(conformal_path, target / "conformal.json")
    for name in ("reason_codes.yaml", "explain.json", "shap_background.npy"):
        if explain_dir is not None and (explain_dir / name).is_file():
            shutil.copyfile(explain_dir / name, target / name)
    _write_json(target / "feature_spec.json", feature_spec)
    _write_json(target / "reference_stats.json", reference_stats(train_features, features))

    files = sorted(p.name for p in target.iterdir())
    manifest = {
        "version": version,
        "model": model,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "data_sha256": data_sha256,
        "engine_version": feature_spec["engine_version"],
        "features": features,
        "metrics": metrics,
        "files": {name: _sha256(target / name) for name in files},
    }
    _write_json(target / MANIFEST, manifest)
    return target


def verify_bundle(path: Path) -> dict[str, Any]:
    """Load the manifest and check every listed file's hash. Raises on any mismatch."""
    manifest: dict[str, Any] = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = _sha256(path / name)
        if actual != expected:
            raise ValueError(f"bundle file {name} changed: {actual} != {expected}")
    return manifest


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

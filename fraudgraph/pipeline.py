"""End-to-end Phase 1 pipeline on an in-memory frame -> a servable bundle.

Used by `fgx all --fixture` (CI builds a bundle from the synthetic data, so the API image
can be smoke-tested without the real dataset, D10) and by the API tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from fraudgraph.config import SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.splits import assign_split, plan_splits
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.decision.costs import CostParams, best_global_threshold
from fraudgraph.features.backfill import backfill
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.models.baselines import feature_sets, fit_lgbm, labels, matrix
from fraudgraph.models.bundle import write_bundle
from fraudgraph.models.calibrate import fit_calibrator


def build_bundle(
    events: pl.DataFrame,
    out_root: Path,
    *,
    split_settings: SplitSettings,
    label_delay_steps: int,
    costs: CostParams,
    trials: int = 3,
    seed: int = 0,
    model_name: str = "lgbm",
    git_sha: str = "fixture",
    data_sha256: str = "fixture",
) -> Path:
    spec = FeatureSpec()
    features, _ = backfill(events, spec)
    plan = plan_splits(events, split_settings, label_delay_steps)
    table = assign_split(features, plan)
    parts = {s: table.filter(pl.col("split") == s) for s in ("train_fit", "train_val", "calib_a")}
    names = feature_sets(table)["" if model_name == "lgbm" else "_no_hour"]
    x = {s: matrix(t, names, spec.horizon_steps) for s, t in parts.items()}
    y = {s: labels(t) for s, t in parts.items()}

    booster, _ = fit_lgbm(
        x["train_fit"],
        y["train_fit"],
        x["train_val"],
        y["train_val"],
        names,
        trials=trials,
        seed=seed,
    )
    calibrator = fit_calibrator(
        np.asarray(booster.predict(x["calib_a"], num_iteration=booster.best_iteration)),
        y["calib_a"],
    )
    calib_p = calibrator(
        np.asarray(booster.predict(x["calib_a"], num_iteration=booster.best_iteration))
    )
    amount = (
        parts["calib_a"]
        .join(events.select("txn_id", "amount"), on="txn_id", how="left")["amount"]
        .to_numpy()
    )
    threshold, _ = best_global_threshold(calib_p, y["calib_a"], amount, costs)

    work = out_root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(work / "model.txt"), num_iteration=booster.best_iteration)
    joblib.dump(calibrator, work / "calibrator.joblib")
    policy = {
        "model": model_name,
        "calibration": "isotonic",
        "costs": costs.__dict__,
        "rule": "expected_cost",
        "global_threshold": threshold,
        "calib_prevalence": float(y["calib_a"].mean()),
    }
    (work / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    feature_spec = {
        "engine_version": spec.version,
        "windows": list(spec.windows),
        "horizon_steps": spec.horizon_steps,
        "label_delay_steps": label_delay_steps,
    }
    return write_bundle(
        out_root,
        model=model_name,
        booster_path=work / "model.txt",
        calibrator_path=work / "calibrator.joblib",
        policy_path=work / "policy.json",
        feature_spec=feature_spec,
        train_features=parts["train_fit"],
        metrics={"source": "synthetic fixture"},
        data_sha256=data_sha256,
        git_sha=git_sha,
    )


def fixture_bundle(out_root: Path, *, seed: int = 7, trials: int = 3) -> Path:
    """Build a bundle from the synthetic fixture (its dense period ends before the sparse days)."""
    cfg = SynthConfig(seed=seed)
    events = to_canonical(generate(cfg).frame)
    return build_bundle(
        events,
        out_root,
        split_settings=SplitSettings(dense_last_step=(cfg.days - cfg.sparse_days) * 24),
        label_delay_steps=12,
        costs=CostParams(c_review=2_000.0, c_friction=20_000.0),
        trials=trials,
    )

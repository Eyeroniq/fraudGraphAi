from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from fraudgraph.models.bundle import (
    bundle_version,
    reference_stats,
    verify_bundle,
    write_bundle,
)
from fraudgraph.models.calibrate import fit_calibrator
from fraudgraph.tracking import Tracker, git_sha, tracked_run, tracking_uri


def test_git_sha_is_short_hash_or_unknown() -> None:
    sha = git_sha()
    assert sha == "unknown" or len(sha.split("-")[0]) == 12


def test_relative_sqlite_uri_resolves_against_repo_root() -> None:
    uri = tracking_uri("sqlite:///mlruns/mlflow.db")
    assert uri.startswith("sqlite:///")
    assert Path(uri.removeprefix("sqlite:///")).is_absolute()
    assert tracking_uri("http://mlflow:5000") == "http://mlflow:5000"


def test_disabled_tracker_is_a_no_op(tmp_path: Path) -> None:
    tracker = Tracker(None)
    tracker.params({"a": 1})
    tracker.metrics({"m": 1.0})
    tracker.artifact(tmp_path)
    assert not tracker.enabled


def test_tracked_run_records_params_metrics_and_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("FG_MLFLOW__TRACKING_URI", uri)
    monkeypatch.setenv("FG_MLFLOW__EXPERIMENT", "test-exp")

    with tracked_run("unit", data_sha256="abc123", task="1.6") as tracker:
        tracker.params({"seed": 7})
        tracker.metrics({"ap": 0.25, "skip": float("nan")})

    mlflow.set_tracking_uri(uri)
    (run,) = mlflow.search_runs(experiment_names=["test-exp"], output_format="list")
    assert run.data.params["seed"] == "7"
    assert run.data.metrics == {"ap": 0.25}
    assert run.data.tags["data_sha256"] == "abc123"
    assert "git_sha" in run.data.tags


def test_bundle_version_format() -> None:
    assert bundle_version("lgbm", "48749db59c30-dirty", dt.date(2026, 10, 21)) == (
        "lgbm-2026.10.21-48749db"
    )


def test_reference_stats_shares_cover_all_rows() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, float("nan")] * 20})
    stats = reference_stats(frame, ["x"])["x"]
    assert sum(stats["shares"]) + stats["nan_share"] == pytest.approx(1.0)
    assert stats["nan_share"] == pytest.approx(0.2)


@pytest.fixture
def bundle_inputs(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    x = rng.random((500, 2))
    y = (x[:, 0] + rng.normal(0, 0.2, 500) > 0.8).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1},
        lgb.Dataset(x, y, feature_name=["t_a", "o_b"]),
        num_boost_round=5,
    )
    paths = {
        "booster": tmp_path / "m.txt",
        "calibrator": tmp_path / "cal.joblib",
        "policy": tmp_path / "policy.json",
    }
    booster.save_model(str(paths["booster"]))
    joblib.dump(fit_calibrator(booster.predict(x), y), paths["calibrator"])
    paths["policy"].write_text(json.dumps({"rule": "expected_cost"}), encoding="utf-8")
    return paths


def test_write_and_verify_bundle(bundle_inputs: dict[str, Path], tmp_path: Path) -> None:
    train = pl.DataFrame({"t_a": np.linspace(0, 1, 50), "o_b": np.linspace(1, 2, 50)})
    path = write_bundle(
        tmp_path / "bundles",
        model="lgbm",
        booster_path=bundle_inputs["booster"],
        calibrator_path=bundle_inputs["calibrator"],
        policy_path=bundle_inputs["policy"],
        feature_spec={"engine_version": "1.2.0", "horizon_steps": 168},
        train_features=train,
        metrics={"ap": 0.3},
        data_sha256="d" * 64,
        git_sha="abcdef123456",
        today=dt.date(2026, 10, 21),
    )

    manifest = verify_bundle(path)
    assert path.name == "lgbm-2026.10.21-abcdef1"
    assert manifest["features"] == ["t_a", "o_b"]
    assert manifest["engine_version"] == "1.2.0"
    assert set(manifest["files"]) == {
        "model.txt",
        "calibrator.joblib",
        "policy.json",
        "feature_spec.json",
        "reference_stats.json",
    }

    (path / "policy.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"policy.json"):
        verify_bundle(path)

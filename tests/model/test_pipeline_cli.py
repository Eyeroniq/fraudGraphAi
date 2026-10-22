"""The full step-by-step pipeline through the CLI, on the synthetic fixture (task 9.3).

The same commands the full-pipeline workflow runs on PaySim: synth -> prepare -> split ->
features -> baselines -> evaluate -> champion -> policy -> bundle -> conformal -> explain
-> console report -> validate. Every step must exit 0 and leave its output behind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fraudgraph.cli import app
from fraudgraph.config import get_settings
from fraudgraph.data.download import sha256_of

pytestmark = pytest.mark.model
runner = CliRunner()


def run(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, (
        f"fgx {' '.join(args)} failed:\n{result.output}\n{result.exception!r}"
    )
    return result.output


def test_full_pipeline_runs_end_to_end_on_the_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, arts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    csv = data / "raw" / "paysim_fixture.csv"
    monkeypatch.setenv("FG_PATHS__DATA_DIR", str(data))
    monkeypatch.setenv("FG_PATHS__ARTIFACTS_DIR", str(arts))
    monkeypatch.setenv("FG_PATHS__REPORTS_DIR", str(reports))
    monkeypatch.setenv(
        "FG_MLFLOW__TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    monkeypatch.setenv("FG_DATA__CSV_NAME", csv.name)
    monkeypatch.setenv("FG_SPLITS__DENSE_LAST_STEP", "192")  # the fixture's dense days
    monkeypatch.setenv("FG_SPLITS__MIN_TEST_DAYS", "1")
    monkeypatch.setenv("FG_LABEL_DELAY_STEPS", "12")  # as the fixture bundle
    get_settings.cache_clear()

    run("data", "synth", "--out", str(csv))
    monkeypatch.setenv("FG_DATA__RAW_SHA256", sha256_of(csv))
    get_settings.cache_clear()
    predictions = str(arts / "baselines" / "predictions.parquet")

    run("data", "prepare")
    run("split")
    run("features", "build")
    run("train", "baselines", "--trials", "2")
    run("evaluate", "--predictions", predictions, "--resamples", "50")
    run("train", "champion", "--trials", "2")
    run("policy", "--predictions", predictions, "--resamples", "50")
    run("bundle")
    run("conformal", "--predictions", predictions, "--resamples", "50")
    run("explain", "--flagged", "150", "--quality-rows", "40")
    run("console-report")
    out = run("validate", "--profile", "fixture")

    assert "0 failed" in out
    bundles = list((arts / "bundles").glob("lgbm_g1-*/manifest.json"))
    assert len(bundles) == 1
    lab = json.loads((reports / "console" / "lab.json").read_text(encoding="utf-8"))
    assert {"lgbm", "lgbm_g1"} <= set(lab["pr_curves"])
    assert len(lab["sample"]["p"]) == len(lab["sample"]["y"]) > 0
    validation = json.loads(
        (reports / "validation" / "full" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["passed"] is True
    assert (reports / "validation" / "full" / "validation.html").is_file()

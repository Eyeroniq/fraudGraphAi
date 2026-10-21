"""Experiment tracking: a thin wrapper so MLflow's API churn touches one file (A11).

Every run records params, metrics, the data hash and the git sha. MLflow is part of
the `research` extra; where it is not installed (serving image, minimal CI) tracking
silently becomes a no-op, so nothing else has to care.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fraudgraph.config import REPO_ROOT, get_settings


def git_sha() -> str:
    """Current commit, with `-dirty` when tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


class Tracker:
    def __init__(self, mlflow: Any | None) -> None:
        self._mlflow = mlflow

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def params(self, values: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.log_params({k: _short(v) for k, v in values.items()})

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            clean = {k: float(v) for k, v in values.items() if v == v}  # drop NaN
            self._mlflow.log_metrics(clean, step=step)

    def artifact(self, path: Path) -> None:
        if self._mlflow:
            if path.is_dir():
                self._mlflow.log_artifacts(str(path), artifact_path=path.name)
            else:
                self._mlflow.log_artifact(str(path))


@contextmanager
def tracked_run(name: str, *, data_sha256: str | None = None, **tags: str) -> Iterator[Tracker]:
    """`with tracked_run("baselines", data_sha256=...) as t: t.params(...)`."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        yield Tracker(None)
        return

    settings = get_settings()
    mlflow.set_tracking_uri(tracking_uri(settings.mlflow.tracking_uri))
    experiment = settings.mlflow.experiment
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = REPO_ROOT / "mlruns" / "artifacts"
        mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)
    all_tags = {"git_sha": git_sha(), **tags}
    if data_sha256:
        all_tags["data_sha256"] = data_sha256
    with mlflow.start_run(run_name=name, tags=all_tags):
        yield Tracker(mlflow)


def tracking_uri(configured: str) -> str:
    """Resolve a relative `sqlite:///` path against the repository root."""
    prefix = "sqlite:///"
    if configured.startswith(prefix):
        path = Path(configured.removeprefix(prefix))
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + path.as_posix()
    return configured


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."

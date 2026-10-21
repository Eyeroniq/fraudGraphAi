from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fraudgraph.config import REPO_ROOT, Settings, get_settings


def test_defaults_come_from_base_yaml() -> None:
    settings = Settings()

    assert settings.seed == 42
    assert settings.label_delay_steps == 24
    assert settings.mlflow.experiment == "fraudgraph"


def test_relative_paths_resolve_against_repo_root() -> None:
    settings = Settings()

    assert settings.paths.data_dir == (REPO_ROOT / "data").resolve()
    assert settings.paths.reports_dir.is_absolute()


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FG_SEED", "7")
    monkeypatch.setenv("FG_PATHS__DATA_DIR", str(tmp_path))

    settings = Settings()

    assert settings.seed == 7
    assert settings.paths.data_dir == tmp_path


def test_fg_config_selects_another_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    other = tmp_path / "other.yaml"
    other.write_text("seed: 99\nlabel_delay_steps: 6\n", encoding="utf-8")
    monkeypatch.setenv("FG_CONFIG", str(other))

    settings = Settings()

    assert (settings.seed, settings.label_delay_steps) == (99, 6)


def test_missing_fg_config_file_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FG_CONFIG", str(tmp_path / "nope.yaml"))

    with pytest.raises(FileNotFoundError, match="FG_CONFIG"):
        Settings()


def test_unknown_keys_are_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    typo = tmp_path / "typo.yaml"
    typo.write_text("sead: 1\n", encoding="utf-8")
    monkeypatch.setenv("FG_CONFIG", str(typo))

    with pytest.raises(ValidationError):
        Settings()


def test_negative_label_delay_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FG_LABEL_DELAY_STEPS", "-1")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()

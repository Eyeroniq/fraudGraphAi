from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from fraudgraph import __version__
from fraudgraph.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_config_show_prints_effective_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FG_SEED", "123")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["seed"] == 123

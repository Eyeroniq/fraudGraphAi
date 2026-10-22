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


def test_api_key_prints_a_key_and_its_hash_entry() -> None:
    from services.api.security import ApiKeys

    result = runner.invoke(app, ["api-key", "--scope", "read"])
    lines = dict(line.split(":", 1) for line in result.stdout.strip().splitlines())
    key, entry = lines["key"].strip(), lines["entry"].strip()

    assert result.exit_code == 0
    assert ApiKeys.parse(entry).authenticate(key, "read").scope == "read"
    assert runner.invoke(app, ["api-key", "--scope", "root"]).exit_code == 1

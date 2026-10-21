from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from fraudgraph.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Each test starts from base.yaml: no FG_* variables from the developer's shell and
    no `.env` file (the working directory is an empty temp dir)."""
    for name in list(os.environ):
        if name.startswith("FG_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

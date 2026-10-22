from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudgraph import console
from tests.api_helpers import api_settings, auth


def test_pr_curve_is_monotone_and_reports_ap() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    score = y * 0.5 + rng.random(2_000)
    curve = console.pr_curve(y, score, points=50)
    assert len(curve["recall"]) == len(curve["precision"]) == 50
    assert curve["recall"][0] == 0.0
    assert curve["recall"][-1] == 1.0
    assert all(
        a >= b - 1e-9 for a, b in zip(curve["precision"], curve["precision"][1:], strict=False)
    )
    assert 0.05 < curve["ap"] < 1
    assert curve["prevalence"] == pytest.approx(y.mean(), abs=1e-6)


def test_clean_turns_nan_into_null_for_json() -> None:
    data = {"a": [1.0, math.nan, {"b": math.inf}], "c": (2.0, -math.inf), "d": "x"}
    assert console.clean(data) == {"a": [1.0, None, {"b": None}], "c": [2.0, None], "d": "x"}


def test_lab_endpoint_serves_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraudgraph.config import get_settings
    from services.api.main import create_app

    monkeypatch.setenv("FG_PATHS__REPORTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    client = TestClient(create_app(None, api_settings()), headers=auth("read"))
    assert client.get("/v1/models/lab").status_code == 404
    console.write({"champion": "x", "value": math.nan}, tmp_path / "console" / "lab.json")
    response = client.get("/v1/models/lab")
    assert response.status_code == 200
    assert json.loads(response.text) == {"champion": "x", "value": None}

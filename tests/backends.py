"""External services for `integration` tests (Redis, Postgres).

Tests read `TEST_REDIS_URL` / `TEST_DATABASE_URL` (not `FG_*`: the autouse fixture in
conftest clears those). Without them an integration test is skipped, unless
`REQUIRE_INTEGRATION=1` (set in CI), where a missing service is a failure.
"""

from __future__ import annotations

import os

import pytest


def service_url(name: str) -> str:
    url = os.environ.get(name)
    if url:
        return url
    if os.environ.get("REQUIRE_INTEGRATION") == "1":
        pytest.fail(f"{name} is required for integration tests")
    pytest.skip(f"{name} not set")

"""API keys and settings for API tests."""

from __future__ import annotations

from typing import Any

from fraudgraph.config import ApiSettings
from services.api.security import hash_key

KEYS = {"read": "fg_test_read", "write": "fg_test_write", "admin": "fg_test_admin"}


def api_settings(**overrides: Any) -> ApiSettings:
    keys = ",".join(f"{scope}:{hash_key(key)}" for scope, key in KEYS.items())
    return ApiSettings(**{"api_keys": keys, "cors_origins": "", **overrides})


def auth(scope: str) -> dict[str, str]:
    return {"X-API-Key": KEYS[scope]}

"""Shared state and dependencies of the API routers."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from fraudgraph.config import ApiSettings
from fraudgraph.features.redis_store import RedisStore
from fraudgraph.serving.scorer import Scorer
from services.api.db import DecisionLog, Scored
from services.api.schemas import ErrorResponse, ScoreResponse
from services.api.security import KEY_HEADER, ApiError, ApiKeys, Principal
from services.api.ws import Broadcaster

log = logging.getLogger("fraudgraph.api")
IDEMPOTENCY_CACHE = 100_000
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class AppContext:
    settings: ApiSettings
    keys: ApiKeys
    log: DecisionLog
    broadcaster: Broadcaster
    reports_dir: Path
    redis: RedisStore | None = None
    scorer: Scorer | None = None
    detail: str | None = None
    # Scoring reads then writes state: one request at a time per process (the Lua
    # script keeps several processes consistent with each other).
    lock: threading.Lock = field(default_factory=threading.Lock)
    seen: OrderedDict[str, ScoreResponse] = field(default_factory=OrderedDict)

    def require_scorer(self) -> Scorer:
        if self.scorer is None:
            raise ApiError(503, "not_ready", str(self.detail))
        return self.scorer

    def remember(self, response: ScoreResponse) -> None:
        self.seen[response.txn_id] = response
        if len(self.seen) > IDEMPOTENCY_CACHE:
            self.seen.popitem(last=False)

    def write(self, scored: Sequence[Scored]) -> None:
        """Background task: the response is already sent, so failures are only logged
        (transaction IDs only, no amounts or accounts)."""
        try:
            self.log.record(scored)
        except Exception as exc:
            ids = ",".join(s.decision.txn_id for s in scored[:5])
            log.error("decision log write failed for %s: %s", ids, type(exc).__name__)


def context(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


api_key_header = APIKeyHeader(name=KEY_HEADER, auto_error=False, scheme_name="ApiKey")


def requires(scope: str) -> Any:
    """Dependency: the caller's key must have at least `scope` (read < write < admin)."""

    def check(request: Request, key: str | None = Security(api_key_header)) -> Principal:
        principal = context(request).keys.authenticate(key, scope)
        request.state.key_id = principal.key_id
        return principal

    return Depends(check)


def rate_key(request: Request) -> str:
    """Limit per API key; per client address when no key is known yet."""
    key_id = getattr(request.state, "key_id", None)
    if key_id and key_id != "anonymous":
        return f"key:{key_id}"
    return f"ip:{get_remote_address(request)}"


def make_limiter(settings: ApiSettings) -> Limiter:
    # Shared counters across workers when Redis is configured; errors fail open (a
    # broken limiter must not take scoring down).
    return Limiter(
        key_func=rate_key,
        storage_uri=settings.redis_url or "memory://",
        swallow_errors=True,
    )


def limited(limiter: Limiter, rule: str) -> Callable[[F], F]:
    decorator: Callable[[F], F] = limiter.limit(rule)
    return decorator


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    text = {
        401: "Missing or invalid API key",
        403: "API key scope too low",
        404: "Not found",
        409: "Conflict",
        413: "Request body too large",
        422: "Invalid request",
        429: "Rate limit exceeded",
        501: "Not implemented yet",
        503: "Not ready (no model or backend down)",
    }
    return {c: {"model": ErrorResponse, "description": text[c]} for c in codes}

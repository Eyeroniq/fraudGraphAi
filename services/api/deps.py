"""Shared state and dependencies of the API routers."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from fraudgraph.config import ApiSettings
from fraudgraph.features.redis_store import RedisStore
from fraudgraph.features.spec import Event
from fraudgraph.serving.scorer import BundleModel, Scorer
from services.api.db import DecisionLog, Scored
from services.api.observability import Metrics
from services.api.schemas import ErrorResponse, ScoreResponse
from services.api.security import KEY_HEADER, ApiError, ApiKeys, Principal
from services.api.ws import Broadcaster

log = logging.getLogger("fraudgraph.api")
IDEMPOTENCY_CACHE = 100_000
EVENT_FIELDS = ("txn_id", "step", "type", "amount", "name_orig", "name_dest")
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
    bundles_root: Path | None = None
    metrics: Metrics = field(default_factory=Metrics)
    # Shadow challengers (7.5): scored on the champion's features, never served.
    challengers: dict[str, BundleModel] = field(default_factory=dict)
    challengers_checked: float = float("-inf")
    # Scoring reads then writes state: one request at a time per process (the Lua
    # script keeps several processes consistent with each other).
    lock: threading.Lock = field(default_factory=threading.Lock)
    seen: OrderedDict[str, ScoreResponse] = field(default_factory=OrderedDict)

    def require_scorer(self) -> Scorer:
        if self.scorer is None:
            raise ApiError(503, "not_ready", str(self.detail))
        return self.scorer

    def refresh_challengers(self, force: bool = False) -> None:
        """Load bundles registered as challengers, drop retired ones (rate-limited)."""
        now = time.monotonic()
        if not force and now - self.challengers_checked < self.settings.challenger_refresh_s:
            return
        self.challengers_checked = now
        if self.scorer is None or self.bundles_root is None:
            return
        try:
            wanted = {m.version: m.bundle_uri for m in self.log.challengers()}
        except Exception as exc:
            log.warning("could not read challengers: %s", type(exc).__name__)
            return
        for version in [v for v in self.challengers if v not in wanted]:
            del self.challengers[version]
        for version, uri in wanted.items():
            if version in self.challengers or version == self.scorer.version:
                continue
            try:
                model = BundleModel(self.bundles_root / uri)
                model.check_features(self.scorer.engine.feature_names)
            except Exception as exc:  # a broken bundle must not stop scoring
                log.warning("challenger %s not loaded: %s", version, exc)
                continue
            self.challengers[version] = model

    def promote(self, version: str) -> Scorer:
        """Serve `version` from now on. The new bundle is loaded and verified first; the
        swap itself happens under the scoring lock, so no request sees half of it. Feature
        state is kept (same store): the engine version guard makes that safe."""
        current = self.require_scorer()
        rows = {m.version: m for m in self.log.models()}
        if version not in rows:
            raise ApiError(404, "unknown_model", f"no model version {version}")
        if version == current.version:
            return current
        if self.bundles_root is None:
            raise ApiError(503, "not_ready", "no bundle directory configured")
        try:
            new = Scorer(self.bundles_root / rows[version].bundle_uri, store=current.engine.store)
        except Exception as exc:
            raise ApiError(409, "bad_bundle", f"{type(exc).__name__}: {exc}") from exc
        with self.lock:
            self.scorer = new
            self.challengers.pop(version, None)
        self.log.register_model(new.version, rows[version].metrics, rows[version].bundle_uri)
        self.metrics.set_model(new.version)
        log.info("promoted %s (was %s)", new.version, current.version)
        return new

    def remember(self, response: ScoreResponse) -> None:
        self.seen[response.txn_id] = response
        if len(self.seen) > IDEMPOTENCY_CACHE:
            self.seen.popitem(last=False)

    def shadow(self, scored: Scored) -> Scored:
        """Challengers' decisions on the champion's features (no state is read)."""
        self.refresh_challengers()
        if not scored.features or not self.challengers:
            return scored
        event = Event(**{k: scored.txn[k] for k in EVENT_FIELDS})
        out = []
        for version, model in list(self.challengers.items()):
            try:
                decision = model.decide(event, scored.features, explain=False)
            except Exception as exc:
                log.warning("shadow scoring with %s failed: %s", version, type(exc).__name__)
                continue
            self.metrics.shadow.labels(model_version=version, decision=decision.decision).inc()
            out.append(decision)
        return replace(scored, shadow=tuple(out))

    def write(self, scored: Sequence[Scored]) -> None:
        """Background task: the response is already sent, so challengers score here and
        failures are only logged (transaction IDs only, no amounts or accounts)."""
        try:
            self.log.record([self.shadow(s) for s in scored])
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

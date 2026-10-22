"""FraudGraph API (A7): scoring, stored decisions, review cases, labels, monitoring.

Run: `uv run uvicorn services.api.main:app` (bundle from `FG_BUNDLE_DIR`, or the newest
bundle under `artifacts/bundles/`). Backends come from `FG_API__*` settings: Redis for
feature state (else in process), Postgres for the decision log (else in-memory SQLite).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraudgraph.config import ApiSettings, get_settings
from fraudgraph.features.redis_store import RedisStore
from fraudgraph.features.state import StateStore
from fraudgraph.serving.scorer import Scorer
from services.api.db import DecisionLog, make_engine
from services.api.deps import AppContext, make_limiter
from services.api.observability import LatencyMiddleware, configure_logging
from services.api.routers import cases, labels, models, monitoring, score, sim, transactions
from services.api.schemas import Health, Ready
from services.api.security import KEY_HEADER, ApiError, ApiKeys, BodyLimit, install_redaction
from services.api.ws import Broadcaster

log = logging.getLogger("fraudgraph.api")

TAGS = [
    {"name": "scoring", "description": "Score transactions (write scope)."},
    {"name": "transactions", "description": "Stored decisions, explanations, ego-graphs."},
    {"name": "cases", "description": "Review queue and analyst verdicts."},
    {"name": "labels", "description": "Late labels: chargebacks and the replayer."},
    {"name": "monitoring", "description": "Drift, alarms and live performance."},
    {"name": "models", "description": "Model versions and the champion."},
    {"name": "simulation", "description": "Replayer and drift injection (admin, demo)."},
    {"name": "ops", "description": "Liveness and readiness."},
]
HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
}


def latest_bundle(root: Path) -> Path | None:
    """Newest bundle by its manifest's `created_at`."""
    candidates = []
    for manifest in root.glob("*/manifest.json"):
        created = json.loads(manifest.read_text(encoding="utf-8")).get("created_at", "")
        candidates.append((created, manifest.parent))
    return max(candidates)[1] if candidates else None


def resolve_bundle(log: DecisionLog | None = None) -> Path | None:
    """`FG_BUNDLE_DIR`, else the champion recorded in the decision log (a promotion
    survives a restart; the worker's newer challenger bundles are not served), else the
    newest bundle on disk."""
    configured = os.environ.get("FG_BUNDLE_DIR")
    if configured:
        return Path(configured)
    root = get_settings().paths.artifacts_dir / "bundles"
    if log is not None:
        try:
            champions = [m for m in log.models() if m.stage == "champion"]
        except Exception:  # no schema yet: fall back to the newest bundle
            champions = []
        if champions and (root / champions[0].bundle_uri / "manifest.json").is_file():
            return root / champions[0].bundle_uri
    return latest_bundle(root)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    bundle_dir: Path | None = None,
    settings: ApiSettings | None = None,
    *,
    store: StateStore | None = None,
) -> FastAPI:
    """Build the app. `store` overrides the configured feature state (tests)."""
    settings = settings or get_settings().api
    if settings.json_logs:
        configure_logging()
    install_redaction()
    app = FastAPI(
        title="FraudGraph API",
        version="0.7.0",
        openapi_tags=TAGS,
        description=(
            "Real-time fraud scoring with reasons and uncertainty. Authenticate with an "
            f"`{KEY_HEADER}` header (scopes read < write < admin). Errors are always "
            '`{"error": {"code", "message"}}`.'
        ),
    )
    ctx = AppContext(
        settings=settings,
        keys=ApiKeys.parse(settings.api_keys, enabled=settings.auth_enabled),
        log=DecisionLog(make_engine(settings.database_url)),
        broadcaster=Broadcaster(),
        reports_dir=get_settings().paths.reports_dir,
    )
    if not settings.auth_enabled:
        log.warning("API authentication is disabled (FG_API__AUTH_ENABLED=false)")
    app.state.ctx = ctx

    path = bundle_dir or resolve_bundle(ctx.log)
    ctx.bundles_root = path.parent if path is not None else None
    try:
        if path is None:
            raise FileNotFoundError("no bundle found; set FG_BUNDLE_DIR or run `fgx bundle`")
        if store is None and settings.redis_url:
            spec = json.loads((path / "feature_spec.json").read_text(encoding="utf-8"))
            ctx.redis = RedisStore.from_url(
                settings.redis_url, spec["horizon_steps"], prefix=settings.redis_prefix
            )
        ctx.scorer = Scorer(path, store=store or ctx.redis)
    except Exception as exc:  # the service stays up and reports not ready
        ctx.detail = f"{type(exc).__name__}: {exc}"
    if ctx.scorer is not None:
        try:
            ctx.log.register_model(
                ctx.scorer.version,
                ctx.scorer.manifest.get("metrics", {}),
                path.name if path else "",
            )
        except Exception as exc:  # e.g. migrations not applied yet; /ready says so
            log.warning("could not register the model version: %s", type(exc).__name__)
        ctx.metrics.set_model(ctx.scorer.version)
        ctx.refresh_challengers(force=True)

    _middleware(app, settings)
    app.add_middleware(LatencyMiddleware, metrics=ctx.metrics)  # outermost: times everything
    _handlers(app)
    limiter = make_limiter(settings)
    app.state.limiter = limiter
    for module in (score, transactions, cases, labels, monitoring, models, sim):
        app.include_router(module.build(limiter, ctx))
    _ops_routes(app, ctx)
    return app


def _middleware(app: FastAPI, settings: ApiSettings) -> None:
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if "*" in origins:
        raise ValueError("FG_API__CORS_ORIGINS must list origins explicitly, not '*'")
    if origins:  # explicit allow-list only
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", KEY_HEADER.lower()],
            max_age=600,
        )
    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)


def _handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid')}")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTP_CODES.get(exc.status_code, "http_error")
        return error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RateLimitExceeded)
    async def _rate(_: Request, exc: RateLimitExceeded) -> JSONResponse:
        return error(429, "rate_limited", f"rate limit exceeded: {exc.detail}")


def _ops_routes(app: FastAPI, ctx: AppContext) -> None:
    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        return Health(status="ok")

    @app.get("/ready", response_model=Ready, tags=["ops"], responses={503: {"model": Ready}})
    def ready() -> JSONResponse | Ready:
        """Ready once the bundle is loaded and Redis and the database answer."""
        checks = {"model": ctx.scorer is not None}
        detail = ctx.detail
        for name, probe in (("redis", ctx.redis.ping if ctx.redis else None), ("db", ctx.log.ping)):
            if probe is None:
                continue
            try:
                checks[name] = bool(probe())
            except Exception as exc:
                checks[name] = False
                detail = detail or f"{name}: {type(exc).__name__}"
        if all(checks.values()) and ctx.scorer is not None:
            return Ready(status="ready", model_version=ctx.scorer.version, checks=checks)
        body = Ready(status="not_ready", detail=str(detail), checks=checks)
        return JSONResponse(status_code=503, content=body.model_dump())

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus exposition. Not routed by Caddy: scraped on the internal network."""
        try:
            open_cases, alarms = ctx.log.open_cases(), ctx.log.alarm_counts()
        except Exception:  # the counters still render with the database down
            open_cases, alarms = 0, {}
        body = ctx.metrics.render(open_cases, alarms)
        return Response(body, media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        """Every decision as it is made. Browsers cannot set headers on a WebSocket, so
        the read key may also come as `?api_key=` (redacted from logs)."""
        key = websocket.headers.get(KEY_HEADER) or websocket.query_params.get("api_key")
        try:
            ctx.keys.authenticate(key, "read")
        except ApiError as exc:  # accept, then close: the client sees code and reason
            await websocket.accept()
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=exc.code)
            return
        queue = ctx.broadcaster.subscribe()
        await websocket.accept()

        async def pump() -> None:
            while True:
                await websocket.send_json(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            while True:  # only to notice the client leaving
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            ctx.broadcaster.unsubscribe(queue)


app = create_app()

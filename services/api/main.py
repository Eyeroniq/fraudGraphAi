"""Scoring API (task 2.2): `POST /v1/score`, `GET /health`, `GET /ready`.

Run: `uv run uvicorn services.api.main:app` (bundle from `FG_BUNDLE_DIR`, or the newest
bundle under `artifacts/bundles/`). State is in memory for now; Redis arrives in 7.1.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fraudgraph.config import get_settings
from fraudgraph.features.spec import Event
from fraudgraph.features.state import OutOfOrderError
from fraudgraph.serving.scorer import Scorer
from services.api.schemas import Health, Ready, ScoreRequest, ScoreResponse

IDEMPOTENCY_CACHE = 100_000
# Local web dev server (`npm run dev`). In compose the page is same-origin via Caddy (D9).
DEFAULT_CORS_ORIGINS = "http://localhost:3000"


def latest_bundle(root: Path) -> Path | None:
    """Newest bundle by its manifest's `created_at`."""
    candidates = []
    for manifest in root.glob("*/manifest.json"):
        created = json.loads(manifest.read_text(encoding="utf-8")).get("created_at", "")
        candidates.append((created, manifest.parent))
    return max(candidates)[1] if candidates else None


def resolve_bundle() -> Path | None:
    configured = os.environ.get("FG_BUNDLE_DIR")
    if configured:
        return Path(configured)
    return latest_bundle(get_settings().paths.artifacts_dir / "bundles")


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(bundle_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="FraudGraph API", version="0.2.0")
    origins = [
        o.strip()
        for o in os.environ.get("FG_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
        if o.strip()
    ]
    if origins:  # explicit allow-list only, never "*"
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )
    state: dict[str, object] = {"scorer": None, "detail": None}
    lock = threading.Lock()  # in-memory state is not thread-safe; one writer at a time
    seen: OrderedDict[str, ScoreResponse] = OrderedDict()

    path = bundle_dir or resolve_bundle()
    try:
        if path is None:
            raise FileNotFoundError("no bundle found; set FG_BUNDLE_DIR or run `fgx bundle`")
        state["scorer"] = Scorer(path)
    except Exception as exc:  # the service stays up and reports not ready
        state["detail"] = f"{type(exc).__name__}: {exc}"

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid')}")

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        return Health(status="ok")

    @app.get("/ready", response_model=Ready, tags=["ops"])
    def ready() -> JSONResponse | Ready:
        scorer = state["scorer"]
        if isinstance(scorer, Scorer):
            return Ready(status="ready", model_version=scorer.version)
        body = Ready(status="not_ready", detail=str(state["detail"]))
        return JSONResponse(status_code=503, content=body.model_dump())

    @app.post(
        "/v1/score",
        response_model=ScoreResponse,
        tags=["scoring"],
        responses={409: {"description": "Out-of-order step"}, 503: {"description": "No model"}},
    )
    def score(request: ScoreRequest) -> JSONResponse | ScoreResponse:
        scorer = state["scorer"]
        if not isinstance(scorer, Scorer):
            return error(503, "not_ready", str(state["detail"]))
        with lock:
            if request.txn_id in seen:  # idempotent: same answer, state not touched twice
                return seen[request.txn_id]
            event = Event(
                txn_id=request.txn_id,
                step=request.step,
                type=request.type,
                amount=request.amount,
                name_orig=request.name_orig,
                name_dest=request.name_dest,
            )
            try:
                decision = scorer.score(event)
            except OutOfOrderError as exc:
                return error(409, "out_of_order", str(exc))
            response = ScoreResponse(**decision.__dict__)
            seen[request.txn_id] = response
            if len(seen) > IDEMPOTENCY_CACHE:
                seen.popitem(last=False)
        return response

    return app


app = create_app()

"""Observability (task 7.6): JSON logs and Prometheus metrics.

- Logs go through the standard library, so the redaction filters (7.7) still apply, and
  are rendered as one JSON object per line by structlog. Decision logs carry `txn_id`,
  `model_version`, the decision and latency, never amounts or account IDs.
- `/metrics` (scraped on the internal network, not routed by Caddy): request latency per
  route, decisions by type, shadow decisions per challenger, the model version, and the
  review queue size and drift alarms read from the decision log at scrape time.
  With several uvicorn workers each process exposes its own counters.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5)
SHARED: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]
# structlog always hands records to the standard library, so its levels and the
# redaction filters apply; `configure_logging` decides how they are rendered.
structlog.configure(
    processors=[*SHARED, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def configure_logging(level: str = "INFO") -> None:
    """Render every stdlib log record as JSON (idempotent)."""
    shared = SHARED
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_fraudgraph", False):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler._fraudgraph = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):  # one format, one handler
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True


class Metrics:
    """One registry per app, so tests can build several apps in one process."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.latency = Histogram(
            "fraudgraph_request_seconds",
            "Request latency by route",
            ["method", "route", "status"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "fraudgraph_decisions",
            "Champion decisions by action",
            ["decision"],
            registry=self.registry,
        )
        self.shadow = Counter(
            "fraudgraph_shadow_decisions",
            "Challenger decisions by version and action",
            ["model_version", "decision"],
            registry=self.registry,
        )
        self.model = Gauge(
            "fraudgraph_model_info",
            "Champion model version being served (value 1)",
            ["model_version"],
            registry=self.registry,
        )
        self.review_queue = Gauge(
            "fraudgraph_review_queue", "Open review cases", registry=self.registry
        )
        self.alarms = Gauge(
            "fraudgraph_drift_alarms",
            "Drift alarms recorded by the worker, by kind",
            ["kind"],
            registry=self.registry,
        )

    def set_model(self, version: str) -> None:
        self.model.clear()
        self.model.labels(model_version=version).set(1)

    def render(self, open_cases: int, alarms: dict[str, int]) -> bytes:
        self.review_queue.set(open_cases)
        for kind in ("input", "prediction", "performance"):
            self.alarms.labels(kind=kind).set(alarms.get(kind, 0))
        return generate_latest(self.registry)


class LatencyMiddleware:
    """Pure ASGI timing (no request body buffering). The label is the route template,
    e.g. `/v1/transactions/{txn_id}`, so IDs never become label values."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def sending(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, sending)
        finally:
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            self.metrics.latency.labels(
                method=scope["method"], route=path, status=str(status)
            ).observe(time.perf_counter() - started)

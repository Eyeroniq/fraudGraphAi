"""FraudGraph worker (task 7.4): `python -m services.worker.main`.

Runs the jobs in `services/worker/jobs.py` every `FG_WORKER__TICK_SECONDS`. It reads and
writes the decision log directly (same `FG_API__DATABASE_URL`) and talks to the API only
to replay traffic (write key) and to promote a winning challenger (admin key). New
challenger bundles go next to the champion's, under the bundle root the API reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

from fraudgraph.config import get_settings
from services.api.db import DecisionLog, make_engine
from services.api.observability import configure_logging
from services.api.security import KEY_HEADER
from services.worker.jobs import Worker

log = logging.getLogger("fraudgraph.worker")


class HttpApi:
    """Minimal JSON client: one key per scope, errors returned, not raised."""

    def __init__(self, base: str, keys: dict[str, str], timeout: float = 60.0) -> None:
        self.base, self.keys, self.timeout = base.rstrip("/"), keys, timeout

    def __call__(self, path: str, body: Any, scope: str) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method="POST",
            headers={"content-type": "application/json", KEY_HEADER: self.keys.get(scope, "")},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")


def bundles_root() -> Path:
    configured = os.environ.get("FG_BUNDLE_DIR")
    if configured:
        return Path(configured).parent
    return get_settings().paths.artifacts_dir / "bundles"


def build_worker() -> Worker:
    settings = get_settings()
    ws = settings.worker
    events = None
    if ws.events_path is not None and ws.events_path.is_file():
        events = pl.read_parquet(ws.events_path)
    return Worker(
        log=DecisionLog(make_engine(settings.api.database_url)),
        settings=ws,
        bundles_root=bundles_root(),
        reports_dir=settings.paths.reports_dir,
        post=HttpApi(ws.api_url, {"write": ws.write_key, "admin": ws.admin_key}),
        events=events,
        seed=settings.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudGraph background worker")
    parser.add_argument("--once", action="store_true", help="run one tick and exit")
    args = parser.parse_args(argv)
    if get_settings().api.json_logs:
        configure_logging()
    else:
        logging.basicConfig(level=logging.INFO)
    worker = build_worker()
    stopping = False

    def stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("worker started (bundles in %s)", worker.bundles_root)
    failed = False
    while not stopping:
        started = time.monotonic()
        try:
            done = worker.tick()
            failed = False
            if any(done.get(k) for k in ("replayed", "drift_windows", "retrain", "judged")):
                log.info("tick: %s", json.dumps(done, default=str))
        except Exception:  # one bad tick must not stop the worker
            failed = True
            log.exception("worker tick failed")
        if args.once:
            break
        time.sleep(max(0.0, worker.settings.tick_seconds - (time.monotonic() - started)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

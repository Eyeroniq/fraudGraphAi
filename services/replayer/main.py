"""Replayer service (task 6.1): stream PaySim events into the scoring API at a set speed.

    python -m services.replayer.main --base http://localhost:8000 --first 356 --last 380 \
        --steps-per-second 2

Each step's events go to `POST /v1/score`; the delayed labels that become known at that
step come from `fraudgraph.online.replay.label_schedule`, the same schedule the offline
drift study uses. The API has no label endpoint until task 7.3 (`/v1/labels`), so labels
are counted and, when `--labels-path` is set, posted there (a 404 is tolerated).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from fraudgraph.online.replay import LabelConfig, label_schedule, replay

Post = Callable[[str, dict[str, Any]], tuple[int, dict[str, Any]]]
EVENT_FIELDS = ("txn_id", "step", "type", "amount", "name_orig", "name_dest")


def http_post(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def run(
    frame: pl.DataFrame,
    base: str,
    *,
    first_step: int,
    last_step: int,
    steps_per_second: float = 0.0,
    labels: LabelConfig | None = None,
    labels_path: str | None = None,
    post: Post = http_post,
) -> dict[str, Any]:
    """Replay [first_step, last_step]; returns counts of decisions, errors and labels."""
    scheduled = label_schedule(frame, labels or LabelConfig())
    decisions: Counter[str] = Counter()
    errors: Counter[int] = Counter()
    sent_labels = 0
    started = time.monotonic()
    for k, batch in enumerate(replay(scheduled, first_step=first_step, last_step=last_step)):
        for event in batch.events.select(EVENT_FIELDS).iter_rows(named=True):
            status, body = post(f"{base}/v1/score", event)
            if status == 200:
                decisions[str(body.get("decision"))] += 1
            else:
                errors[status] += 1
        for row in batch.labels.select("txn_id", "observed_fraud").iter_rows(named=True):
            sent_labels += 1
            if labels_path:
                post(f"{base}{labels_path}", {**row, "step": batch.step})
        if steps_per_second > 0:  # pace by step, not by event
            wait = started + (k + 1) / steps_per_second - time.monotonic()
            if wait > 0:
                time.sleep(wait)
    return {
        "steps": last_step - first_step + 1,
        "scored": sum(decisions.values()),
        "decisions": dict(decisions),
        "errors": {str(k): v for k, v in errors.items()},
        "labels": sent_labels,
        "seconds": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--events", type=Path, default=Path("data/processed/paysim.parquet"))
    parser.add_argument("--first", type=int, default=356)
    parser.add_argument("--last", type=int, default=360)
    parser.add_argument("--steps-per-second", type=float, default=1.0)
    parser.add_argument("--labels-path", default=None, help="e.g. /v1/labels (task 7.3)")
    args = parser.parse_args(argv)
    frame = pl.read_parquet(args.events).filter(pl.col("step") <= args.last)
    summary = run(
        frame,
        args.base,
        first_step=args.first,
        last_step=args.last,
        steps_per_second=args.steps_per_second,
        labels_path=args.labels_path,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""Open-loop load test for `POST /v1/score` (task 7.6).

    python scripts/load_test.py --base http://localhost:8000 --rps 200 --seconds 60 --key fg_...

Requests are sent on a fixed schedule (request i at t0 + i / rps) whatever the server's
speed, so a slow server shows up as latency instead of quietly lowering the load (no
coordinated omission). Transactions come from the synthetic fixture in step order, each
with a fresh txn_id. Prints p50 / p95 / p99 latency, achieved rate and errors as JSON;
exits non-zero if p95 is over `--p95-ms`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import httpx
import numpy as np

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate

FIELDS = ("step", "type", "amount", "name_orig", "name_dest")


def payloads(n: int, step: int) -> list[dict[str, Any]]:
    """Fixture payments, all at one step: concurrent requests arrive slightly out of order,
    and the API rejects a step older than the newest it has scored (409)."""
    rows = to_canonical(generate().frame).select(FIELDS).to_dicts()
    run = int(time.time())
    return [{**rows[i % len(rows)], "step": step, "txn_id": f"L{run}-{i}"} for i in range(n)]


async def run(base: str, key: str, rps: float, seconds: float, step: int) -> dict[str, Any]:
    bodies = payloads(int(rps * seconds), step)
    latencies: list[float] = []
    statuses: dict[str, int] = {}
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    async with httpx.AsyncClient(
        base_url=base, headers={"X-API-Key": key}, limits=limits
    ) as client:

        async def one(body: dict[str, Any]) -> None:
            started = time.perf_counter()
            try:
                response = await client.post("/v1/score", json=body, timeout=10)
                code = str(response.status_code)
            except httpx.HTTPError as exc:
                code = type(exc).__name__
            latencies.append(time.perf_counter() - started)
            statuses[code] = statuses.get(code, 0) + 1

        t0 = time.perf_counter()
        tasks = []
        for i, body in enumerate(bodies):
            delay = t0 + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(body)))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0
    ms = np.array(latencies) * 1_000
    return {
        "target_rps": rps,
        "achieved_rps": round(len(bodies) / elapsed, 1),
        "requests": len(bodies),
        "statuses": statuses,
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "p99_ms": round(float(np.percentile(ms, 99)), 1),
        "max_ms": round(float(ms.max()), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--key", required=True, help="an API key with write scope")
    parser.add_argument("--rps", type=float, default=200.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--p95-ms", type=float, default=50.0)
    parser.add_argument("--step", type=int, default=700, help="step of every payment")
    args = parser.parse_args(argv)
    summary = asyncio.run(run(args.base, args.key, args.rps, args.seconds, args.step))
    print(json.dumps(summary, indent=2))
    ok = summary["statuses"].get("200", 0) == summary["requests"]
    return 0 if ok and summary["p95_ms"] <= args.p95_ms else 1


if __name__ == "__main__":
    sys.exit(main())

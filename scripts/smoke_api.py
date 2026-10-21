"""Smoke test a running API: wait for /ready, then score three fixture transactions.

Usage: python scripts/smoke_api.py http://localhost:8000
Exits non-zero on any failure. Used by the CI docker-smoke job.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

EVENTS = [
    {"txn_id": "S1", "step": 30, "type": "PAYMENT", "amount": 9839.64,
     "name_orig": "C1231006815", "name_dest": "M1979787155"},
    {"txn_id": "S2", "step": 31, "type": "TRANSFER", "amount": 181.0,
     "name_orig": "C1305486145", "name_dest": "C553264065"},
    {"txn_id": "S3", "step": 32, "type": "CASH_OUT", "amount": 229133.94,
     "name_orig": "C905080434", "name_dest": "C476402209"},
]  # fmt: skip


def call(url: str, body: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main(base: str, timeout_s: float = 120.0) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            status, body = call(f"{base}/ready")
            if status == 200:
                print(f"ready: {body}")
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            print("API never became ready", file=sys.stderr)
            return 1
        time.sleep(2)

    for event in EVENTS:
        status, body = call(f"{base}/v1/score", event)
        print(f"{event['txn_id']}: {status} {body}")
        if status != 200 or body.get("decision") not in {"APPROVE", "REVIEW", "BLOCK"}:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))

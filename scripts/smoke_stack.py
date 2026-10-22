"""Smoke-test the running compose stack through Caddy, as a browser would (task 9.3).

    python scripts/smoke_stack.py http://localhost:8080

Checks: the page is served; /api/ready is ready; /api/metrics is NOT public; payments are
scored; a held payment becomes a case; resolving it creates a label; the worker is up
(it wrote nothing yet, but /api/v1/sim/status answers through the shared database).
Exits non-zero on the first failure.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def call(url: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read()
            kind = response.headers.get("content-type", "")
            return response.status, json.loads(text) if "json" in kind else text.decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        sys.exit(1)


def main(base: str) -> int:
    status, page = call(f"{base}/")
    check(status == 200 and "FRAUD" in str(page), "console page served")
    status, ready = call(f"{base}/api/ready")
    check(status == 200 and ready.get("status") == "ready", f"api ready through Caddy: {ready}")
    status, _ = call(f"{base}/api/metrics")
    check(status == 404, "metrics are not public")
    status, sim = call(f"{base}/api/v1/sim/status")
    check(status == 200 and "running" in sim, "sim status answers")

    # Large transfers to fresh receivers: some are held for review.
    case = None
    for i in range(40):
        body = {
            "txn_id": f"SMOKE{int(time.time())}{i}",
            "step": 100 + i,
            "type": "TRANSFER",
            "amount": 900_000.0 + i,
            "name_orig": f"C55{i:04d}",
            "name_dest": f"C66{i:04d}",
        }
        status, decision = call(f"{base}/api/v1/score", body)
        check(
            status == 200,
            f"score {body['txn_id']} -> {decision.get('decision') if status == 200 else decision}",
        )
    for _ in range(30):  # decisions are written after the response
        status, page_ = call(f"{base}/api/v1/cases?status=open&limit=1")
        if status == 200 and page_["items"]:
            case = page_["items"][0]
            break
        time.sleep(1)
    check(case is not None, "a held payment became a review case")
    assert case is not None
    status, resolved = call(f"{base}/api/v1/cases/{case['case_id']}/resolve", {"verdict": "fraud"})
    check(status == 200 and resolved["case"]["status"] == "resolved", "case resolved")
    status, detail = call(f"{base}/api/v1/transactions/{case['txn_id']}")
    check(status == 200 and detail["label"]["is_fraud"] is True, "resolving created a fraud label")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"))

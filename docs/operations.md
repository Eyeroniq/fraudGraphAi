# Operations: worker, champion/challenger, observability

Phase 7.4–7.6 and Juspay J4. The services:

| Service | Runs | Talks to |
|---|---|---|
| `api` (`services/api`) | scoring, decision log, review cases, labels, monitoring reads, promotion | Redis (feature state), Postgres (decision log) |
| `worker` (`services/worker`) | replayer, drift and performance monitors, daily retrain, challenger judging | Postgres directly; the API for replayed traffic (write key) and promotion (admin key) |
| `clients/rust` | `POST /v1/score` from a payment switch's pre-authorisation hook | the API |

## Worker (7.4)

`python -m services.worker.main` (`uv run poe worker`, or the `worker` service in
`deploy/docker-compose.yml`) runs one tick every `FG_WORKER__TICK_SECONDS`. A tick does
whatever has become due:

- **Replayer (demo).** `POST /v1/sim/start {first_step, last_step, steps_per_tick, scenario}`
  makes the worker stream PaySim events (`FG_WORKER__EVENTS_PATH`) into
  `/v1/score/batch`, step by step. It also posts each step's labels to
  `/v1/labels/batch` once they are due: 24 ± 6 steps late, and 10 % of frauds never
  reported, as in [drift_study.md](drift_study.md).
  - `POST /v1/sim/scenario {"name": "D3"}` injects a drift scenario (D0–D6) from the next
    replayed step. `/v1/sim/stop` pauses and `/v1/sim/status` reports the state.
  - Progress is stored after every step (`sim_control`), so a restarted worker resumes.
  - The replayer backs off when the API rate-limits it.
- **Drift monitor.** The reference is the stream's first two days. Each later full day
  (24 steps) is compared with it once a later step has been scored:
  - per-feature PSI and KS or χ² with Benjamini–Hochberg on the model matrix;
  - score PSI and the alert-rate z-score.
  - Results go to `drift_windows`, one row per feature plus a `__score__` row, and alarms
    go to `alarms`. The thresholds are the ones tuned on D0 in the drift study.
  - An Evidently report per window goes to `reports/evidently/window-<a>-<b>.html|json`
    (served by `/v1/monitoring/reports/{id}`) when Evidently is installed. The slim worker
    image skips it.
- **Performance monitor.** Every fraud label the champion approved is a miss. ADWIN and
  Page-Hinkley watch the miss stream: this is the monitor that saw every damaging drift
  in the study. Alarms go to `alarms` with kind `performance`.
- **Daily retrain (the 6.6 policy).**
  - Every 24 steps, LightGBM is refit with the champion's parameters on the labels known
    from the last 168 steps. Legit rows are subsampled to 20 % and weighted ×5.
  - The newest 24 labelled steps are held out to calibrate it (isotonic) and to
    re-conformalise it at the champion's α.
  - The result is a normal hashed bundle (`<model>-r<step>-<date>-<sha>`), written next to
    the champion's and registered as the only challenger.
  - It is skipped when there are fewer than `min_fraud_labels` fraud labels to fit on, or
    fewer than 5 to calibrate.
- **Judge.** Once a challenger has `min_shadow_labels` labelled shadow decisions, its
  realised cost (Phase 4 costs, true labels) is compared with the champion's on the same
  transactions. It is promoted through the API if it is lower, otherwise archived.

The river online learner from the roadmap is not run. In the study it had the highest
cost on all seven scenarios, and the shadow switch never picked it.

Monitor state is kept in memory. After a restart the performance monitor starts fresh at
the newest step rather than raising old alarms again. The drift monitor continues from
the last stored window.

## Champion and challenger (7.5)

- **Shadow scoring.** The API loads every bundle registered as a challenger, checking at
  most every `FG_API__CHALLENGER_REFRESH_S` seconds. Challengers score each transaction
  on the champion's own features, in the background task that writes the decision log,
  so they add nothing to `/v1/score` latency. They don't read feature state, and they
  skip explanations.
- **Storage and comparison.** Shadow decisions go to `shadow_predictions`.
  `/v1/monitoring/performance` compares champion and challengers on the same labelled
  transactions.
- **Promotion.** `POST /v1/models/{version}/promote` (admin) loads and verifies the new
  bundle first (hashes, engine version, feature list). It then swaps it in under the
  scoring lock, so a request uses either the old model or the new one, never a mix.
  - Feature state carries over: the engine version guard makes that safe.
  - The old champion is archived, and the `fraudgraph_model_info` gauge changes.
  - The daily review capacity counter restarts with the new policy object.

On the synthetic fixture (`tests/unit/test_worker.py`), the replayer streams 150 steps
with D3 mule rings from step 97. The worker writes day windows, raises alarms after the
rings appear, registers a challenger every day, and promotes or archives each one by the
cost rule. In a longer run (190 steps), the first retrain after the rings cut realised
cost on shadow traffic from 1.22 M to 0.11 M and was promoted.

## Observability (7.6)

- **Logs.** One JSON object per line when `FG_API__JSON_LOGS=true` (set in the image).
  Every decision logs `txn_id`, `model_version`, `decision`, `decision_reason` and
  `latency_ms`, never amounts or account IDs, and the 7.7 redaction filters still apply.
- **Metrics.** `GET /metrics` serves Prometheus metrics on the internal network; Caddy
  does not route it.

| Metric | Type | Labels |
|---|---|---|
| `fraudgraph_request_seconds` | histogram | method, route template, status |
| `fraudgraph_decisions_total` | counter | decision |
| `fraudgraph_shadow_decisions_total` | counter | model_version, decision |
| `fraudgraph_model_info` | gauge (1) | model_version |
| `fraudgraph_review_queue` | gauge | — (open cases, read at scrape time) |
| `fraudgraph_drift_alarms` | gauge | kind (input, prediction, performance) |

With several uvicorn workers, each process exposes its own counters.

### Load test

`uv run poe loadtest --key <write key> --rps 200 --seconds 60` is an open-loop load test:
requests go out on a fixed schedule whatever the server's speed, so overload shows up as
latency. It prints p50/p95/p99 and exits non-zero if p95 is over `--p95-ms` (50).

Results on the development laptop (Windows 11, i5-13420H, 12 threads), with the PaySim
champion `lgbm_g1`, 2 uvicorn workers, in-process feature state, the in-memory SQLite
fallback and the generator on the same machine:

| Target rps | Achieved | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 50 | 49.9 | 14.6 ms | 32.3 ms | 192 ms | 0 |
| 100 | 99.9 | 18.8 ms | **47.8 ms** | 169 ms | 0 |
| 150 | overloaded | 28.7 s | 57.7 s | — | 1 read error |

So locally the budget (p95 < 50 ms) holds at 100 rps on 2 workers, and the 200 rps target
is **not met on this machine**.

Two changes made during this task roughly halved the cost per request:
- LightGBM single-row predictions with one thread: 4.7 ms → 0.4 ms. A one-row call
  otherwise starts the whole OpenMP pool.
- One cached `INSERT … ON CONFLICT` statement for the decision log, instead of a
  multi-row statement compiled per call: 9.3 ms → 3.8 ms.

The remaining cost per request is the framework (validation, rate limiter, middleware) and
the SQLite fallback, which serialises all sessions.

The target is meant for the deployment stack: Linux, Postgres, Redis. CI runs the same
test against that stack in the `docker smoke` job, at 200 rps for 30 s on 2 workers. It
reports the numbers in the job summary but does not fail the build: a shared runner also
runs the load generator.

## Rust client and FRM hook (J4)

`clients/rust` is a blocking client (`ureq` and `serde`) for `/v1/score` and
`/v1/labels`, with typed requests, responses and the API's error model. It has a 200 ms
default timeout because it sits on the payment path.

`frm::PreAuthHook` is a pre-authorisation fraud-risk check in the style of a Hyperswitch
FRM connector. It maps a decision to the switch's action:

| FraudGraph | Switch action |
|---|---|
| `APPROVE` | continue |
| `STEP_UP` | 3DS / OTP challenge |
| `REVIEW` | hold for manual review |
| `BLOCK` | cancel |

When the API can't be reached in time it applies an explicit fail mode: *open* continues
the payment, *closed* holds it for review. The verdict says it was not scored.

```bash
cd clients/rust
cargo test
FRAUDGRAPH_URL=http://localhost:8000 FRAUDGRAPH_KEY=<write key> cargo run --example frm_hook
```

The CI job `rust client (J4)` runs `cargo fmt --check`, `clippy -D warnings` and the
tests. The tests use a mock server, so no API is needed.

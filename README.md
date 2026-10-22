# FraudGraph

A **pre-authorisation risk service for a payment switch**: every payment gets a decision —
approve, **step-up** (OTP / 3DS challenge), send to a human, or block — chosen by expected
cost in money, in about a millisecond. It says when it is unsure (conformal prediction sets),
uses the payment graph (is the receiving account new, how many new senders hit it today),
and is built to learn from delayed labels and notice drift.

| | |
|---|---|
| Latency | 0.12 ms p95 per decision (features + model, CPU); ~1 ms through the API |
| Graph features | test cost halves vs the tabular baseline (133 M → 66 M), [graph study](docs/graph_study.md) |
| Full policy | 66 M on test vs 154 M for a single global threshold, [conformal + policy](docs/conformal.md) |
| Guarantee | 80 % fraud-class coverage target (0.805 on test, 0.79–0.87 per day), conformal sets |

Data: PaySim (simulated mobile money); costs are in its currency units — read them as ₹ for
an Indian deployment. Honest limits are in [the data audit](docs/data_audit.md) and the
docs above. Build plan and architecture: [ROADMAP.md](ROADMAP.md).

## Development setup

Works the same on Windows, macOS and Linux. You need [uv](https://docs.astral.sh/uv/)
(Windows: `winget install astral-sh.uv`) and, from Phase 2 on, Docker.

```bash
uv run poe setup   # Python 3.12 env from uv.lock + git pre-commit hook
uv run poe lint    # ruff, format check, mypy
uv run poe test    # pytest
uv run poe check   # lint + test
uv run fgx --help  # pipeline CLI
```

`uv run poe` with no task lists every task.

## Data

FraudGraph uses the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) dataset (~470 MB).

```bash
# with a Kaggle API token in KAGGLE_USERNAME / KAGGLE_KEY (or ~/.kaggle/kaggle.json)
uv run poe data
# or, after downloading the zip from Kaggle in your browser
uv run fgx data download --from-file path/to/archive.zip
uv run fgx data prepare
```

The first download prints the file's sha256. Pin it as `data.raw_sha256` in
`configs/base.yaml` so every later run is checked against it.

## Run the service

```bash
uv run fgx all --fixture --out artifacts/bundles   # a bundle from synthetic data (or `fgx bundle` after the real pipeline)
cp deploy/.env.example deploy/.env                 # fill in keys from `uv run fgx api-key`
uv run poe up                                      # API + Redis + Postgres + worker + web page; open http://localhost:8080
uv run poe down
```

Every `/v1` route needs an `X-API-Key` header (scopes read < write < admin); only the
keys' sha256 hashes are configured (`FG_API__API_KEYS`). The OpenAPI docs are at
`http://localhost:8000/docs`.

The worker monitors drift, retrains daily into a shadow challenger and promotes it when
it costs less on labelled traffic; `POST /v1/sim/start` (admin) replays PaySim through
the stack with a drift scenario. Details, metrics and the load test:
[docs/operations.md](docs/operations.md). A Rust client with a pre-authorisation
fraud-risk hook for a payment switch is in [clients/rust](clients/rust).

## Web console

The console (`web/`, Next.js static export behind Caddy) has seven pages:

- **Live:** the decision stream, decision mix and score distribution.
- **Queue:** held payments, most savings first.
- **Case:** an analyst's view. Reasons, SHAP waterfall, money graph, and F / L to label.
- **Lab:** PR curves and live cost sliders.
- **Drift:** the replayer, scenario injection, PSI heat map, alarms and the retraining study.
- **Explain:** explanation quality.
- **Model:** the model card and promotion.

```bash
cd web && NEXT_PUBLIC_MOCK=1 npm run dev           # demo data in the browser, no API needed
```

Against a local API instead:

1. Start it with `FG_API__AUTH_ENABLED=false uv run uvicorn services.api.main:app`. Feature state
   stays in process and the decision log in in-memory SQLite.
2. Run `uv run fgx console-report` once, for the Lab, Explain and drift-study panels.
3. In `web/`, run `NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev` and open
   http://localhost:3000.

The console's API types are generated from the OpenAPI schema:

```bash
uv run python scripts/export_openapi.py
```

Then run `npm run gen:api` in `web/`; CI fails if they drift. In `web/`, `npm test` runs the
unit tests. `NEXT_PUBLIC_MOCK=1 npm run build && npm run e2e` runs the Playwright flow and the
accessibility checks.

## Quality gates

These run on every pull request:

| Gate | What it checks |
|---|---|
| Lint | ruff and mypy (strict). |
| Tests | Ubuntu and Windows; coverage ≥ 80 % of `fraudgraph/`. |
| Integration | Redis and Postgres. |
| Model validation | `fgx validate`: data integrity, train/test validation and model evaluation conditions ([configs/validation.yaml](configs/validation.yaml)), and behaviour tests (invariance, directional, planted rings, AP floors, conformal coverage). |
| Web | API contract, types, unit tests, e2e and accessibility. |
| Rust client | Formatting, lints and tests. |
| Docker smoke | The full compose stack, driven through Caddy. |

On `main` and nightly, Playwright also runs against the live stack.

```bash
uv run fgx validate --fixture   # the gates on synthetic data; drop --fixture after the real pipeline
```

`full-pipeline.yml` reruns everything on PaySim. It is started by hand and needs Kaggle secrets.
A `v*` tag publishes images to GHCR (`release.yml`). See [deploy/README.md](deploy/README.md)
for the runbook and [docs/model_card.md](docs/model_card.md) for the model card.

## Configuration

Settings come from [configs/base.yaml](configs/base.yaml). Override any value with an
environment variable `FG_<FIELD>` (nested levels joined by `__`, e.g.
`FG_PATHS__DATA_DIR`), or point `FG_CONFIG` at another YAML file. `uv run fgx config show`
prints the effective settings.

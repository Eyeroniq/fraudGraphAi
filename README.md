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
uv run poe up                                      # API + Redis + Postgres + web page; open http://localhost:8080
uv run poe down
```

Every `/v1` route needs an `X-API-Key` header (scopes read < write < admin); only the
keys' sha256 hashes are configured (`FG_API__API_KEYS`). The OpenAPI docs are at
`http://localhost:8000/docs`.

For UI work: `FG_API__AUTH_ENABLED=false uv run uvicorn services.api.main:app` (feature
state in process, in-memory SQLite) and, in `web/`,
`NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev` (http://localhost:3000).

## Configuration

Settings come from [configs/base.yaml](configs/base.yaml). Override any value with an
environment variable `FG_<FIELD>` (nested levels joined by `__`, e.g.
`FG_PATHS__DATA_DIR`), or point `FG_CONFIG` at another YAML file. `uv run fgx config show`
prints the effective settings.

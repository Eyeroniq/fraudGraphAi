# FraudGraph

Real-time fraud detection for payment transactions: it scores each transaction, says why,
says when it is unsure, learns from new labels as a stream and notices when fraud patterns
change. The build plan and architecture are in [ROADMAP.md](ROADMAP.md).

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

## Configuration

Settings come from [configs/base.yaml](configs/base.yaml). Override any value with an
environment variable `FG_<FIELD>` (nested levels joined by `__`, e.g.
`FG_PATHS__DATA_DIR`), or point `FG_CONFIG` at another YAML file. `uv run fgx config show`
prints the effective settings.

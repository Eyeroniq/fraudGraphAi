"""Raw PaySim CSV -> validated, typed parquet sorted by step.

`txn_id` is assigned from the row's position in the raw file (`T0000000`, ...), so it
is stable across runs. Rows are sorted by `step` keeping file order within a step.
Running prepare twice on the same input gives the same file; when the output is
already up to date (same source sha256, same schema version) it is not rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandera.errors
import polars as pl

from fraudgraph.data.download import verify
from fraudgraph.data.schema import PAYSIM_SCHEMA, RAW_DTYPES, RENAME

SCHEMA_VERSION = 1
PARQUET_NAME = "paysim.parquet"
META_NAME = "paysim.meta.json"


class DataValidationError(ValueError):
    """The raw data breaks the PaySim contract."""


@dataclass(frozen=True)
class PrepareMeta:
    schema_version: int
    source_sha256: str
    rows: int
    fraud_rows: int
    min_step: int
    max_step: int


def prepare(
    raw_csv: Path,
    out_dir: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> tuple[Path, PrepareMeta, bool]:
    """Validate `raw_csv` and write `out_dir/paysim.parquet`.

    Returns the parquet path, its metadata, and whether it was (re)written.
    """
    source_sha = verify(raw_csv, expected_sha256).sha256
    parquet, meta_path = out_dir / PARQUET_NAME, out_dir / META_NAME

    if not force and parquet.exists() and meta_path.exists():
        existing = PrepareMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        if existing.source_sha256 == source_sha and existing.schema_version == SCHEMA_VERSION:
            return parquet, existing, False

    frame = to_canonical(read_raw(raw_csv))
    meta = PrepareMeta(
        schema_version=SCHEMA_VERSION,
        source_sha256=source_sha,
        rows=frame.height,
        fraud_rows=int(frame["is_fraud"].sum()),
        min_step=int(frame["step"].min()),  # type: ignore[arg-type]
        max_step=int(frame["step"].max()),  # type: ignore[arg-type]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, parquet)
    meta_path.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
    return parquet, meta, True


def read_raw(raw_csv: Path) -> pl.DataFrame:
    """Read the CSV with fixed types after checking the header is exactly PaySim's."""
    header = pl.read_csv(raw_csv, n_rows=0).columns
    missing = [c for c in RAW_DTYPES if c not in header]
    extra = [c for c in header if c not in RAW_DTYPES]
    if missing or extra:
        raise DataValidationError(f"unexpected CSV header: missing={missing} extra={extra}")
    try:
        return pl.read_csv(raw_csv, schema_overrides=RAW_DTYPES)
    except pl.exceptions.PolarsError as exc:  # e.g. text in a numeric column
        raise DataValidationError(f"cannot parse {raw_csv.name}: {exc}") from exc


def to_canonical(raw: pl.DataFrame) -> pl.DataFrame:
    """Rename, add `txn_id`, validate, cast flags to bool and sort by step."""
    frame = (
        raw.rename(RENAME)
        .with_row_index("row")
        .with_columns(
            txn_id=pl.concat_str(pl.lit("T"), pl.col("row").cast(pl.String).str.pad_start(7, "0"))
        )
        .drop("row")
        .select(["txn_id", *RENAME.values()])
    )
    try:
        PAYSIM_SCHEMA.validate(frame, lazy=True)
    except pandera.errors.SchemaErrors as exc:
        raise DataValidationError(_summarise(exc)) from exc
    return frame.with_columns(
        pl.col("is_fraud").cast(pl.Boolean),
        pl.col("is_flagged_fraud").cast(pl.Boolean),
    ).sort("step", maintain_order=True)


def _summarise(exc: pandera.errors.SchemaErrors) -> str:
    lines = ["PaySim schema check failed:"]
    for error in exc.schema_errors:
        lines.append(f"  - {error}")
    return "\n".join(lines[:21])

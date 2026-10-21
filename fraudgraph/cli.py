"""`fgx` command-line entry point. Each pipeline step (A3) becomes a sub-command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from fraudgraph import __version__
from fraudgraph.config import config_file, get_settings

app = typer.Typer(help="FraudGraph pipeline commands.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
data_app = typer.Typer(help="Download and prepare PaySim.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@config_app.command("show")
def config_show() -> None:
    """Print the effective settings (YAML + environment overrides) as JSON."""
    settings = get_settings()
    typer.echo(f"# source: {config_file()}", err=True)
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2))


@data_app.command("download")
def data_download(
    from_file: Annotated[
        Path | None,
        typer.Option(
            "--from-file",
            exists=True,
            dir_okay=False,
            help="Use a zip or CSV you downloaded yourself instead of the Kaggle API.",
        ),
    ] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing raw file.")] = False,
) -> None:
    """Fetch the raw PaySim CSV into data/raw and verify its sha256."""
    from fraudgraph.data.download import ChecksumMismatchError, MissingCredentialsError, fetch

    settings = get_settings()
    try:
        result = fetch(
            settings.paths.data_dir / "raw",
            settings.data.csv_name,
            dataset=settings.data.kaggle_dataset,
            expected_sha256=settings.data.raw_sha256,
            from_file=from_file,
            force=force,
        )
    except (ChecksumMismatchError, MissingCredentialsError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"raw file: {result.path}")
    typer.echo(f"sha256:   {result.sha256}")
    if not result.verified:
        typer.echo(
            "No checksum pinned yet. To lock this exact file, set data.raw_sha256 in "
            "configs/base.yaml to the value above.",
            err=True,
        )


@data_app.command("prepare")
def data_prepare(
    force: Annotated[bool, typer.Option(help="Rewrite even if up to date.")] = False,
) -> None:
    """Validate the raw CSV and write data/processed/paysim.parquet."""
    from fraudgraph.data.download import ChecksumMismatchError
    from fraudgraph.data.prepare import DataValidationError, prepare

    settings = get_settings()
    raw = settings.paths.data_dir / "raw" / settings.data.csv_name
    if not raw.is_file():
        typer.echo(f"error: {raw} not found; run `fgx data download` first.", err=True)
        raise typer.Exit(1)
    try:
        parquet, meta, written = prepare(
            raw,
            settings.paths.data_dir / "processed",
            expected_sha256=settings.data.raw_sha256,
            force=force,
        )
    except (DataValidationError, ChecksumMismatchError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    state = "written" if written else "already up to date"
    typer.echo(f"{parquet} ({state})")
    typer.echo(
        f"rows={meta.rows:,} fraud={meta.fraud_rows:,} steps={meta.min_step}..{meta.max_step}"
    )


@data_app.command("synth")
def data_synth(
    out: Annotated[Path, typer.Option(help="CSV to write (raw PaySim layout).")] = Path(
        "data/fixture/paysim_fixture.csv"
    ),
    seed: Annotated[int, typer.Option(help="Generator seed; same seed, same bytes.")] = 7,
) -> None:
    """Write the seeded synthetic PaySim-format fixture (with planted mule rings)."""
    from fraudgraph.data.download import sha256_of
    from fraudgraph.data.synth import SynthConfig, generate, write_csv

    result = generate(SynthConfig(seed=seed))
    path = write_csv(result, out)
    fraud = int(result.frame["isFraud"].sum())
    typer.echo(f"{path}: rows={result.frame.height:,} fraud={fraud} sha256={sha256_of(path)}")
    typer.echo(f"ring destinations: {', '.join(result.ring_destinations)}")


@data_app.command("audit")
def data_audit(
    probe: Annotated[
        bool, typer.Option(help="Also train quick LightGBM probes per column set (~5 min).")
    ] = True,
    figures_dir: Annotated[
        Path, typer.Option(help="Where to write the PNG figures for docs/data_audit.md.")
    ] = Path("docs/figures"),
) -> None:
    """Compute the data-audit numbers -> reports/audit/audit.json (+ figures)."""
    import polars as pl

    from fraudgraph.data.audit import probe_feature_sets, run_audit, write_figures
    from fraudgraph.data.prepare import PARQUET_NAME

    settings = get_settings()
    parquet = settings.paths.data_dir / "processed" / PARQUET_NAME
    if not parquet.is_file():
        typer.echo(f"error: {parquet} not found; run `fgx data prepare` first.", err=True)
        raise typer.Exit(1)

    frame = pl.read_parquet(parquet)
    audit = run_audit(frame)
    if probe:
        audit["probe"] = probe_feature_sets(frame, seed=settings.seed)

    out = settings.paths.reports_dir / "audit" / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    typer.echo(f"audit: {out}")
    for path in write_figures(audit, figures_dir):
        typer.echo(f"figure: {path}")


@app.command("split")
def split() -> None:
    """Choose time-based splits with the label-delay embargo -> data/processed/splits.json."""
    import polars as pl

    from fraudgraph.data.prepare import META_NAME, PARQUET_NAME
    from fraudgraph.data.splits import SplitError, plan_splits

    settings = get_settings()
    processed = settings.paths.data_dir / "processed"
    parquet = processed / PARQUET_NAME
    if not parquet.is_file():
        typer.echo(f"error: {parquet} not found; run `fgx data prepare` first.", err=True)
        raise typer.Exit(1)

    frame = pl.read_parquet(parquet, columns=["step", "type", "is_fraud"])
    try:
        plan = plan_splits(frame, settings.splits, settings.label_delay_steps)
    except SplitError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    meta = json.loads((processed / META_NAME).read_text(encoding="utf-8"))
    out = processed / "splits.json"
    out.write_text(
        plan.to_json(
            source_sha256=meta["source_sha256"],
            settings=settings.splits.model_dump(),
        ),
        encoding="utf-8",
    )
    typer.echo(f"{out}")
    typer.echo(f"{'split':<10} {'steps':>9} {'T+CO rows':>10} {'fraud':>6} {'prevalence':>10}")
    for name, step_range in plan.ranges.items():
        s = plan.stats[name]
        typer.echo(
            f"{name:<10} {step_range.start:>4}-{step_range.end:<4} {s.modelled_rows:>10,} "
            f"{s.fraud:>6,} {s.prevalence:>10.4%}"
        )
    typer.echo(
        f"embargo: {plan.label_delay_steps} steps; test covers {plan.test_days:.1f} days "
        f"-> bootstrap blocks of {plan.bootstrap_block_steps} steps"
    )


if __name__ == "__main__":
    app()

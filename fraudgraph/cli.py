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


if __name__ == "__main__":
    app()

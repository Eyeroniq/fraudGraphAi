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


def _data_sha256() -> str | None:
    meta = get_settings().paths.data_dir / "processed" / "paysim.meta.json"
    if not meta.is_file():
        return None
    return str(json.loads(meta.read_text(encoding="utf-8"))["source_sha256"])


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


features_app = typer.Typer(help="Decision-time features.", no_args_is_help=True)
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build() -> None:
    """Replay all events through the feature engine -> data/features/tabular.parquet."""
    import polars as pl

    from fraudgraph.data.prepare import PARQUET_NAME
    from fraudgraph.features.backfill import backfill
    from fraudgraph.features.spec import FeatureSpec

    settings = get_settings()
    parquet = settings.paths.data_dir / "processed" / PARQUET_NAME
    if not parquet.is_file():
        typer.echo(f"error: {parquet} not found; run `fgx data prepare` first.", err=True)
        raise typer.Exit(1)

    spec = FeatureSpec()
    table, report = backfill(pl.read_parquet(parquet), spec)
    out = settings.paths.data_dir / "features" / "tabular.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out, compression="zstd")
    (out.parent / "tabular.spec.json").write_text(
        json.dumps({"engine_version": spec.version, "windows": list(spec.windows),
                    "horizon_steps": spec.horizon_steps,
                    "label_delay_steps": settings.label_delay_steps}, indent=2) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    typer.echo(f"{out}: {report.feature_rows:,} rows x {table.width - 3} features")
    typer.echo(
        f"replayed {report.events:,} events in {report.seconds:.0f}s "
        f"({report.events / max(report.seconds, 1e-9):,.0f}/s); "
        f"peak state keys {report.peak_state_keys:,}"
    )


train_app = typer.Typer(help="Train models.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("baselines")
def train_baselines(
    trials: Annotated[int, typer.Option(help="LightGBM random-search trials per variant.")] = 20,
) -> None:
    """Fit B-rule, B-lr and B-gbm (with and without hour) -> artifacts/baselines/."""
    import polars as pl

    from fraudgraph.data.prepare import PARQUET_NAME
    from fraudgraph.data.splits import load_plan
    from fraudgraph.features.spec import FeatureSpec
    from fraudgraph.models.train import preview_ap, run_baselines, save, training_table

    settings = get_settings()
    data = settings.paths.data_dir
    needed = {
        "events": data / "processed" / PARQUET_NAME,
        "features": data / "features" / "tabular.parquet",
        "splits": data / "processed" / "splits.json",
    }
    missing = [str(p) for p in needed.values() if not p.is_file()]
    if missing:
        typer.echo(f"error: missing {missing}; run data prepare, split, features build.", err=True)
        raise typer.Exit(1)

    horizon = FeatureSpec().horizon_steps
    table = training_table(
        pl.read_parquet(needed["events"], columns=["txn_id", "step", "name_dest"]),
        pl.read_parquet(needed["features"]),
        load_plan(needed["splits"]),
        horizon,
    )
    results, predictions, models = run_baselines(
        table, horizon_steps=horizon, trials=trials, seed=settings.seed
    )
    from fraudgraph.tracking import tracked_run

    out = settings.paths.artifacts_dir / "baselines"
    save(out, results, predictions, models, {"trials": trials, "seed": settings.seed})
    previews = preview_ap(predictions, results)
    with tracked_run("baselines", data_sha256=_data_sha256(), task="1.3") as tracker:
        tracker.params({"trials": trials, "seed": settings.seed, "horizon_steps": horizon})
        for r in results:
            tracker.metrics(
                {
                    f"{r.name}.val_ap": r.val_ap,
                    **{f"{r.name}.{k}_ap": v for k, v in previews[r.name].items()},
                }
            )
        tracker.artifact(out / "summary.json")

    typer.echo(f"{out}")
    typer.echo(f"{'model':<20} {'AP train_val':>12} {'AP test':>8} {'AP late':>8} {'secs':>6}")
    for r in results:
        p = previews[r.name]
        typer.echo(
            f"{r.name:<20} {p['train_val']:>12.4f} {p['test']:>8.4f} {p['late_test']:>8.4f} "
            f"{r.seconds:>6.0f}"
        )


@app.command("evaluate")
def evaluate_cmd(
    predictions: Annotated[
        Path, typer.Option(help="Predictions parquet with score_<model> columns.")
    ] = Path("artifacts/baselines/predictions.parquet"),
    run: Annotated[str, typer.Option(help="Report name under reports/eval/.")] = "baselines",
    splits: Annotated[str, typer.Option(help="Comma-separated splits.")] = "test,late_test",
    reference: Annotated[
        str, typer.Option(help="Model the others are compared against (paired bootstrap).")
    ] = "lgbm_no_hour",
    alerts_per_day: Annotated[int, typer.Option(help="Review budget for precision@k.")] = 100,
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 1_000,
) -> None:
    """Metrics with block-bootstrap CIs, paired AP differences and plots."""
    import polars as pl

    from fraudgraph.data.prepare import PARQUET_NAME
    from fraudgraph.data.splits import load_plan
    from fraudgraph.models.evaluate import run_evaluation, save_report, write_plots

    settings = get_settings()
    plan = load_plan(settings.paths.data_dir / "processed" / "splits.json")
    frame = pl.read_parquet(predictions)
    events = settings.paths.data_dir / "processed" / PARQUET_NAME
    if "amount" not in frame.columns and events.is_file():
        amounts = pl.read_parquet(events, columns=["txn_id", "amount"])
        frame = frame.join(amounts, on="txn_id", how="left", maintain_order="left")

    wanted = tuple(s.strip() for s in splits.split(",") if s.strip())
    report = run_evaluation(
        frame,
        splits=wanted,
        block_steps=plan.bootstrap_block_steps,
        reference=reference,
        alerts_per_day=alerts_per_day,
        resamples=resamples,
        seed=settings.seed,
    )
    out_dir = settings.paths.reports_dir / "eval" / run
    path = save_report(report, out_dir)
    for split in wanted:
        write_plots(frame, split, plan.bootstrap_block_steps, out_dir)
    typer.echo(f"{path}")
    for split, data in report["splits"].items():
        typer.echo(
            f"\n{split}: {data['rows']:,} rows, {data['fraud']} fraud, {data['blocks']} blocks"
        )
        typer.echo(f"{'model':<18} {'AP':>7} {'95% CI':>17} {'R@P0.8':>7} {'P@k':>6}")
        for name, m in data["models"].items():
            lo, hi = m["ap"]["ci95"]
            typer.echo(
                f"{name:<18} {m['ap']['value']:>7.4f} [{lo:>6.4f}, {hi:>6.4f}] "
                f"{m['recall_at_p0.8']['value']:>7.3f} {m['precision_at_k']['value']:>6.3f}"
            )
        for pair, d in data["paired_ap_vs_reference"].items():
            lo, hi = d["ci95"]
            typer.echo(f"  dAP {pair:<34} {d['delta_ap']:+.4f} [{lo:+.4f}, {hi:+.4f}]")


@app.command("policy")
def policy_cmd(
    model: Annotated[
        str | None, typer.Option(help="Model to calibrate (default: champion).")
    ] = None,
    predictions: Annotated[Path, typer.Option(help="Predictions parquet.")] = Path(
        "artifacts/baselines/predictions.parquet"
    ),
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples for cost CIs.")] = 1_000,
) -> None:
    """Calibrate on calib_a, fit thresholds and the expected-cost rule, cost them on test."""
    import polars as pl

    from fraudgraph.data.prepare import PARQUET_NAME
    from fraudgraph.data.splits import load_plan
    from fraudgraph.decision.costs import CostParams
    from fraudgraph.decision.fit_policy import fit_policy, save, write_plots

    settings = get_settings()
    name = model or settings.champion
    plan = load_plan(settings.paths.data_dir / "processed" / "splits.json")
    frame = pl.read_parquet(predictions).join(
        pl.read_parquet(
            settings.paths.data_dir / "processed" / PARQUET_NAME, columns=["txn_id", "amount"]
        ),
        on="txn_id",
        how="left",
        maintain_order="left",
    )
    costs = CostParams(**settings.costs.model_dump())
    calibrator, policy, report = fit_policy(
        frame,
        name,
        costs,
        block_steps=plan.bootstrap_block_steps,
        resamples=resamples,
        seed=settings.seed,
    )
    out = settings.paths.artifacts_dir / "policy" / name
    report_dir = settings.paths.reports_dir / "eval" / f"policy_{name}"
    save(out, report_dir, calibrator, policy, report)
    write_plots(frame, name, calibrator, costs, report_dir)

    from fraudgraph.tracking import tracked_run

    with tracked_run(f"policy-{name}", data_sha256=_data_sha256(), task="1.5") as tracker:
        tracker.params(
            {"model": name, **policy["costs"], "global_threshold": policy["global_threshold"]}
        )
        tracker.metrics(
            {f"test.{rule}.cost": r["cost"]["total"] for rule, r in report["test"].items()}
            | {
                f"test.{rule}.review_share": r["action_share"]["REVIEW"]
                for rule, r in report["test"].items()
            }
        )
        tracker.artifact(report_dir / "policy_report.json")

    typer.echo(f"{out}  (global threshold on calibrated p: {policy['global_threshold']:.4g})")
    for split, stats in report["calibration"].items():
        line = "  ".join(
            f"{m}: brier {s['brier']:.5f} ece {s['ece']:.5f}" for m, s in stats.items()
        )
        typer.echo(f"calibration {split:<9} {line}")
    typer.echo(
        f"\n{'test rule':<27} {'total cost':>14} {'95% CI':>31} "
        f"{'review':>7} {'block':>6} {'stopped':>8}"
    )
    for rule, r in report["test"].items():
        lo, hi = r["cost"]["ci95"]
        typer.echo(
            f"{rule:<27} {r['cost']['total']:>14,.0f} [{lo:>13,.0f}, {hi:>13,.0f}] "
            f"{r['action_share']['REVIEW']:>7.2%} {r['action_share']['BLOCK']:>6.2%} "
            f"{r['fraud_amount_stopped_share']:>8.1%}"
        )


@app.command("bundle")
def bundle_cmd(
    model: Annotated[str | None, typer.Option(help="Model to package (default: champion).")] = None,
) -> None:
    """Package the champion into artifacts/bundles/<version>/ (the only thing the API loads)."""
    import polars as pl

    from fraudgraph.data.splits import assign_split, load_plan
    from fraudgraph.models.bundle import write_bundle
    from fraudgraph.tracking import git_sha, tracked_run

    settings = get_settings()
    name = model or settings.champion
    arts, reports, data = (
        settings.paths.artifacts_dir,
        settings.paths.reports_dir,
        settings.paths.data_dir,
    )
    needed = {
        "booster": arts / "baselines" / f"{name}.txt",
        "calibrator": arts / "policy" / name / "calibrator.joblib",
        "policy": arts / "policy" / name / "policy.json",
        "features": data / "features" / "tabular.parquet",
        "spec": data / "features" / "tabular.spec.json",
        "splits": data / "processed" / "splits.json",
    }
    missing = [str(p) for p in needed.values() if not p.is_file()]
    if missing:
        typer.echo(f"error: missing {missing}; run train baselines and policy first.", err=True)
        raise typer.Exit(1)

    metrics: dict[str, object] = {}
    eval_path = reports / "eval" / "baselines" / "metrics.json"
    if eval_path.is_file():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics["eval"] = {
            split: data_["models"].get(name, {}) for split, data_ in evaluation["splits"].items()
        }
    policy_report = reports / "eval" / f"policy_{name}" / "policy_report.json"
    if policy_report.is_file():
        metrics["policy_test"] = json.loads(policy_report.read_text(encoding="utf-8"))["test"]

    train = assign_split(pl.read_parquet(needed["features"]), load_plan(needed["splits"])).filter(
        pl.col("split") == "train_fit"
    )
    sha = git_sha()
    path = write_bundle(
        arts / "bundles",
        model=name,
        booster_path=needed["booster"],
        calibrator_path=needed["calibrator"],
        policy_path=needed["policy"],
        feature_spec=json.loads(needed["spec"].read_text(encoding="utf-8")),
        train_features=train,
        metrics=metrics,
        data_sha256=_data_sha256() or "unknown",
        git_sha=sha,
    )
    with tracked_run(f"bundle-{name}", data_sha256=_data_sha256(), task="1.6") as tracker:
        tracker.params({"model": name, "bundle": path.name})
        tracker.artifact(path)
    typer.echo(f"{path}")


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

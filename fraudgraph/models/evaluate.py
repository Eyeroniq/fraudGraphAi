"""`fgx evaluate`: metrics, CIs, paired comparisons and plots for a predictions file.

A predictions file has `txn_id, step, split, is_fraud` and one `score_<model>` column
per model (as written by `fgx train baselines`). Output: `reports/eval/<run>/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudgraph.models.compare import evaluate, paired_difference, pr_band, reliability

STEPS_PER_DAY = 24


def score_columns(predictions: pl.DataFrame) -> list[str]:
    return [c for c in predictions.columns if c.startswith("score_")]


def run_evaluation(
    predictions: pl.DataFrame,
    *,
    splits: tuple[str, ...],
    block_steps: int,
    reference: str | None,
    alerts_per_day: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Every model on every split, plus paired AP differences against `reference`."""
    alerts_per_block = max(1, round(alerts_per_day * block_steps / STEPS_PER_DAY))
    report: dict[str, Any] = {
        "block_steps": block_steps,
        "alerts_per_day": alerts_per_day,
        "resamples": resamples,
        "splits": {},
    }
    for split in splits:
        part = predictions.filter(pl.col("split") == split)
        if part.is_empty():
            continue
        y = part["is_fraud"].cast(pl.Int8).to_numpy()
        blocks = (part["step"].to_numpy() - 1) // block_steps
        amount = part["amount"].to_numpy() if "amount" in part.columns else None
        models: dict[str, Any] = {}
        for column in score_columns(part):
            name = column.removeprefix("score_")
            result = evaluate(
                y,
                part[column].to_numpy(),
                blocks,
                amount=amount,
                alerts_per_block=alerts_per_block,
                resamples=resamples,
                seed=seed,
            )
            models[name] = result.as_dict()
        comparisons = {}
        if reference and f"score_{reference}" in part.columns:
            for column in score_columns(part):
                name = column.removeprefix("score_")
                if name != reference:
                    comparisons[f"{name} - {reference}"] = paired_difference(
                        y,
                        part[column].to_numpy(),
                        part[f"score_{reference}"].to_numpy(),
                        blocks,
                        resamples=resamples,
                        seed=seed,
                    )
        report["splits"][split] = {
            "rows": part.height,
            "fraud": int(y.sum()),
            "blocks": int(np.unique(blocks).size),
            "models": models,
            "paired_ap_vs_reference": comparisons,
        }
    return report


def write_plots(
    predictions: pl.DataFrame, split: str, block_steps: int, out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = predictions.filter(pl.col("split") == split)
    y = part["is_fraud"].cast(pl.Int8).to_numpy()
    blocks = (part["step"].to_numpy() - 1) // block_steps
    paths = []

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for column in score_columns(part):
        band = pr_band(y, part[column].to_numpy(), blocks)
        (line,) = ax.plot(
            band["recall"], band["precision"], lw=1.5, label=column.removeprefix("score_")
        )
        ax.fill_between(
            band["recall"], band["low"], band["high"], color=line.get_color(), alpha=0.15
        )
    ax.axhline(y.mean(), color="grey", ls=":", lw=1, label=f"prevalence {y.mean():.4f}")
    ax.set(
        xlabel="recall",
        ylabel="precision",
        title=f"PR curves on {split} (95 % block-bootstrap bands)",
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / f"pr_{split}.png")
    fig.savefig(paths[-1], dpi=110)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    top = 0.0
    for column in score_columns(part):
        rel = reliability(y, part[column].to_numpy())
        ax.plot(
            rel["predicted"],
            rel["observed"],
            marker="o",
            ms=3,
            lw=1,
            label=column.removeprefix("score_"),
        )
        top = max(top, float(rel["predicted"].max()), float(rel["observed"].max()))
    ax.plot([0, top], [0, top], color="grey", ls=":", lw=1)
    ax.set(
        xlabel="mean predicted score",
        ylabel="observed fraud rate",
        title=f"Reliability on {split} (raw scores)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / f"reliability_{split}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close("all")
    return paths


def save_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path

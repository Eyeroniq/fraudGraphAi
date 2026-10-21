"""`fgx policy`: calibrate one model on calib_a, fit the decision policy, and cost it out
on test with block-bootstrap CIs (task 1.5)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from fraudgraph.decision.costs import (
    ACTIONS,
    CostParams,
    best_global_threshold,
    expected_cost_decisions,
    prior_shift,
    realised_costs,
    threshold_decisions,
)
from fraudgraph.models.calibrate import (
    Calibrator,
    brier,
    expected_calibration_error,
    fit_calibrator,
)
from fraudgraph.models.compare import block_weights

SENSITIVITY = ("c_review", "c_friction", "catch_rate")


def split_arrays(frame: pl.DataFrame, split: str, model: str) -> dict[str, np.ndarray]:
    part = frame.filter(pl.col("split") == split)
    return {
        "y": part["is_fraud"].cast(pl.Int8).to_numpy(),
        "score": part[f"score_{model}"].to_numpy(),
        "amount": part["amount"].to_numpy(),
        "step": part["step"].to_numpy(),
    }


def cost_with_ci(
    costs_per_row: np.ndarray, blocks: np.ndarray, resamples: int, seed: int
) -> dict[str, Any]:
    index, counts = block_weights(blocks, resamples, seed)
    per_block = np.bincount(index, weights=costs_per_row)
    samples = counts @ per_block
    return {
        "total": float(costs_per_row.sum()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def evaluate_rules(
    data: dict[str, np.ndarray],
    p: np.ndarray,
    threshold: float,
    costs: CostParams,
    *,
    calib_prevalence: float,
    block_steps: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    y, amount = data["y"], data["amount"]
    blocks = (data["step"] - 1) // block_steps
    test_prevalence = float(y.mean())
    rules = {
        "approve_all": np.zeros(len(y), dtype=np.int64),
        "global_threshold": threshold_decisions(p, threshold),
        "expected_cost": expected_cost_decisions(p, amount, costs),
        "expected_cost_prior_shift": expected_cost_decisions(
            prior_shift(p, calib_prevalence, test_prevalence), amount, costs
        ),
    }
    out: dict[str, Any] = {}
    baseline = None
    for name, actions in rules.items():
        cost = cost_with_ci(realised_costs(actions, y, amount, costs), blocks, resamples, seed)
        if baseline is None:
            baseline = cost["total"]
        fraud = y.astype(bool)
        out[name] = {
            "cost": cost,
            "savings_vs_approve_all": baseline - cost["total"],
            "action_share": {a: float(np.mean(actions == i)) for i, a in enumerate(ACTIONS)},
            "fraud_amount_stopped_share": float(
                np.sum(amount[fraud & (actions != 0)]) / max(np.sum(amount[fraud]), 1e-12)
            ),
        }
    return out


def fit_policy(
    frame: pl.DataFrame,
    model: str,
    costs: CostParams,
    *,
    block_steps: int,
    resamples: int,
    seed: int,
) -> tuple[Calibrator, dict[str, Any], dict[str, Any]]:
    """Returns (calibrator, policy, report)."""
    calib = split_arrays(frame, "calib_a", model)
    calibrators = {m: fit_calibrator(calib["score"], calib["y"], m) for m in ("isotonic", "platt")}
    calibrator = calibrators["isotonic"]
    calib_p = calibrator(calib["score"])
    threshold, _ = best_global_threshold(calib_p, calib["y"], calib["amount"], costs)
    calib_prevalence = float(calib["y"].mean())

    report: dict[str, Any] = {"model": model, "costs": asdict(costs), "global_threshold": threshold}
    report["calibration"] = {}
    for split in ("calib_b", "test", "late_test"):
        data = split_arrays(frame, split, model)
        report["calibration"][split] = {
            "raw": _calibration_stats(data["y"], np.clip(data["score"], 0, 1)),
            **{m: _calibration_stats(data["y"], c(data["score"])) for m, c in calibrators.items()},
        }

    test = split_arrays(frame, "test", model)
    test_p = calibrator(test["score"])
    report["test"] = evaluate_rules(
        test,
        test_p,
        threshold,
        costs,
        calib_prevalence=calib_prevalence,
        block_steps=block_steps,
        resamples=resamples,
        seed=seed,
    )
    report["sensitivity"] = sensitivity(calibrator, frame, model, costs)

    policy = {
        "model": model,
        "calibration": "isotonic",
        "costs": asdict(costs),
        "rule": "expected_cost",
        "global_threshold": threshold,
        "calib_prevalence": calib_prevalence,
    }
    return calibrator, policy, report


def sensitivity(
    calibrator: Calibrator, frame: pl.DataFrame, model: str, costs: CostParams
) -> list[dict[str, Any]]:
    """How threshold and test cost move when one cost parameter moves x0.5 / x2."""
    calib = split_arrays(frame, "calib_a", model)
    test = split_arrays(frame, "test", model)
    calib_p, test_p = calibrator(calib["score"]), calibrator(test["score"])
    rows = []
    for name in SENSITIVITY:
        for factor in (0.5, 1.0, 2.0):
            if name == "catch_rate" and costs.catch_rate * factor > 1:
                continue
            c = costs.scaled(**{name: factor})
            t, _ = best_global_threshold(calib_p, calib["y"], calib["amount"], c)
            ec = expected_cost_decisions(test_p, test["amount"], c)
            rows.append(
                {
                    "parameter": name,
                    "factor": factor,
                    "global_threshold": t,
                    "test_cost_global": float(
                        realised_costs(
                            threshold_decisions(test_p, t), test["y"], test["amount"], c
                        ).sum()
                    ),
                    "test_cost_expected": float(
                        realised_costs(ec, test["y"], test["amount"], c).sum()
                    ),
                    "test_review_share_expected": float(np.mean(ec == 1)),
                }
            )
    return rows


def save(
    out_dir: Path,
    report_dir: Path,
    calibrator: Calibrator,
    policy: dict[str, Any],
    report: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (report_dir / "policy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def write_plots(
    frame: pl.DataFrame, model: str, calibrator: Calibrator, costs: CostParams, out_dir: Path
) -> list[Path]:
    """Total cost vs global threshold (calib_a and test) and the calibrated reliability curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fraudgraph.models.compare import reliability

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = np.logspace(-6, 0, 200)
    for split in ("calib_a", "test"):
        data = split_arrays(frame, split, model)
        p = calibrator(data["score"])
        total = [
            realised_costs(threshold_decisions(p, t), data["y"], data["amount"], costs).sum()
            for t in grid
        ]
        ax.plot(grid, np.array(total) / 1e6, label=split)
    ax.set(xscale="log", xlabel="global threshold on calibrated p", ylabel="total cost (millions)")
    ax.set_title("Cost of the single global threshold (alert -> review)")
    ax.legend()
    fig.tight_layout()
    paths.append(out_dir / "cost_vs_threshold.png")
    fig.savefig(paths[-1], dpi=110)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    top = 0.0
    for split in ("calib_b", "test"):
        data = split_arrays(frame, split, model)
        rel = reliability(data["y"], calibrator(data["score"]))
        ax.plot(rel["predicted"], rel["observed"], marker="o", ms=3, label=split)
        top = max(top, float(rel["predicted"].max()), float(rel["observed"].max()))
    ax.plot([0, top], [0, top], color="grey", ls=":")
    ax.set(xlabel="calibrated p (mean per bin)", ylabel="observed fraud rate")
    ax.set_title(f"Isotonic calibration ({model})")
    ax.legend()
    fig.tight_layout()
    paths.append(out_dir / "reliability_calibrated.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close("all")
    return paths


def _calibration_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"brier": brier(y, p), "ece": expected_calibration_error(y, p)}

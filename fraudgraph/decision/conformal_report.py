"""`fgx conformal` (tasks 4.1-4.3): conformalise the champion on calib_b, compare
policies on test, and check coverage over time.

Outputs: artifacts/policy/<model>/{conformal.json, policy.json} (what the bundle and API
use) and reports/eval/conformal_<model>/report.json (+ plots) for docs/conformal.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudgraph.decision import conformal
from fraudgraph.decision.conformal import ConformalModel, coverage, set_labels
from fraudgraph.decision.costs import (
    ACTIONS,
    APPROVE,
    BLOCK,
    REVIEW,
    STEP_UP,
    CostParams,
    best_global_threshold,
    realised_costs,
    threshold_decisions,
)
from fraudgraph.decision.fit_policy import cost_with_ci, split_arrays
from fraudgraph.decision.policy import STEPS_PER_DAY, PolicyConfig, decide
from fraudgraph.models.calibrate import Calibrator, fit_calibrator

ALPHAS = (0.01, 0.05, 0.1, 0.2)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


def run(
    frame: pl.DataFrame,
    model: str,
    costs: CostParams,
    *,
    alpha: float,
    score: str,
    mondrian: bool,
    uncertain_action: str,
    capacity: int | None,
    block_steps: int,
    resamples: int,
    seed: int,
) -> tuple[Calibrator, ConformalModel, dict[str, Any], dict[str, Any]]:
    """Returns (calibrator, chosen conformal model, policy.json content, report)."""
    data = {s: split_arrays(frame, s, model) for s in ("calib_a", "calib_b", "test", "late_test")}
    calibrator = fit_calibrator(data["calib_a"]["score"], data["calib_a"]["y"])
    p = {s: calibrator(d["score"]) for s, d in data.items()}

    # ------------------------------------------------ 4.1 coverage grid
    grid = []
    for sc in ("lac", "aps"):
        for mond in (False, True):
            for a in ALPHAS:
                cm = conformal.fit(
                    p["calib_b"], data["calib_b"]["y"], alpha=a, score=sc, mondrian=mond
                )
                row: dict[str, Any] = {"score": sc, "mondrian": mond, "alpha": a}
                for split in ("test", "late_test"):
                    sets = cm.sets(p[split])
                    row[split] = {
                        "coverage": coverage(sets, data[split]["y"]),
                        "set_share": _shares(set_labels(sets)),
                    }
                grid.append(row)

    chosen = conformal.fit(
        p["calib_b"], data["calib_b"]["y"], alpha=alpha, score=score, mondrian=mondrian
    )
    per_day = {s: _per_day_coverage(chosen, p[s], data[s]) for s in ("test", "late_test")}

    # ------------------------------------------------ 4.2 policies on test
    test = data["test"]
    blocks = (test["step"] - 1) // block_steps
    threshold, _ = best_global_threshold(
        p["calib_a"], data["calib_a"]["y"], data["calib_a"]["amount"], costs
    )
    three = replace(costs, step_up_enabled=False)
    unsure = ACTION_INDEX[uncertain_action]
    base = PolicyConfig(costs=costs, conformal=chosen, uncertain_action=unsure)
    rules: dict[str, PolicyConfig | None] = {
        "global_threshold": None,
        "expected_cost (3 actions)": PolicyConfig(costs=three),
        "expected_cost + step_up": PolicyConfig(costs=costs),
        "+ conformal (uncertain -> REVIEW)": replace(base, uncertain_action=REVIEW),
        "+ conformal (uncertain -> STEP_UP)": replace(base, uncertain_action=STEP_UP)
        if costs.step_up_enabled
        else None,
    }
    if capacity is not None:
        rules[f"+ capacity {capacity}/day, first-come"] = replace(
            base, review_capacity_per_day=capacity
        )
        rules[f"+ capacity {capacity}/day, oracle"] = replace(
            base, review_capacity_per_day=capacity, capacity_mode="oracle"
        )
    policies: dict[str, Any] = {}
    chosen_actions = None
    for name, cfg in rules.items():
        if name.endswith("STEP_UP)") and cfg is None:
            continue
        if cfg is None:
            actions = threshold_decisions(p["test"], threshold)
            reasons = np.full(len(actions), "threshold", dtype=object)
        else:
            out = decide(p["test"], test["amount"], test["step"], cfg)
            actions, reasons = out.actions, out.reasons
        if name.startswith(f"+ capacity {capacity}/day, first-come"):
            chosen_actions = actions
        policies[name] = _summarise(actions, reasons, test, costs, blocks, resamples, seed)
    if chosen_actions is None:
        chosen_actions = decide(p["test"], test["amount"], test["step"], base).actions

    # ------------------------------------------------ 4.3 load vs alpha, uncertain cases
    load = []
    for a in ALPHAS:
        cm = conformal.fit(
            p["calib_b"], data["calib_b"]["y"], alpha=a, score=score, mondrian=mondrian
        )
        out = decide(p["test"], test["amount"], test["step"], replace(base, conformal=cm))
        day = (test["step"] - 1) // STEPS_PER_DAY
        per_day_reviews = np.bincount(
            day[out.actions == REVIEW] - day.min(), minlength=day.max() - day.min() + 1
        )
        load.append(
            {
                "alpha": a,
                "uncertain_share": float(np.mean(set_labels(out.sets) == "uncertain")),  # type: ignore[arg-type]
                "review_share": float(np.mean(out.actions == REVIEW)),
                "reviews_per_day_mean": float(per_day_reviews.mean()),
                "reviews_per_day_max": int(per_day_reviews.max()),
                "test_cost": float(
                    realised_costs(out.actions, test["y"], test["amount"], costs).sum()
                ),
            }
        )
    labels = set_labels(chosen.sets(p["test"]))
    fraud_rate_by_set = {
        lab: {
            "rows": int((labels == lab).sum()),
            "fraud_rate": float(test["y"][labels == lab].mean()),
        }
        for lab in ("legit", "uncertain", "fraud", "empty")
        if (labels == lab).any()
    }
    fraud_rate_by_action = {
        ACTIONS[a]: {
            "rows": int((chosen_actions == a).sum()),
            "fraud_rate": float(test["y"][chosen_actions == a].mean()),
        }
        for a in (APPROVE, STEP_UP, REVIEW, BLOCK)
        if (chosen_actions == a).any()
    }

    target = 1 - alpha
    fraud_cov = coverage(chosen.sets(p["test"]), test["y"])["fraud"]
    report = {
        "model": model,
        "chosen": {
            "alpha": alpha,
            "score": score,
            "mondrian": mondrian,
            "q_hat": list(chosen.q_hat),
        },
        "n_calib_b": {"legit": chosen.n_calib[0], "fraud": chosen.n_calib[1]},
        "coverage_grid": grid,
        "coverage_per_day": per_day,
        "done_when": {
            "target": target,
            "test_fraud_coverage": fraud_cov,
            "within_2pp": abs(fraud_cov - target) <= 0.02,
        },
        "policies_test": policies,
        "review_load_vs_alpha": load,
        "fraud_rate_by_set": fraud_rate_by_set,
        "fraud_rate_by_action": fraud_rate_by_action,
        "global_threshold": threshold,
        "costs": asdict(costs),
    }
    policy = {
        "model": model,
        "calibration": "isotonic",
        "costs": asdict(costs),
        "rule": "expected_cost",
        "global_threshold": threshold,
        "conformal": {"alpha": alpha, "score": score, "mondrian": mondrian},
        "uncertain_action": uncertain_action,
        "review_capacity_per_day": capacity,
        "calib_prevalence": float(data["calib_a"]["y"].mean()),
    }
    return calibrator, chosen, policy, report


def _shares(labels: np.ndarray) -> dict[str, float]:
    return {lab: float(np.mean(labels == lab)) for lab in ("legit", "fraud", "uncertain", "empty")}


def _per_day_coverage(
    model: ConformalModel, p: np.ndarray, d: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    day = (d["step"] - 1) // STEPS_PER_DAY
    sets = model.sets(p)
    rows = []
    for x in np.unique(day):
        m = day == x
        cov = coverage(sets[m], d["y"][m])
        rows.append(
            {
                "day": int(x),
                "fraud": int(d["y"][m].sum()),
                "fraud_coverage": cov["fraud"],
                "legit_coverage": cov["legit"],
            }
        )
    return rows


def _summarise(
    actions: np.ndarray,
    reasons: np.ndarray,
    d: dict[str, np.ndarray],
    costs: CostParams,
    blocks: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    fraud = d["y"].astype(bool)
    stopped = (actions != APPROVE) & fraud
    return {
        "cost": cost_with_ci(
            realised_costs(actions, d["y"], d["amount"], costs), blocks, resamples, seed
        ),
        "action_share": {
            ACTIONS[a]: float(np.mean(actions == a)) for a in (APPROVE, STEP_UP, REVIEW, BLOCK)
        },
        "reason_share": {str(r): float(np.mean(reasons == r)) for r in np.unique(reasons)},
        "fraud_amount_touched_share": float(
            d["amount"][stopped].sum() / max(d["amount"][fraud].sum(), 1e-12)
        ),
    }


def write_plots(report: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    target = report["done_when"]["target"]
    paths = []
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for split, colour in (("test", "#2c7fb8"), ("late_test", "#c0392b")):
        rows = report["coverage_per_day"][split]
        ax.plot(
            [r["day"] for r in rows],
            [r["fraud_coverage"] for r in rows],
            marker="o",
            color=colour,
            label=split,
        )
    ax.axhline(target, color="black", ls="--", lw=1, label=f"target {target:.2f}")
    ax.set(xlabel="simulated day", ylabel="fraud-class coverage", ylim=(0, 1.02))
    ax.set_title("Class-conditional coverage per day (holds on test, drifts later)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / "coverage_per_day.png")
    fig.savefig(paths[-1], dpi=110)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    load = report["review_load_vs_alpha"]
    ax.plot([r["alpha"] for r in load], [r["reviews_per_day_mean"] for r in load], marker="o")
    ax.set(xlabel="alpha", ylabel="reviews per day (mean)", xscale="log")
    ax.set_title("Review load vs alpha")
    fig.tight_layout()
    paths.append(out_dir / "review_load_vs_alpha.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close("all")
    return paths


def save(
    out_dir: Path,
    report_dir: Path,
    calibrator: Calibrator,
    model: ConformalModel,
    policy: dict[str, Any],
    report: dict[str, Any],
) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    conformal.save(model, out_dir / "conformal.json")
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )

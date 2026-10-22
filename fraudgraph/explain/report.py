"""`fgx explain` (Phase 5): reason codes, stability and explanation quality on the champion.

Studied payments are the ones the deployed policy does not simply approve (STEP_UP,
REVIEW, BLOCK on test): those are the explanations an analyst or customer sees.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from fraudgraph.decision.costs import ACTIONS, CostParams
from fraudgraph.decision.policy import PolicyConfig, decide
from fraudgraph.explain import quality, stability
from fraudgraph.explain.reasons import ReasonCodes, contributions
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.models.calibrate import Calibrator


def run(
    table: pl.DataFrame,
    booster: lgb.Booster,
    codes: ReasonCodes,
    calibrator: Calibrator,
    policy_config: PolicyConfig,
    *,
    horizon_steps: int,
    n_flagged: int = 2_000,
    n_quality: int = 500,
    n_models: int = 5,
    resamples: int = 1_000,
    seed: int = 0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Returns (report, SHAP background sample)."""
    rng = np.random.default_rng(seed)
    features = booster.feature_name()
    train = table.filter(pl.col("split") == "train_fit")
    test = table.filter(pl.col("split") == "test")
    x_train = frame_to_matrix(train, features, horizon_steps)
    y_train = train["is_fraud"].cast(pl.Int8).to_numpy()
    x_test = frame_to_matrix(test, features, horizon_steps)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)

    # Which payments does the policy flag?
    raw = np.asarray(booster.predict(x_test, num_iteration=booster.best_iteration or None))
    p = calibrator(raw)
    actions = decide(p, test["amount"].to_numpy(), test["step"].to_numpy(), policy_config).actions
    flagged = np.flatnonzero(actions != 0)
    pick = np.sort(rng.choice(flagged, min(n_flagged, len(flagged)), replace=False))
    x_flag = x_test[pick]
    q_rows = pick[: min(n_quality, len(pick))]
    x_q = x_test[q_rows]

    # ---------------------------------------------------------------- 5.1
    import shap

    reference = np.asarray(shap.TreeExplainer(booster).shap_values(x_flag[:200]))
    ours = contributions(booster, x_flag[:200])[0]
    sample = rng.choice(len(x_test), min(20_000, len(x_test)), replace=False)
    group_test = codes.grouped(contributions(booster, x_test[sample])[0])
    group_flag = codes.grouped(contributions(booster, x_flag)[0])
    examples = []
    for i in range(min(5, len(pick))):
        row = test.row(int(pick[i]), named=True)
        values = {f: row[f] for f in features} | {"amount": row["amount"], "type": row["type"]}
        examples.append(
            {
                "txn_id": row["txn_id"],
                "action": ACTIONS[int(actions[pick[i]])],
                "p_fraud": float(p[pick[i]]),
                "reasons": [r.__dict__ for r in codes.reasons(group_flag[i], values)],
            }
        )
    shap_check = {
        "max_abs_diff_vs_shap_tree_explainer": float(np.max(np.abs(reference - ours))),
        "rows": 200,
    }

    # ---------------------------------------------------------------- 5.2
    models = stability.refit_models(booster, x_train, y_train, n_models=n_models, seed=seed)
    seed_stab = stability.seed_stability(models, x_flag, codes)
    threshold = stability.stability_threshold(seed_stab["gap"], seed_stab["per_row_jaccard"])
    perturb = stability.perturbation_stability(booster, x_flag, codes, std, seed=seed)
    backgrounds = [
        x_train[rng.choice(len(x_train), min(1_000, len(x_train)), replace=False)] for _ in range(3)
    ]
    background = stability.background_stability(booster, x_q, backgrounds, codes)

    # ---------------------------------------------------------------- 5.3
    importance = quality.permutation_importance(
        booster, x_train[rng.choice(len(x_train), min(20_000, len(x_train)), replace=False)], seed
    )
    shuffled = lgb.train(
        {
            **{
                k: v
                for k, v in booster.params.items()
                if k in ("objective", "num_leaves", "min_data_in_leaf", "learning_rate")
            },
            "verbosity": -1,
            "seed": seed,
        },
        lgb.Dataset(x_train, rng.permutation(y_train), feature_name=features),
        num_boost_round=booster.current_iteration(),
    )
    explain_fns = quality.explainers(booster, importance, seed)
    shuffled_fns = quality.explainers(
        shuffled,
        quality.permutation_importance(shuffled, x_train[:20_000], seed),
        seed + 1,  # independent noise: a random explainer must not "agree" with itself
    )
    methods: dict[str, dict[str, Any]] = {}
    for name, fn in explain_fns.items():
        attr = fn(x_q)
        scores: dict[str, np.ndarray] = {
            "faithfulness": quality.faithfulness_correlation(booster, x_q, attr, mean, seed=seed),
            "randomisation_corr": quality.randomisation_correlation(attr, shuffled_fns[name](x_q)),
        }
        scores |= quality.quantus_scores(booster, x_q, fn, mean, std)
        methods[name] = {k: quality.bootstrap_ci(v, resamples, seed) for k, v in scores.items()}

    report = {
        "studied": {
            "flagged_in_test": len(flagged),
            "sampled": len(pick),
            "quality_rows": len(q_rows),
        },
        "shap_check": shap_check,
        "global_mean_abs_group_shap": dict(
            sorted(
                zip(codes.codes, np.abs(group_test).mean(axis=0).tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        ),
        "flagged_mean_group_shap": dict(
            sorted(
                zip(codes.codes, group_flag.mean(axis=0).tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        ),
        "examples": examples,
        "seed_stability": {
            k: v for k, v in seed_stab.items() if not k.startswith("per_row") and k != "gap"
        },
        "stability_threshold": threshold,
        "perturbation": perturb,
        "background": background,
        "quality": methods,
        "top_features": [
            features[i]
            for i in np.argsort(-np.abs(contributions(booster, x_test[sample])[0]).mean(axis=0))[:5]
        ],
    }
    return report, backgrounds[0]


def write_plots(
    booster: lgb.Booster, table: pl.DataFrame, report: dict[str, Any], horizon: int, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    groups = report["global_mean_abs_group_shap"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = list(groups)[::-1]
    ax.barh(names, [groups[n] for n in names], color="#2c7fb8")
    ax.set(xlabel="mean |SHAP| (log-odds), test sample", title="What drives the champion's scores")
    fig.tight_layout()
    fig.savefig(out / "global_groups.png", dpi=110)

    features = booster.feature_name()
    test = table.filter(pl.col("split") == "test")
    test = test.sample(n=min(20_000, test.height), seed=0)  # cap at the test split's size
    x = frame_to_matrix(test, features, horizon)
    c = contributions(booster, x)[0]
    top = report["top_features"]
    fig, axes = plt.subplots(1, len(top), figsize=(3.2 * len(top), 3))
    for ax, name in zip(np.atleast_1d(axes), top, strict=False):
        j = features.index(name)
        ax.scatter(x[:, j], c[:, j], s=2, alpha=0.3)
        ax.set(xlabel=name, ylabel="SHAP")
    fig.suptitle("Dependence plots, top 5 features")
    fig.tight_layout()
    fig.savefig(out / "dependence_top5.png", dpi=110)
    plt.close("all")


def save(
    report: dict[str, Any],
    background: np.ndarray,
    reason_codes_path: Path,
    artifacts: Path,
    reports: Path,
) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reason_codes_path, artifacts / "reason_codes.yaml")
    np.save(artifacts / "shap_background.npy", background.astype(np.float32))
    (artifacts / "explain.json").write_text(
        json.dumps(
            {"stability_gap_threshold": report["stability_threshold"]["threshold"]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )


def policy_from_artifacts(policy_dir: Path) -> tuple[Calibrator, PolicyConfig]:
    import joblib

    from fraudgraph.decision import conformal

    policy = json.loads((policy_dir / "policy.json").read_text(encoding="utf-8"))
    conf = policy_dir / "conformal.json"
    return joblib.load(policy_dir / "calibrator.joblib"), PolicyConfig(
        costs=CostParams(**policy["costs"]),
        conformal=conformal.load(conf) if conf.is_file() else None,
        uncertain_action=ACTIONS.index(policy.get("uncertain_action", "REVIEW")),
        review_capacity_per_day=policy.get("review_capacity_per_day"),
    )

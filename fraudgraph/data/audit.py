"""Data audit for PaySim: the numbers behind `docs/data_audit.md`.

`run_audit` computes every statistic from the prepared parquet and returns a plain
dict (written to `reports/audit/audit.json`). `probe_feature_sets` trains quick
LightGBM models on candidate column sets to measure how much signal each leaks.
Nothing here is used by the production pipeline; it exists so the audit can be
re-run and checked.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

MODELLED_TYPES = ("TRANSFER", "CASH_OUT")
STEPS_PER_DAY = 24

# Steps where legit traffic is dense enough for representative splits. Outside this
# window legit volume collapses while fraud stays flat (see `daily`).
DENSE_LAST_STEP = 408
PROBE_TRAIN_LAST_STEP = 300


def run_audit(frame: pl.DataFrame) -> dict[str, Any]:
    """All audit statistics for a prepared PaySim frame."""
    frame = frame.with_columns(
        day=((pl.col("step") - 1) // STEPS_PER_DAY).cast(pl.Int32),
        hour=(pl.col("step") % STEPS_PER_DAY).cast(pl.Int32),
        # Events into the same destination at strictly earlier steps (every type is an edge).
        dest_prior_in=(pl.col("step").rank("min").over("name_dest") - 1).cast(pl.Int64),
    )
    modelled = frame.filter(pl.col("type").is_in(MODELLED_TYPES))
    return {
        "overview": overview(frame, modelled),
        "by_type": by_type(frame),
        "daily": daily(frame),
        "hourly": hourly(frame),
        "split_guidance": split_guidance(modelled),
        "id_reuse": id_reuse(frame),
        "destination_history": destination_history(modelled),
        "transfer_cashout_chain": transfer_cashout_chain(frame),
        "leak_checks": leak_checks(frame, modelled),
    }


def overview(frame: pl.DataFrame, modelled: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "fraud_rows": int(frame["is_fraud"].sum()),
        "fraud_rate": float(frame["is_fraud"].mean()),  # type: ignore[arg-type]
        "modelled_rows": modelled.height,
        "modelled_fraud_rate": float(modelled["is_fraud"].mean()),  # type: ignore[arg-type]
        "steps": [int(frame["step"].min()), int(frame["step"].max())],  # type: ignore[arg-type]
    }


def by_type(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.group_by("type")
        .agg(rows=pl.len(), fraud=pl.col("is_fraud").sum(), fraud_rate=pl.col("is_fraud").mean())
        .sort("rows", descending=True)
        .to_dicts()
    )


def daily(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Per simulated day: all rows, modelled rows, fraud and the modelled fraud rate."""
    return (
        frame.group_by("day")
        .agg(
            rows=pl.len(),
            modelled_rows=pl.col("type").is_in(MODELLED_TYPES).sum(),
            fraud=pl.col("is_fraud").sum(),
        )
        .with_columns(modelled_fraud_rate=pl.col("fraud") / pl.col("modelled_rows"))
        .sort("day")
        .to_dicts()
    )


def hourly(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.group_by("hour")
        .agg(rows=pl.len(), fraud=pl.col("is_fraud").sum(), fraud_rate=pl.col("is_fraud").mean())
        .sort("hour")
        .to_dicts()
    )


def split_guidance(modelled: pl.DataFrame) -> dict[str, Any]:
    """Where row-count quantiles fall in time, and how much fraud sits before them."""
    per_step = (
        modelled.group_by("step")
        .agg(rows=pl.len(), fraud=pl.col("is_fraud").sum())
        .sort("step")
        .with_columns(
            cum_rows=pl.col("rows").cum_sum() / pl.col("rows").sum(),
            cum_fraud=pl.col("fraud").cum_sum() / pl.col("fraud").sum(),
        )
    )
    quantiles = []
    for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        row = per_step.filter(pl.col("cum_rows") >= q).row(0, named=True)
        quantiles.append(
            {"row_share": q, "step": row["step"], "fraud_share_before": row["cum_fraud"]}
        )
    dense = modelled.filter(pl.col("step") <= DENSE_LAST_STEP)
    late = modelled.filter(pl.col("step") > DENSE_LAST_STEP)
    return {
        "row_quantiles": quantiles,
        "dense_last_step": DENSE_LAST_STEP,
        "dense": _counts(dense),
        "late": _counts(late),
    }


def id_reuse(frame: pl.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("name_orig", "name_dest"):
        counts = frame.group_by(column).len()
        repeated = counts.filter(pl.col("len") > 1)
        result[column] = {
            "unique": counts.height,
            "share_ids_seen_more_than_once": repeated.height / counts.height,
            "share_rows_with_repeated_id": int(repeated["len"].sum()) / frame.height,
            "max_rows_per_id": int(counts["len"].max()),  # type: ignore[arg-type]
        }
    origins = frame.select(pl.col("name_orig").unique())
    dests = frame.select(pl.col("name_dest").unique().alias("name_orig"))
    result["accounts_both_sender_and_receiver"] = origins.join(dests, on="name_orig").height
    result["merchant_origins"] = frame.filter(pl.col("name_orig").str.starts_with("M")).height
    return result


def destination_history(modelled: pl.DataFrame) -> dict[str, Any]:
    """Is the destination new? Decision-time safe: counts only strictly earlier steps."""
    by_label = (
        modelled.group_by("is_fraud")
        .agg(
            share_new_destination=(pl.col("dest_prior_in") == 0).mean(),
            median_prior_in=pl.col("dest_prior_in").median(),
            p90_prior_in=pl.col("dest_prior_in").quantile(0.9),
        )
        .sort("is_fraud")
        .to_dicts()
    )
    bucket = (
        pl.when(pl.col("dest_prior_in") == 0)
        .then(pl.lit("a: 0"))
        .when(pl.col("dest_prior_in") < 3)
        .then(pl.lit("b: 1-2"))
        .when(pl.col("dest_prior_in") < 10)
        .then(pl.lit("c: 3-9"))
        .otherwise(pl.lit("d: 10+"))
    )
    by_bucket = (
        modelled.with_columns(prior_in=bucket)
        .group_by("prior_in")
        .agg(rows=pl.len(), fraud_rate=pl.col("is_fraud").mean())
        .sort("prior_in")
        .to_dicts()
    )
    fraud_dests = modelled.filter("is_fraud").group_by("name_dest").len()
    return {
        "by_label": by_label,
        "fraud_rate_by_prior_in": by_bucket,
        "distinct_fraud_destinations": fraud_dests.height,
        "destinations_with_more_than_one_fraud": fraud_dests.filter(pl.col("len") > 1).height,
    }


def transfer_cashout_chain(frame: pl.DataFrame) -> dict[str, int]:
    """Does a fraudulent TRANSFER's destination later cash out as a fraud origin?"""
    fraud = frame.filter("is_fraud")
    transfers = fraud.filter(pl.col("type") == "TRANSFER")
    cash_outs = fraud.filter(pl.col("type") == "CASH_OUT")
    any_origin = frame.select(pl.col("name_orig").unique())
    return {
        "fraud_transfers": transfers.height,
        "dest_is_fraud_cashout_origin": transfers.join(
            cash_outs, left_on="name_dest", right_on="name_orig"
        ).height,
        "dest_is_any_origin": transfers.join(
            any_origin, left_on="name_dest", right_on="name_orig"
        ).height,
    }


def leak_checks(frame: pl.DataFrame, modelled: pl.DataFrame) -> dict[str, Any]:
    flagged = frame.filter("is_flagged_fraud")
    per_label = (
        modelled.group_by("is_fraud")
        .agg(
            amount_equals_oldbalance_orig=(pl.col("amount") == pl.col("oldbalance_orig")).mean(),
            oldbalance_orig_zero=(pl.col("oldbalance_orig") == 0).mean(),
            orig_balance_inconsistent=(
                (pl.col("oldbalance_orig") - pl.col("amount") - pl.col("newbalance_orig")).abs()
                > 0.01
            ).mean(),
            dest_balance_inconsistent=(
                (pl.col("oldbalance_dest") + pl.col("amount") - pl.col("newbalance_dest")).abs()
                > 0.01
            ).mean(),
            dest_balances_both_zero=(
                (pl.col("oldbalance_dest") == 0)
                & (pl.col("newbalance_dest") == 0)
                & (pl.col("amount") > 0)
            ).mean(),
            amount_exactly_10m=(pl.col("amount") == 1e7).mean(),
            amount_zero=(pl.col("amount") == 0).sum(),
            amount_median=pl.col("amount").median(),
            amount_max=pl.col("amount").max(),
        )
        .sort("is_fraud")
        .to_dicts()
    )
    total_fraud = int(frame["is_fraud"].sum())
    merchant_dest = frame.filter(pl.col("name_dest").str.starts_with("M"))
    return {
        "per_label_modelled": per_label,
        "is_flagged_fraud": {
            "rows": flagged.height,
            "fraud": int(flagged["is_fraud"].sum()),
            "recall": int(flagged["is_fraud"].sum()) / total_fraud,
            "types": sorted(flagged["type"].unique().to_list()),
            "min_amount": float(flagged["amount"].min()) if flagged.height else None,  # type: ignore[arg-type]
        },
        "merchant_destinations": {
            "rows": merchant_dest.height,
            "fraud": int(merchant_dest["is_fraud"].sum()),
            "types": sorted(merchant_dest["type"].unique().to_list()),
        },
    }


# ------------------------------------------------------------------ model probe

PROBE_SETS: dict[str, list[str]] = {
    "all columns incl. post-transaction balances (leaky)": [
        "log_amount",
        "is_transfer",
        "hour",
        "oldbalance_orig",
        "newbalance_orig",
        "oldbalance_dest",
        "newbalance_dest",
        "error_orig",
        "error_dest",
    ],
    "destination balances only": [
        "log_amount",
        "is_transfer",
        "oldbalance_dest",
        "newbalance_dest",
    ],
    "single flag amount == oldbalance_orig": ["amount_equals_oldbalance"],
    "amount, type, oldbalance_orig": [
        "log_amount",
        "is_transfer",
        "oldbalance_orig",
        "amount_ratio",
    ],
    "amount, type, oldbalance_orig, hour": [
        "log_amount",
        "is_transfer",
        "hour",
        "oldbalance_orig",
        "amount_ratio",
    ],
    "amount, type, hour (no balances)": ["log_amount", "is_transfer", "hour", "amount_10m"],
    "amount, type, hour + new-destination flag": [
        "log_amount",
        "is_transfer",
        "hour",
        "amount_10m",
        "dest_is_new",
        "log_dest_prior_in",
    ],
}


def probe_feature_sets(frame: pl.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Average precision of a small LightGBM per column set, on TRANSFER + CASH_OUT.

    Train: steps <= 300. Test: dense steps 301..408. Late: steps > 408 (natural shift).
    This is an audit probe with fixed hyper-parameters, not the Phase 1 baseline.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    data = (
        frame.filter(pl.col("type").is_in(MODELLED_TYPES))
        .with_columns(
            dest_prior_in=(pl.col("step").rank("min").over("name_dest") - 1).cast(pl.Int64)
        )
        .with_columns(
            log_amount=pl.col("amount").log1p(),
            is_transfer=(pl.col("type") == "TRANSFER").cast(pl.Int8),
            hour=pl.col("step") % STEPS_PER_DAY,
            amount_ratio=pl.col("amount") / (pl.col("oldbalance_orig") + 1),
            amount_equals_oldbalance=(pl.col("amount") == pl.col("oldbalance_orig")).cast(pl.Int8),
            amount_10m=(pl.col("amount") == 1e7).cast(pl.Int8),
            error_orig=pl.col("oldbalance_orig") - pl.col("amount") - pl.col("newbalance_orig"),
            error_dest=pl.col("oldbalance_dest") + pl.col("amount") - pl.col("newbalance_dest"),
            dest_is_new=(pl.col("dest_prior_in") == 0).cast(pl.Int8),
            log_dest_prior_in=pl.col("dest_prior_in").log1p(),
        )
    )
    train = data.filter(pl.col("step") <= PROBE_TRAIN_LAST_STEP)
    test = data.filter(pl.col("step").is_between(PROBE_TRAIN_LAST_STEP + 1, DENSE_LAST_STEP))
    late = data.filter(pl.col("step") > DENSE_LAST_STEP)
    y_train = train["is_fraud"].to_numpy()

    results = []
    for name, columns in PROBE_SETS.items():
        started = time.perf_counter()
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=seed,
            deterministic=True,  # same numbers on every run and thread count
            force_row_wise=True,
            verbose=-1,
        )
        model.fit(train.select(columns).to_numpy(), y_train)
        scores = {}
        for split_name, split in (("test", test), ("late", late)):
            proba = np.asarray(model.predict_proba(split.select(columns).to_numpy()))[:, 1]
            scores[split_name] = float(average_precision_score(split["is_fraud"].to_numpy(), proba))
        results.append(
            {
                "features": name,
                "columns": columns,
                "ap_test": scores["test"],
                "ap_late": scores["late"],
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    prevalence = {
        "test": int(test["is_fraud"].sum()) / test.height,
        "late": int(late["is_fraud"].sum()) / late.height,
    }
    return {"prevalence": prevalence, "sets": results}


# ------------------------------------------------------------------ figures


def write_figures(audit: dict[str, Any], out_dir: Path) -> list[Path]:
    """Small PNGs for the audit doc. Needs the `research` extra (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    days = audit["daily"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.array([d["day"] for d in days])
    ax.bar(x, [d["modelled_rows"] for d in days], color="#9aa5b1", label="TRANSFER + CASH_OUT rows")
    ax.set_xlabel("simulated day")
    ax.set_ylabel("rows per day")
    ax2 = ax.twinx()
    ax2.plot(x, [d["fraud"] for d in days], color="#c0392b", marker="o", ms=3, label="fraud")
    ax2.set_ylabel("fraud per day", color="#c0392b")
    ax2.set_ylim(0, max(d["fraud"] for d in days) * 1.3)
    ax.axvline(DENSE_LAST_STEP / STEPS_PER_DAY - 0.5, color="black", ls="--", lw=1)
    ax.set_title("Legit volume collapses after day 16; fraud stays flat")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "volume_vs_fraud_by_day.png"))

    hours = audit["hourly"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar([h["hour"] for h in hours], [100 * h["fraud_rate"] for h in hours], color="#c0392b")
    ax.set_xlabel("hour of day (step % 24)")
    ax.set_ylabel("fraud rate, %")
    ax.set_title("Fraud is uniform over the day, legit is not: night hours look risky")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "fraud_rate_by_hour.png"))

    buckets = audit["destination_history"]["fraud_rate_by_prior_in"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(
        [b["prior_in"].split(": ")[1] for b in buckets],
        [100 * b["fraud_rate"] for b in buckets],
        color="#2c7fb8",
    )
    ax.set_xlabel("earlier transactions into the destination")
    ax.set_ylabel("fraud rate, %")
    ax.set_title("New destinations are ~10x riskier")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "fraud_rate_by_destination_history.png"))

    plt.close("all")
    return paths


def _counts(frame: pl.DataFrame) -> dict[str, Any]:
    fraud = int(frame["is_fraud"].sum())
    return {"rows": frame.height, "fraud": fraud, "fraud_rate": fraud / max(frame.height, 1)}


def _save(fig: Any, path: Path) -> Path:
    fig.savefig(path, dpi=110)
    return path

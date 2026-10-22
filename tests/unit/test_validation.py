from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from fraudgraph.validation import suites as v


def events(n: int = 400, **overrides: list[object]) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, list[object]] = {
        "txn_id": [f"T{i}" for i in range(n)],
        "step": [1 + i % 200 for i in range(n)],
        "type": ["TRANSFER" if i % 2 else "CASH_OUT" for i in range(n)],
        "amount": list(rng.uniform(10, 1000, n)),
        "name_orig": [f"C{i}" for i in range(n)],
        "name_dest": [f"C{1000 + i % 37}" for i in range(n)],
        "is_fraud": [i % 50 == 0 for i in range(n)],
    }
    return pl.DataFrame(data | overrides)


def features(frame: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return frame.select("txn_id", "step", "is_fraud").with_columns(
        f_a=pl.Series(rng.normal(size=frame.height)), f_b=pl.Series(rng.normal(size=frame.height))
    )


def statuses(checks: list[v.Check]) -> dict[str, str]:
    return {c.name: c.status for c in checks}


def test_clean_data_passes_integrity() -> None:
    e = events()
    result = statuses(v.data_integrity(e, features(e), ["f_a", "f_b"], v.Limits()))
    assert set(result.values()) == {"pass"}


def test_integrity_fails_on_duplicates_negative_amounts_and_unmodelled_fraud() -> None:
    e = events()
    bad = pl.concat([e, e.head(1)]).with_columns(
        amount=pl.when(pl.col("txn_id") == "T3").then(-1.0).otherwise(pl.col("amount")),
        type=pl.when(pl.col("txn_id") == "T0").then(pl.lit("PAYMENT")).otherwise(pl.col("type")),
    )
    result = statuses(v.data_integrity(bad, features(bad), ["f_a", "f_b"], v.Limits()))
    assert result["unique_txn_id"] == "fail"
    assert result["amount_non_negative"] == "fail"
    assert result["fraud_only_on_modelled_types"] == "fail"


def test_constant_features_only_warn() -> None:
    e = events()
    table = features(e).with_columns(f_b=pl.lit(1.0))
    result = statuses(v.data_integrity(e, table, ["f_a", "f_b"], v.Limits()))
    assert result["features_not_constant"] == "warn"


def test_train_test_checks_catch_overlap_order_and_label_shift() -> None:
    table = features(events()).with_columns(type=pl.lit("TRANSFER"))
    train = table.filter(pl.col("step") <= 150)
    test = table.filter(pl.col("step") > 100)  # overlaps train and starts too early
    result = statuses(v.train_test_validation(train, test, ["f_a", "f_b"], 12, v.Limits()))
    assert result["time_ordered"] == "fail"
    assert result["no_shared_transactions"] == "fail"
    shifted = test.filter(pl.col("step") > 180).with_columns(is_fraud=pl.lit(True))
    late = statuses(v.train_test_validation(train, shifted, ["f_a", "f_b"], 12, v.Limits()))
    assert late["label_drift"] == "fail"
    assert late["time_ordered"] == "pass"


def test_reports_are_written_and_a_failure_fails_the_run(tmp_path: Path) -> None:
    suites = {
        "s": [
            v.Check("ok", "always", True, 1),
            v.Check("bad", "never", False, 2),
            v.Check("meh", "maybe", False, 3, severity="warn"),
        ]
    }
    summary = v.summarise(suites)
    assert summary["passed"] is False
    assert summary["counts"] == {"pass": 1, "warn": 1, "fail": 1}
    json_path, html_path = v.write_reports(summary, tmp_path, "test <run>")
    assert json.loads(json_path.read_text(encoding="utf-8"))["counts"]["fail"] == 1
    page = html_path.read_text(encoding="utf-8")
    assert "test &lt;run&gt;" in page  # escaped
    assert "FAIL" in page
    assert "WARN" in page

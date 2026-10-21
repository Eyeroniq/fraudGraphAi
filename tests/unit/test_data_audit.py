from __future__ import annotations

import polars as pl
import pytest

from fraudgraph.data.audit import run_audit


def txn(
    step: int,
    type_: str,
    amount: float,
    orig: str,
    dest: str,
    *,
    fraud: bool = False,
    old_orig: float = 1000.0,
) -> dict[str, object]:
    return {
        "txn_id": f"T{step:03d}{orig}",
        "step": step,
        "type": type_,
        "amount": amount,
        "name_orig": orig,
        "oldbalance_orig": old_orig,
        "newbalance_orig": max(old_orig - amount, 0.0),
        "name_dest": dest,
        "oldbalance_dest": 0.0,
        "newbalance_dest": amount,
        "is_fraud": fraud,
        "is_flagged_fraud": False,
    }


@pytest.fixture
def audit() -> dict[str, object]:
    rows = [
        # Destination C9 receives at step 1, twice at step 2 (same step: neither sees the
        # other), then at step 5.
        txn(1, "PAYMENT", 10.0, "C1", "M1"),
        txn(1, "CASH_OUT", 50.0, "C2", "C9"),
        txn(2, "TRANSFER", 1000.0, "C3", "C9", fraud=True, old_orig=1000.0),
        txn(2, "CASH_OUT", 70.0, "C4", "C9"),
        txn(5, "TRANSFER", 300.0, "C5", "C9"),
        # Fraud to a brand-new destination; that destination later sends money.
        txn(30, "TRANSFER", 500.0, "C6", "C7", fraud=True, old_orig=500.0),
        txn(31, "CASH_OUT", 20.0, "C7", "C8"),
    ]
    frame = pl.DataFrame(rows).with_columns(pl.col("step").cast(pl.Int32))
    return run_audit(frame)


def test_overview_counts_only_modelled_types(audit: dict[str, object]) -> None:
    overview = audit["overview"]
    assert isinstance(overview, dict)
    assert (overview["rows"], overview["fraud_rows"], overview["modelled_rows"]) == (7, 2, 6)


def test_destination_history_ignores_same_step_events(audit: dict[str, object]) -> None:
    history = audit["destination_history"]
    assert isinstance(history, dict)
    by_label = {row["is_fraud"]: row for row in history["by_label"]}
    # Frauds: C9 at step 2 has 1 earlier event (step 1, not its step-2 sibling); C7 is new.
    assert by_label[True]["share_new_destination"] == pytest.approx(0.5)
    assert history["distinct_fraud_destinations"] == 2


def test_transfer_chain_and_leak_flags(audit: dict[str, object]) -> None:
    chain = audit["transfer_cashout_chain"]
    assert chain == {
        "fraud_transfers": 2,
        "dest_is_fraud_cashout_origin": 0,
        "dest_is_any_origin": 1,
    }

    leaks = audit["leak_checks"]
    assert isinstance(leaks, dict)
    per_label = {row["is_fraud"]: row for row in leaks["per_label_modelled"]}
    assert per_label[True]["amount_equals_oldbalance_orig"] == 1.0
    assert per_label[False]["amount_equals_oldbalance_orig"] == 0.0
    assert leaks["merchant_destinations"] == {"rows": 1, "fraud": 0, "types": ["PAYMENT"]}


def test_daily_buckets_use_24_steps(audit: dict[str, object]) -> None:
    days = {row["day"]: row for row in audit["daily"]}  # type: ignore[attr-defined]
    assert sorted(days) == [0, 1]
    assert days[1]["fraud"] == 1  # step 30 is day 1

from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl
import pytest

from fraudgraph.config import SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.splits import (
    SPLITS,
    SplitError,
    SplitPlan,
    assign_split,
    load_plan,
    plan_splits,
)
from fraudgraph.data.synth import SynthConfig, generate

SYNTH = SynthConfig()
# The fixture's dense period ends where its sparse days begin.
FIXTURE_SPLITS = SplitSettings(dense_last_step=(SYNTH.days - SYNTH.sparse_days) * 24)
DELAY = 12


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return to_canonical(generate(SYNTH).frame)


@pytest.fixture(scope="module")
def plan(frame: pl.DataFrame) -> SplitPlan:
    return plan_splits(frame, FIXTURE_SPLITS, DELAY)


def test_embargo_separates_training_from_what_follows(plan: SplitPlan) -> None:
    r = plan.ranges
    assert r["train_fit"].end + DELAY < r["train_val"].start
    assert r["train_val"].end + DELAY < r["calib_a"].start


def test_splits_are_ordered_and_disjoint_in_time(plan: SplitPlan) -> None:
    ranges = [plan.ranges[name] for name in SPLITS]
    for earlier, later in itertools.pairwise(ranges):
        assert earlier.end < later.start
    assert plan.ranges["calib_a"].end + 1 == plan.ranges["calib_b"].start


def test_dense_and_late_periods(plan: SplitPlan, frame: pl.DataFrame) -> None:
    assert plan.ranges["test"].end == FIXTURE_SPLITS.dense_last_step
    assert plan.ranges["late_test"].start == FIXTURE_SPLITS.dense_last_step + 1
    assert plan.ranges["late_test"].end == frame["step"].max()


def test_no_transaction_in_two_splits(plan: SplitPlan, frame: pl.DataFrame) -> None:
    labelled = assign_split(frame, plan)
    assert labelled["txn_id"].n_unique() == labelled.height
    counted = labelled.filter(pl.col("split").is_not_null())
    assert counted.group_by("txn_id").agg(pl.col("split").n_unique()).get_column("split").max() == 1
    # Embargoed rows exist and belong to no split.
    embargoed = labelled.filter(pl.col("split").is_null())
    assert embargoed.height > 0
    assert embargoed["step"].is_in(_embargo_steps(plan)).all()


def test_shares_roughly_follow_settings(plan: SplitPlan) -> None:
    rows = {name: plan.stats[name].modelled_rows for name in SPLITS}
    dense_total = sum(rows[name] for name in SPLITS[:-1])
    # Embargoes remove rows from training, so train is below 70 % but not far below.
    assert 0.45 < (rows["train_fit"] + rows["train_val"]) / dense_total < 0.70
    assert 0.08 < (rows["calib_a"] + rows["calib_b"]) / dense_total < 0.25


def test_stats_match_assigned_rows(plan: SplitPlan, frame: pl.DataFrame) -> None:
    modelled = assign_split(frame, plan).filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
    for name in SPLITS:
        part = modelled.filter(pl.col("split") == name)
        assert plan.stats[name].modelled_rows == part.height
        assert plan.stats[name].fraud == part["is_fraud"].sum()


def test_short_test_window_switches_to_short_bootstrap_blocks(plan: SplitPlan) -> None:
    assert plan.test_days < FIXTURE_SPLITS.min_test_days
    assert plan.bootstrap_block_steps == 6


def test_zero_delay_has_no_embargo(frame: pl.DataFrame) -> None:
    plan = plan_splits(frame, FIXTURE_SPLITS, 0)
    assert plan.embargoes == {"before_train_val": None, "before_calib_a": None}
    assert plan.ranges["train_fit"].end + 1 == plan.ranges["train_val"].start


def test_plan_round_trips_through_json(plan: SplitPlan, tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(plan.to_json(source_sha256="x"), encoding="utf-8")
    assert load_plan(path) == plan


def test_delay_too_long_for_the_data_fails(frame: pl.DataFrame) -> None:
    with pytest.raises(SplitError, match="empty"):
        plan_splits(frame, FIXTURE_SPLITS, 500)


def test_no_late_period_fails(frame: pl.DataFrame) -> None:
    settings = SplitSettings(dense_last_step=int(frame["step"].max()))  # type: ignore[arg-type]
    with pytest.raises(SplitError, match="no late period"):
        plan_splits(frame, settings, DELAY)


def test_shares_must_leave_room_for_test(frame: pl.DataFrame) -> None:
    settings = FIXTURE_SPLITS.model_copy(update={"train_share": 0.8, "calib_share": 0.2})
    with pytest.raises(SplitError, match="room for the test"):
        plan_splits(frame, settings, DELAY)


def _embargo_steps(plan: SplitPlan) -> list[int]:
    steps: list[int] = []
    for r in plan.embargoes.values():
        if r is not None:
            steps.extend(range(r.start, r.end + 1))
    return steps

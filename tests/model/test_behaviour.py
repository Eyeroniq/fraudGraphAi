"""Model behaviour tests (task 9.2) on the synthetic fixture.

Checks what a model must do whatever its AP: ignore account names, never treat an unknown
receiver or a known-fraud receiver as safer, flag the planted mule rings, hold its AP
floor (and not regress below the recorded champion), and keep conformal coverage. The
thresholds and the recorded numbers live in configs/validation.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

from fraudgraph.config import REPO_ROOT, SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.splits import assign_split, plan_splits
from fraudgraph.data.synth import SynthConfig, SynthResult, generate
from fraudgraph.features.backfill import backfill
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.features.spec import Event, FeatureSpec
from fraudgraph.models.compare import Ranked
from fraudgraph.pipeline import fixture_bundle
from fraudgraph.serving.scorer import Scorer

pytestmark = pytest.mark.model
LIMITS: dict[str, float] = yaml.safe_load(
    (REPO_ROOT / "configs" / "validation.yaml").read_text(encoding="utf-8")
)["behaviour"]


@dataclass
class Fixture:
    synth: SynthResult
    events: pl.DataFrame
    table: pl.DataFrame  # engine features with `split`, `name_dest` and `type`
    plain: Path  # the bundle CI ships (T + O features)
    graph: Path  # T + O + G1, like the champion


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    cfg = SynthConfig()
    synth = generate(cfg)
    events = to_canonical(synth.frame)
    table, _ = backfill(events, FeatureSpec())
    plan = plan_splits(events, SplitSettings(dense_last_step=(cfg.days - cfg.sparse_days) * 24), 12)
    table = assign_split(table, plan).join(
        events.select("txn_id", "name_dest", "type"), on="txn_id", how="left"
    )
    root = tmp_path_factory.mktemp("bundles")
    return Fixture(
        synth,
        events,
        table,
        fixture_bundle(root / "plain", trials=2),
        fixture_bundle(root / "graph", trials=2, graph=True),
    )


def event(row: dict[str, Any], **overrides: Any) -> Event:
    fields = {k: row[k] for k in ("txn_id", "step", "type", "amount", "name_orig", "name_dest")}
    return Event(**(fields | overrides))


def raw_scores(scorer: Scorer, table: pl.DataFrame) -> np.ndarray:
    return np.asarray(scorer.booster.predict(frame_to_matrix(table, scorer.features, 168)))


# ------------------------------------------------------------------ invariance


def test_renaming_accounts_does_not_change_any_score(fx: Fixture) -> None:
    """Account IDs are labels, not features: a consistent renaming (keeping the C / M
    prefix) must give bit-identical scores for the whole stream."""
    rng = np.random.default_rng(3)
    mapping: dict[str, str] = {}

    def rename(account: str) -> str:
        if account not in mapping:
            mapping[account] = f"{account[0]}{10**9 + 11 * len(mapping) + int(rng.integers(0, 7))}"
        return mapping[account]

    original, renamed = Scorer(fx.graph), Scorer(fx.graph)
    for row in fx.events.head(3_000).iter_rows(named=True):
        a = original.score(event(row)).p_fraud
        b = renamed.score(
            event(row, name_orig=rename(row["name_orig"]), name_dest=rename(row["name_dest"]))
        ).p_fraud
        assert a == b, row["txn_id"]


# ------------------------------------------------------------------ directional


@pytest.fixture(scope="module")
def warmed(fx: Fixture) -> tuple[Scorer, list[str], list[dict[str, Any]]]:
    """A graph scorer whose state holds steps 1-150, legit destinations with history, and
    modelled payments at step 151 to probe with."""
    scorer = Scorer(fx.graph)
    history = fx.events.filter(pl.col("step") <= 150)
    for row in history.iter_rows(named=True):
        scorer.engine.apply(event(row))
    known = (
        history.filter(~pl.col("is_fraud") & pl.col("name_dest").str.starts_with("C"))
        .group_by("name_dest")
        .len()
        .filter(pl.col("len") >= 3)
        .sort("name_dest")["name_dest"]
        .to_list()
    )
    probes = (
        fx.events.filter((pl.col("step") == 151) & pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
        .head(150)
        .to_dicts()
    )
    return scorer, known, probes


def test_a_never_seen_destination_does_not_lower_risk(
    warmed: tuple[Scorer, list[str], list[dict[str, Any]]],
) -> None:
    scorer, known, probes = warmed
    deltas = []
    for i, row in enumerate(probes):
        before = scorer.probability(event(row, name_dest=known[i % len(known)]))
        after = scorer.probability(event(row, name_dest=f"C88888{i:05d}"))
        deltas.append(after - before)
    deltas_ = np.array(deltas)
    assert (deltas_ >= -1e-12).all(), f"{int((deltas_ < -1e-12).sum())} probes got safer"
    assert deltas_.mean() > 0  # and on average it is riskier


def test_a_known_fraud_neighbour_does_not_lower_risk(
    warmed: tuple[Scorer, list[str], list[dict[str, Any]]],
) -> None:
    """The feature counts the label (from the next step on, D4) and the score never drops.
    On the fixture the model barely uses this feature (few labelled receivers in train)."""
    scorer, known, probes = warmed
    for i, row in enumerate(probes[:60]):
        dest = known[(i * 7) % len(known)]
        probe = event(row, step=152, name_dest=dest)
        before_f = scorer.engine.features(probe)["g_dest_known_fraud_in"]
        before = scorer.probability(probe)
        scorer.engine.apply_label(dest, True, 151, txn_id=f"BEH{i}")
        assert scorer.engine.features(probe)["g_dest_known_fraud_in"] == before_f + 1
        assert scorer.probability(probe) >= before - 1e-12


# ------------------------------------------------------------------ minimum functionality


def test_planted_mule_rings_are_flagged(fx: Fixture) -> None:
    scorer = Scorer(fx.graph)
    scores = raw_scores(scorer, fx.table)
    ring = fx.table["name_dest"].is_in(list(fx.synth.ring_destinations)).to_numpy()
    assert ring.sum() > 50
    share = float(np.mean(scores[ring] >= np.quantile(scores, 0.95)))
    assert share >= LIMITS["ring_top_share"], share


# ------------------------------------------------------------------ floors


def test_fixture_ap_floor_and_no_regression_against_the_champion(fx: Fixture) -> None:
    test = fx.table.filter(pl.col("split") == "test")
    y = test["is_fraud"].cast(pl.Int8).to_numpy()
    ap = Ranked(y, raw_scores(Scorer(fx.plain), test)).metrics()["ap"]
    assert ap >= LIMITS["fixture_ap_floor"], ap
    assert ap >= LIMITS["fixture_champion_ap"] - LIMITS["ap_tolerance"], ap


def test_graph_features_do_not_hurt_on_the_fixture(fx: Fixture) -> None:
    test = fx.table.filter(pl.col("split") == "test")
    y = test["is_fraud"].cast(pl.Int8).to_numpy()
    plain = Ranked(y, raw_scores(Scorer(fx.plain), test)).metrics()["ap"]
    graph = Ranked(y, raw_scores(Scorer(fx.graph), test)).metrics()["ap"]
    assert graph >= plain - LIMITS["ap_tolerance"], (plain, graph)


# ------------------------------------------------------------------ conformal


@pytest.mark.parametrize("which", ["plain", "graph"])
def test_class_conditional_coverage_holds_on_test(fx: Fixture, which: str) -> None:
    scorer = Scorer(getattr(fx, which))
    assert scorer.conformal is not None
    test = fx.table.filter(pl.col("split") == "test")
    y = test["is_fraud"].cast(pl.Int8).to_numpy()
    sets = scorer.conformal.sets(scorer.calibrator(raw_scores(scorer, test)))
    target = 1 - scorer.conformal.alpha
    for label in (0, 1):
        coverage = float(sets[y == label, label].mean())
        assert abs(coverage - target) <= LIMITS["coverage_tolerance"], (label, coverage)

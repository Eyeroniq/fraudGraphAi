from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from fraudgraph.config import REPO_ROOT
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.decision.costs import CostParams
from fraudgraph.drift.detectors import PerformanceMonitor
from fraudgraph.drift.psi_ks import (
    InputDriftMonitor,
    PredictionDriftMonitor,
    benjamini_hochberg,
    bin_shares,
    psi,
)
from fraudgraph.drift.scenarios import KINDS, Scenario, apply, load_scenarios
from fraudgraph.features.backfill import backfill
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.online import study
from fraudgraph.online.replay import LabelConfig, label_schedule, replay


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return to_canonical(generate().frame)


# ------------------------------------------------------------------ PSI / KS / BH


def test_psi_is_zero_for_identical_inputs_and_matches_hand_value() -> None:
    e = np.array([0.25, 0.25, 0.5])
    assert psi(e, e) == pytest.approx(0.0, abs=1e-12)
    a = np.array([0.5, 0.25, 0.25])
    hand = (0.5 - 0.25) * np.log(0.5 / 0.25) + (0.25 - 0.5) * np.log(0.25 / 0.5)
    assert psi(e, a) == pytest.approx(hand)


def test_bin_shares_puts_nan_last_and_sums_to_one() -> None:
    shares = bin_shares(np.array([0.0, 1.0, 2.0, 3.0, np.nan]), np.array([1.5]))
    assert shares.tolist() == pytest.approx([0.4, 0.4, 0.2])


def test_benjamini_hochberg_matches_hand_example() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8
    assert not benjamini_hochberg(np.array([0.9, 0.8]), 0.05).any()


def test_input_monitor_alarms_on_shift_and_not_on_same_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    edges = np.quantile(ref, np.linspace(0, 1, 11))[1:-1]
    stats = {"f": {"edges": edges.tolist(), "shares": bin_shares(ref, edges)[:-1].tolist()}}
    stats["f"]["nan_share"] = 0.0
    monitor = InputDriftMonitor(stats, {"f": ref})
    same = monitor.evaluate({"f": rng.normal(size=5_000)})
    moved = monitor.evaluate({"f": rng.normal(1.0, 1.0, size=5_000)})
    assert not same["alarm"]
    assert same["max_psi"] < 0.02
    assert moved["alarm"]
    assert moved["alarm_features"] == ["f"]
    assert moved["features"][0]["test"] == "ks"


def test_prediction_monitor_flags_alert_rate_jump() -> None:
    rng = np.random.default_rng(1)
    ref = rng.beta(1, 20, 10_000)
    monitor = PredictionDriftMonitor(ref, np.array([0.02, 0.021, 0.019, 0.02]))
    assert not monitor.evaluate(rng.beta(1, 20, 3_000), 0.0205)["alarm"]
    assert monitor.evaluate(rng.beta(1, 20, 3_000), 0.05)["alarm"]


def test_performance_monitor_fires_when_misses_rise() -> None:
    monitor = PerformanceMonitor()
    for k in range(200):
        monitor.update(k % 10 == 0, k)  # 10 % missed
    assert not monitor.alarms
    for k in range(200, 400):
        monitor.update(k % 10 != 0, k)  # 90 % missed
    assert monitor.alarms
    assert monitor.alarms[0]["step"] >= 200


# ------------------------------------------------------------------ scenarios


def test_config_scenarios_cover_every_kind() -> None:
    scenarios = load_scenarios(REPO_ROOT / "configs" / "drift_scenarios.yaml")
    assert {s.kind for s in scenarios} == set(KINDS)
    assert len({s.id for s in scenarios}) == len(scenarios)


def test_gradual_intensity_ramps() -> None:
    s = Scenario("x", "none", 10, 100, mode="gradual", ramp_steps=10)
    assert s.intensity(np.array([5, 10, 14, 19, 50, 101])).tolist() == pytest.approx(
        [0, 0.1, 0.5, 1.0, 1.0, 0]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_each_scenario_keeps_ids_unique_and_labels_injected_rows(
    events: pl.DataFrame, kind: str
) -> None:
    params = {"rings": 2, "mules": 3} if kind in ("mule_ring", "upi_mule_vpas") else {}
    s = Scenario("t", kind, 120, 190, seed=3, params=params)
    out = apply(events, s)
    frame = out.events
    assert frame["txn_id"].n_unique() == frame.height
    assert frame["step"].is_sorted()
    assert frame.columns == events.columns
    added = frame.filter(pl.col("txn_id").is_in(out.injected_txn_ids))
    assert added.height == len(out.injected_txn_ids)
    if kind in ("mule_ring", "upi_mule_vpas"):
        assert added.height > 0
        assert added["is_fraud"].all()
        assert added["step"].is_between(120, 190).all()
        assert not added["name_dest"].is_in(events["name_dest"].implode()).any()
    if kind == "volume_surge":
        assert added.height > 0
        assert not added["is_fraud"].any()
    if kind == "none":
        assert frame.equals(events.sort("step", maintain_order=True))
    before = frame.filter(pl.col("step") < 120)
    assert before.equals(events.filter(pl.col("step") < 120))


def test_amount_transforms_keep_balances_consistent(events: pl.DataFrame) -> None:
    out = apply(events, Scenario("t", "benign_inflation", 120, 190, params={"factor": 2.0}))
    changed = set(out.affected_txn_ids)
    joined = events.join(out.events, on="txn_id", suffix="_new").filter(
        pl.col("txn_id").is_in(changed)
    )
    assert joined.height > 0
    assert not joined["is_fraud"].any()
    ratio = joined["amount_new"] / joined["amount"]
    assert ratio.is_between(1.99, 2.01).all()


# ------------------------------------------------------------------ labels / replay


def test_label_schedule_delays_and_drops_some_fraud(events: pl.DataFrame) -> None:
    cfg = LabelConfig(delay_steps=10, jitter_steps=2, missing_fraud_share=0.5, seed=1)
    out = label_schedule(events, cfg)
    assert (out["label_step"] >= out["step"]).all()
    fraud = out.filter(pl.col("is_fraud"))
    assert 0 < fraud["observed_fraud"].sum() < fraud.height
    assert not out.filter(~pl.col("is_fraud"))["observed_fraud"].any()


def test_replay_partitions_events_and_labels_by_step(events: pl.DataFrame) -> None:
    frame = label_schedule(events, LabelConfig())
    batches = list(replay(frame, first_step=100, last_step=150))
    assert [b.step for b in batches] == list(range(100, 151))
    assert (
        sum(b.events.height for b in batches)
        == frame.filter(pl.col("step").is_between(100, 150)).height
    )
    for b in batches:
        assert (b.events["step"] == b.step).all()
        assert (b.labels["label_step"] == b.step).all()


# ------------------------------------------------------------------ study end to end


def test_study_runs_end_to_end_on_fixture(events: pl.DataFrame) -> None:
    spec = FeatureSpec()
    table, _ = backfill(events, spec)
    features = [c for c in table.columns if c.startswith(("t_", "o_", "g_"))][:12]
    train = table.filter(pl.col("step") <= 96)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7, "min_data_in_leaf": 20, "verbosity": -1},
        lgb.Dataset(
            frame_to_matrix(train, features, spec.horizon_steps),
            train["is_fraud"].to_numpy().astype(int),
            feature_name=features,
        ),
        num_boost_round=10,
    )
    cfg = study.StudyConfig(stream_first=97, stream_last=192, retrain_window=72, cooldown=24)
    river = study.warm_river(table, features, cfg, train_last_step=96)
    costs = CostParams(c_review=2_000.0, c_friction=20_000.0)
    labels = LabelConfig(delay_steps=6, jitter_steps=2)
    d0 = study.run_scenario(
        events, Scenario("D0", "none", 150, 192), booster, river, costs, cfg,
        study.Thresholds(), labels,
    )  # fmt: skip
    assert d0["windows"][0]["max_psi"] < 0.5  # first window lies inside the reference
    thresholds = study.tune_on_d0(d0)
    assert thresholds.input_psi >= 0.2
    ring = study.run_scenario(
        events, Scenario("D3", "mule_ring", 150, 192, params={"rings": 3}), booster, river,
        costs, cfg, thresholds, labels,
    )  # fmt: skip
    assert set(ring["policies"]) == set(study.POLICIES)
    assert ring["injected_rows"] == 3 * 3 * 15
    assert ring["policies"]["static"]["injected_caught_top1pct"] is not None
    assert len(ring["shadow_active"]) == len(ring["windows"])
    assert ring["policies"]["sliding"]["compute_s"] >= 0
    assert "_samples" in ring
    assert set(ring["_samples"]["post"]) == set(features)


def test_replayer_posts_every_event_and_label(events: pl.DataFrame) -> None:
    from services.replayer.main import run

    calls: list[str] = []

    def fake_post(url: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        calls.append(url)
        return (200, {"decision": "APPROVE"}) if url.endswith("/v1/score") else (404, {})

    summary = run(
        events, "http://api", first_step=100, last_step=110, labels_path="/v1/labels",
        post=fake_post,
    )  # fmt: skip
    expected = events.filter(pl.col("step").is_between(100, 110)).height
    assert summary["scored"] == expected == summary["decisions"]["APPROVE"]
    assert summary["labels"] > 0
    assert not summary["errors"]
    assert calls.count("http://api/v1/labels") == summary["labels"]


def test_windows_are_full_days_only() -> None:
    cfg = study.StudyConfig(stream_first=261, stream_last=408, window_steps=24)
    windows = study._windows(np.arange(261, 409), cfg)
    assert windows[-1] == (381, 404)
    assert all(b - a == 23 for a, b in windows)


def test_tuning_quiets_the_performance_monitor_on_the_null_stream() -> None:
    rng = np.random.default_rng(0)
    stream = [(s, bool(rng.random() < 0.3 + 0.2 * (s > 300))) for s in range(200, 400)]
    d0 = {"windows": [{"max_psi": 0.1, "score_psi": 0.05, "alert_rate_z": 1.0}] * 3}
    tuned = study.tune_on_d0({**d0, "label_stream": stream})
    alarms = study.run_performance_monitor(stream, tuned).alarms
    assert sum(a["detector"] == "page_hinkley" for a in alarms) <= 1
    assert sum(a["detector"] == "adwin" for a in alarms) <= 1

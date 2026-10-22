"""Drift and retraining study (tasks 6.3-6.6): `fgx drift-study`.

For every scenario (configs/drift_scenarios.yaml): drift the raw events, recompute the
engine features, replay the stream (steps 261-408) with delayed, noisy labels, and run:

    (a) static     the champion, never retrained
    (b) online     a river learner, warm-started on train_fit, test-then-train on labels
    (c) sliding    LightGBM retrained every 24 steps on the last 168 labelled steps
    (d) triggered  LightGBM retrained when a monitor alarm fires (48-step cooldown)
    (e) shadow     champion with a river challenger; the better one on recent labels serves

Monitors watch the static model: input PSI/KS/chi2 (+BH), score PSI and alert-rate z,
ADWIN/Page-Hinkley on missed fraud labels. Thresholds are tuned on D0 to at most one false
alarm in the stream (~0.9 simulated weeks). Everything is judged on the true labels.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from fraudgraph.decision.costs import CostParams, expected_cost_decisions, realised_costs
from fraudgraph.drift.detectors import PerformanceMonitor
from fraudgraph.drift.psi_ks import (
    InputDriftMonitor,
    PredictionDriftMonitor,
    matrix_stats,
    reference_alert_rates,
)
from fraudgraph.drift.scenarios import Scenario, apply
from fraudgraph.features.backfill import MODELLED_TYPES, backfill
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.models.calibrate import fit_calibrator
from fraudgraph.models.compare import Ranked
from fraudgraph.online import river_models
from fraudgraph.online.replay import LabelConfig, label_schedule
from fraudgraph.online.retrain import fit_like

POLICIES = ("static", "online", "sliding", "triggered", "shadow")


@dataclass(frozen=True)
class StudyConfig:
    stream_first: int = 261
    stream_last: int = 408
    window_steps: int = 24  # a whole day: shorter windows see the hour-of-day cycle as drift
    reference_steps: int = 48  # monitor reference = the stream's first two days (pre-drift)
    retrain_every: int = 24
    retrain_window: int = 168
    cooldown: int = 48
    legit_keep: float = 0.2  # legit subsample for LightGBM retrains (weight 1 / keep)
    river_model: str = "logreg"
    river_legit_keep: float = 0.05
    shadow_lookback: int = 72
    shadow_margin: float = 0.02
    alert_quantile: float = 0.98  # "caught" = score above this reference quantile
    seed: int = 0


@dataclass
class Thresholds:
    input_psi: float = 0.2
    score_psi: float = 0.2
    alert_z: float = 4.0
    ph_threshold: float = 20.0
    adwin_delta: float = 0.002


PH_GRID = (20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
ADWIN_GRID = (0.002, 0.0005, 0.0001, 0.00002)


# ------------------------------------------------------------------ helpers


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    return Ranked(y, s).metrics()["ap"] if y.sum() > 0 else float("nan")


def retrain(
    x: np.ndarray,
    observed: np.ndarray,
    label_step: np.ndarray,
    step: np.ndarray,
    now: int,
    booster: lgb.Booster,
    cfg: StudyConfig,
    rng: np.random.Generator,
) -> lgb.Booster:
    """LightGBM on labels known at `now`, from the last `retrain_window` steps."""
    known = (label_step <= now) & (step >= now - cfg.retrain_window)
    return fit_like(booster, x, observed, known, legit_keep=cfg.legit_keep, rng=rng)


def _windows(step: np.ndarray, cfg: StudyConfig) -> list[tuple[int, int]]:
    """Full windows only: a partial last window covers part of the day, and the hour-of-day
    feature alone then reads as drift."""
    starts = range(cfg.stream_first, cfg.stream_last - cfg.window_steps + 2, cfg.window_steps)
    return [(a, a + cfg.window_steps - 1) for a in starts]


# ------------------------------------------------------------------ one scenario


def run_scenario(
    events: pl.DataFrame,
    scenario: Scenario,
    booster: lgb.Booster,
    river_seed_model: Any,
    costs: CostParams,
    cfg: StudyConfig,
    thresholds: Thresholds,
    labels_cfg: LabelConfig,
) -> dict[str, Any]:
    if scenario.kind != "none" and scenario.start_step < cfg.stream_first + cfg.reference_steps:
        raise ValueError("a scenario must start after the monitor reference period")
    started = time.perf_counter()
    spec = FeatureSpec()
    features = booster.feature_name()
    drifted = apply(events.filter(pl.col("step") <= cfg.stream_last), scenario)
    injected = set(drifted.injected_txn_ids)
    table, _ = backfill(drifted.events, spec)
    table = table.join(
        label_schedule(drifted.events, labels_cfg).select(
            "txn_id", "amount", "label_step", "observed_fraud"
        ),
        on="txn_id",
        how="left",
        maintain_order="left",
    )
    backfill_s = time.perf_counter() - started

    hist = table.filter(pl.col("step") >= cfg.stream_first - cfg.retrain_window - 1)
    x_all = frame_to_matrix(hist, features, spec.horizon_steps)
    step_all = hist["step"].to_numpy()
    lstep_all = hist["label_step"].to_numpy()
    obs_all = hist["observed_fraud"].to_numpy().astype(bool)
    in_stream = (step_all >= cfg.stream_first) & (step_all <= cfg.stream_last)
    x = x_all[in_stream]
    step = step_all[in_stream]
    y = hist["is_fraud"].to_numpy().astype(np.int8)[in_stream]
    amount = hist["amount"].to_numpy()[in_stream]
    # hash lookup: np.isin on object arrays compares every row with every injected id
    is_injected = (
        hist["txn_id"].is_in(pl.Series(sorted(injected), dtype=pl.String).implode()).to_numpy()
    )[in_stream]
    rng = np.random.default_rng(cfg.seed)
    scores: dict[str, np.ndarray] = {}
    compute: dict[str, float] = {}

    # (a) static
    t0 = time.perf_counter()
    scores["static"] = np.asarray(booster.predict(x))
    compute["static"] = time.perf_counter() - t0

    # monitors on the static model
    ref_end = cfg.stream_first + cfg.reference_steps
    ref_mask = (step >= cfg.stream_first) & (step < ref_end)
    pre_mask = step < scenario.start_step if scenario.kind != "none" else ref_mask
    ref_scores = scores["static"][ref_mask]
    alert_cut = float(np.quantile(ref_scores, cfg.alert_quantile))
    windows = _windows(step, cfg)
    ref_rates = reference_alert_rates(
        scores["static"], step, cfg.stream_first, ref_end - 1, alert_cut
    )
    ref_rows = np.flatnonzero(ref_mask)
    sample_rows = np.random.default_rng(1).choice(
        ref_rows, min(50_000, len(ref_rows)), replace=False
    )
    reference_sample = {f: x[sample_rows, i] for i, f in enumerate(features)}
    ref_stats = matrix_stats(x[ref_mask], features)
    input_monitor = InputDriftMonitor(ref_stats, reference_sample, psi_alarm=thresholds.input_psi)
    pred_monitor = PredictionDriftMonitor(
        ref_scores, np.array(ref_rates), psi_alarm=thresholds.score_psi, z_alarm=thresholds.alert_z
    )
    window_rows = []
    for a, b in windows:
        m = (step >= a) & (step <= b)
        inp = input_monitor.evaluate({f: x[m, i] for i, f in enumerate(features)})
        pred = pred_monitor.evaluate(
            scores["static"][m], float(np.mean(scores["static"][m] >= alert_cut))
        )
        window_rows.append(
            {
                "start": a,
                "end": b,
                "max_psi": inp["max_psi"],
                "max_psi_feature": inp["max_psi_feature"],
                "input_alarm": inp["alarm"],
                "score_psi": pred["score_psi"],
                "alert_rate_z": pred["alert_rate_z"],
                "prediction_alarm": pred["alarm"],
                "static_ap": _ap(y[m], scores["static"][m]),
            }
        )
    # performance monitor: fraud labels as they arrive, missed = scored below alert cut
    fraud_labels = np.flatnonzero(obs_all & in_stream & (lstep_all <= cfg.stream_last))
    static_all = np.full(len(x_all), np.nan)
    static_all[in_stream] = scores["static"]
    label_stream = [
        (int(lstep_all[i]), bool(static_all[i] < alert_cut))
        for i in fraud_labels[np.argsort(lstep_all[fraud_labels], kind="stable")]
    ]
    perf = run_performance_monitor(label_stream, thresholds)

    alarm_steps = sorted(
        {w["end"] for w in window_rows if w["input_alarm"] or w["prediction_alarm"]}
        | {int(a["step"]) for a in perf.alarms}
    )

    # (c) sliding and (d) triggered retrains
    for name, times in (
        ("sliding", list(range(cfg.stream_first, cfg.stream_last + 1, cfg.retrain_every))),
        ("triggered", _with_cooldown(alarm_steps, cfg.cooldown)),
    ):
        t0 = time.perf_counter()
        out = scores["static"].copy()
        for k, t in enumerate(times):
            model = retrain(x_all, obs_all, lstep_all, step_all, t, booster, cfg, rng)
            end = times[k + 1] if k + 1 < len(times) else cfg.stream_last + 1
            m = (step >= t) & (step < end)
            out[m] = np.asarray(model.predict(x[m]))
        scores[name] = out
        compute[name] = time.perf_counter() - t0
        compute[f"{name}_retrains"] = float(len(times))

    # (b) online river, test-then-train with delayed labels; (e) shadow uses it
    t0 = time.perf_counter()
    model = copy.deepcopy(river_seed_model)
    online = np.zeros(len(x))
    queue_idx = np.flatnonzero(
        (step_all >= cfg.stream_first - cfg.retrain_window) & (lstep_all >= cfg.stream_first)
    )
    queue_idx = queue_idx[
        (obs_all[queue_idx]) | (rng.random(len(queue_idx)) < cfg.river_legit_keep)
    ]
    queue_idx = queue_idx[np.argsort(lstep_all[queue_idx], kind="stable")]
    q = 0
    for s in range(cfg.stream_first, cfg.stream_last + 1):
        while q < len(queue_idx) and lstep_all[queue_idx[q]] <= s:
            i = queue_idx[q]
            model.learn_one(
                river_models.row_dicts(x_all[i : i + 1], features, spec.horizon_steps)[0],
                bool(obs_all[i]),
            )
            q += 1
        idx = np.flatnonzero(step == s)
        if len(idx):
            online[idx] = river_models.predict(
                model, river_models.row_dicts(x[idx], features, spec.horizon_steps)
            )
    scores["online"] = online
    compute["online"] = time.perf_counter() - t0

    shadow = scores["static"].copy()
    active = []
    for a, b in windows:
        recent = (lstep_all[in_stream] <= a) & (step >= a - cfg.shadow_lookback) & (step < a)
        champ = _ap(obs_all[in_stream][recent].astype(int), scores["static"][recent])
        chall = _ap(obs_all[in_stream][recent].astype(int), scores["online"][recent])
        use = (
            "online"
            if (np.isfinite(chall) and np.isfinite(champ) and chall > champ + cfg.shadow_margin)
            else "static"
        )
        active.append(use)
        m = (step >= a) & (step <= b)
        shadow[m] = scores[use][m]
    scores["shadow"] = shadow
    compute["shadow"] = compute["online"]

    # ------------------------------------------------------------------ judge
    post = step >= scenario.start_step
    pre = ~post if scenario.kind != "none" else pre_mask
    results = {}
    for name in POLICIES:
        sv = scores[name]
        calibrator = fit_calibrator(sv[pre], y[pre])
        actions = expected_cost_decisions(calibrator(sv[post]), amount[post], costs)
        rolling = [
            _ap(y[(step >= a) & (step <= b)], sv[(step >= a) & (step <= b)]) for a, b in windows
        ]
        results[name] = {
            "ap_pre": _ap(y[pre], sv[pre]),
            "ap_post": _ap(y[post], sv[post]),
            "cost_post": float(realised_costs(actions, y[post], amount[post], costs).sum()),
            "rolling_ap": rolling,
            "time_to_recover_steps": _recovery(rolling, windows, scenario.start_step),
            "injected_caught_top1pct": _caught(sv[post], is_injected[post]),
            "compute_s": round(compute[name], 1),
        }
    first_alarm = _first_after(window_rows, perf.alarms, scenario.start_step)
    post_rows = np.flatnonzero(post)
    sample_post = np.random.default_rng(2).choice(
        post_rows, min(5_000, len(post_rows)), replace=False
    )
    samples = {
        "reference": reference_sample,
        "reference_stats": ref_stats,
        "post": {f: x[sample_post, i] for i, f in enumerate(features)},
    }
    return {
        "_samples": samples,
        "scenario": scenario.id,
        "kind": scenario.kind,
        "mode": scenario.mode,
        "start_step": scenario.start_step,
        "injected_rows": len(injected),
        "affected_rows": len(drifted.affected_txn_ids),
        "stream_rows": len(x),
        "stream_fraud": int(y.sum()),
        "windows": window_rows,
        "performance_alarms": perf.alarms,
        "label_stream": label_stream,
        "alarm_steps": alarm_steps,
        "detection": first_alarm,
        "false_alarms_before_start": _false_alarms(window_rows, perf.alarms, scenario),
        "shadow_active": active,
        "policies": results,
        "seconds": {
            "backfill": round(backfill_s, 1),
            "total": round(time.perf_counter() - started, 1),
        },
    }


def _with_cooldown(steps: list[int], cooldown: int) -> list[int]:
    out: list[int] = []
    for s in steps:
        if not out or s - out[-1] >= cooldown:
            out.append(s)
    return out


def _recovery(rolling: list[float], windows: list[tuple[int, int]], start: int) -> float | None:
    before = [v for v, (a, b) in zip(rolling, windows, strict=True) if b < start and np.isfinite(v)]
    after = [
        (a, v) for v, (a, b) in zip(rolling, windows, strict=True) if a >= start and np.isfinite(v)
    ]
    if not before or not after:
        return None
    level = 0.9 * float(np.mean(before))
    dropped = [a for a, v in after if v < level]
    if not dropped:
        return 0.0  # never dropped below 90 % of the pre-drift level
    for a, v in after:
        if a > dropped[0] and v >= level:
            return float(a - start)
    return None  # did not recover within the stream


def _caught(scores: np.ndarray, injected: np.ndarray) -> float | None:
    if not injected.any():
        return None
    return float(np.mean(scores[injected] >= np.quantile(scores, 0.99)))


def _first_after(
    windows: list[dict[str, Any]], perf: list[dict[str, float]], start: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("input_alarm", "prediction_alarm"):
        hits = [w["end"] for w in windows if w[key] and w["end"] >= start]
        out[key.replace("_alarm", "")] = (hits[0] - start) if hits else None
    perf_hits = [a["step"] for a in perf if a["step"] >= start]
    out["performance"] = (perf_hits[0] - start) if perf_hits else None
    return out


def _false_alarms(
    windows: list[dict[str, Any]], perf: list[dict[str, float]], s: Scenario
) -> dict[str, int]:
    cut = s.start_step if s.kind != "none" else 10**9
    return {
        "input": sum(1 for w in windows if w["input_alarm"] and w["end"] < cut),
        "prediction": sum(1 for w in windows if w["prediction_alarm"] and w["end"] < cut),
        "performance": sum(1 for a in perf if a["step"] < cut),
    }


def run_performance_monitor(
    label_stream: list[tuple[int, bool]], thresholds: Thresholds
) -> PerformanceMonitor:
    """Feed (label step, missed) pairs for arriving fraud labels to ADWIN / Page-Hinkley."""
    perf = PerformanceMonitor(
        adwin_delta=thresholds.adwin_delta, ph_threshold=thresholds.ph_threshold
    )
    for step, missed in label_stream:
        perf.update(missed, step)
    return perf


def tune_on_d0(d0: dict[str, Any], allowed: int = 1) -> Thresholds:
    """Most sensitive thresholds that leave each monitor at most `allowed` false alarms on
    the null stream (PSI / z cut just above the (allowed+1)-th largest D0 value; ADWIN and
    Page-Hinkley from a grid, replayed on D0's label stream)."""

    def cut(values: list[float], floor: float) -> float:
        v = sorted((x for x in values if np.isfinite(x)), reverse=True)
        return max(floor, (v[allowed] * 1.05) if len(v) > allowed else floor)

    def quiet(**kw: float) -> bool:
        alarms = run_performance_monitor(stream, Thresholds(**kw)).alarms
        return sum(1 for a in alarms if a["detector"] == detector) <= allowed

    windows = d0["windows"]
    stream = [(int(s), bool(m)) for s, m in d0["label_stream"]]
    detector = "page_hinkley"
    ph = next((v for v in PH_GRID if quiet(ph_threshold=v)), PH_GRID[-1])
    detector = "adwin"
    delta = next((v for v in ADWIN_GRID if quiet(adwin_delta=v)), ADWIN_GRID[-1])
    return Thresholds(
        input_psi=cut([w["max_psi"] for w in windows], 0.2),
        score_psi=cut([w["score_psi"] for w in windows], 0.2),
        alert_z=cut([abs(w["alert_rate_z"]) for w in windows], 4.0),
        ph_threshold=ph,
        adwin_delta=delta,
    )


def warm_river(
    table: pl.DataFrame, features: list[str], cfg: StudyConfig, train_last_step: int = 236
) -> Any:
    """River model warm-started on train_fit (all fraud + a legit sample)."""
    spec = FeatureSpec()
    train = table.filter(pl.col("step") <= train_last_step)
    train = (
        train.filter(pl.col("type").is_in(list(MODELLED_TYPES)))
        if "type" in train.columns
        else train
    )
    y = train["is_fraud"].to_numpy().astype(bool)
    rng = np.random.default_rng(cfg.seed)
    keep = y | (rng.random(len(y)) < 9 * y.sum() / max((~y).sum(), 1))
    x = frame_to_matrix(train.filter(pl.Series(keep)), features, spec.horizon_steps)
    rows = river_models.row_dicts(x, features, spec.horizon_steps)
    return river_models.warm_start(
        river_models.make_model(cfg.river_model, cfg.seed), rows, y[keep], cfg.seed
    )


def natural_late_period(
    table: pl.DataFrame,
    booster: lgb.Booster,
    thresholds: Thresholds,
    *,
    first: int = 409,
    last: int = 743,
    window: int = 24,
    ref_first: int = 261,
    ref_last: int = 329,
) -> list[dict[str, Any]]:
    """Monitors and static AP on PaySim's own (drifting) late period, no injection."""
    spec = FeatureSpec()
    features = booster.feature_name()
    ref = table.filter(pl.col("step").is_between(ref_first, ref_last))
    late = table.filter(pl.col("step").is_between(first, last))
    x_ref = frame_to_matrix(ref, features, spec.horizon_steps)
    x_late = frame_to_matrix(late, features, spec.horizon_steps)
    s_ref = np.asarray(booster.predict(x_ref))
    s_late = np.asarray(booster.predict(x_late))
    cut = float(np.quantile(s_ref, 0.98))
    rates = reference_alert_rates(s_ref, ref["step"].to_numpy(), ref_first, ref_last, cut)
    inp = InputDriftMonitor(
        matrix_stats(x_ref, features),
        {f: x_ref[:, i] for i, f in enumerate(features)},
        psi_alarm=thresholds.input_psi,
    )
    pred = PredictionDriftMonitor(
        s_ref, np.array(rates), psi_alarm=thresholds.score_psi, z_alarm=thresholds.alert_z
    )
    y = late["is_fraud"].to_numpy().astype(np.int8)
    st = late["step"].to_numpy()
    rows = []
    for a in range(first, last + 1, window):
        m = (st >= a) & (st < a + window)
        if not m.any():
            continue
        i = inp.evaluate({f: x_late[m, j] for j, f in enumerate(features)})
        p = pred.evaluate(s_late[m], float(np.mean(s_late[m] >= cut)))
        rows.append(
            {
                "start": a,
                "rows": int(m.sum()),
                "fraud_rate": float(y[m].mean()),
                "max_psi": i["max_psi"],
                "max_psi_feature": i["max_psi_feature"],
                "input_alarm": i["alarm"],
                "score_psi": p["score_psi"],
                "alert_rate_z": p["alert_rate_z"],
                "prediction_alarm": p["alarm"],
                "static_ap": _ap(y[m], s_late[m]),
            }
        )
    return rows


def write_plots(report: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = report["scenarios"]
    fig, axes = plt.subplots(
        2, (len(scenarios) + 1) // 2, figsize=(4 * ((len(scenarios) + 1) // 2), 6.5), squeeze=False
    )
    for ax, sc in zip(axes.ravel(), scenarios, strict=False):
        starts = [w["start"] for w in sc["windows"]]
        for name in POLICIES:
            ax.plot(starts, sc["policies"][name]["rolling_ap"], label=name, lw=1.3)
        ax.axvline(sc["start_step"], color="black", ls="--", lw=1)
        for a in sc["alarm_steps"]:
            ax.axvline(a, color="#c0392b", alpha=0.25, lw=3)
        ax.set_title(f"{sc['scenario']} {sc['kind']}", fontsize=9)
        ax.set_ylim(0, 1)
    for ax in axes.ravel()[len(scenarios) :]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Rolling AP per policy (dashed: drift start; red: monitor alarms)")
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_ap.png", dpi=110)

    late = report["natural_late_period"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(
        [w["start"] for w in late],
        [w["max_psi"] for w in late],
        marker="o",
        label="max feature PSI",
    )
    ax.plot(
        [w["start"] for w in late], [w["score_psi"] for w in late], marker="s", label="score PSI"
    )
    ax.axhline(report["thresholds"]["input_psi"], color="grey", ls=":", label="tuned alarm")
    ax.set(xlabel="step", ylabel="PSI", title="PaySim's own late period: natural drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "natural_late_psi.png", dpi=110)
    plt.close("all")

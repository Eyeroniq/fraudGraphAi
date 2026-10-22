"""Worker jobs (task 7.4). Each tick runs whatever has become due:

- **replayer** (demo, `/v1/sim/*`): streams the next steps of the event file into the API,
  with the chosen drift scenario applied, and posts the labels that become known at each
  step (delay and missing chargebacks as in `online/replay.py`);
- **drift monitor**: every completed day-long window of scored traffic is compared with
  the stream's first two days (PSI/KS/chi-squared with BH, score PSI, alert-rate z) and
  written to `drift_windows` / `alarms`, with an Evidently report per window;
- **performance monitor**: ADWIN / Page-Hinkley on arriving fraud labels the champion
  approved (missed fraud). In docs/drift_study.md this is the monitor that sees damage;
- **retrain**: the policy from 6.6, daily on the last week of known labels. The newest
  labelled day is held out to calibrate (and conformalise) the new model, which becomes a
  hashed bundle registered as the challenger;
- **judge**: once a challenger has enough labelled shadow decisions, it is promoted
  through the API if its realised cost is lower than the champion's, else archived.

The river online learner is not run here: in the study it was the costliest policy on
every scenario and never won the shadow switch.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from fraudgraph.config import REPO_ROOT, WorkerSettings
from fraudgraph.decision import conformal
from fraudgraph.decision.costs import ACTIONS, realised_costs
from fraudgraph.drift.detectors import PerformanceMonitor
from fraudgraph.drift.psi_ks import (
    InputDriftMonitor,
    PredictionDriftMonitor,
    matrix_stats,
    reference_alert_rates,
)
from fraudgraph.drift.scenarios import Scenario, apply, load_scenarios
from fraudgraph.features.matrix import row_to_vector
from fraudgraph.models.bundle import write_bundle
from fraudgraph.models.calibrate import fit_calibrator
from fraudgraph.models.compare import Ranked
from fraudgraph.online.replay import LabelConfig, label_schedule
from fraudgraph.online.retrain import fit_like
from fraudgraph.serving.scorer import BundleModel
from services.api.db import DecisionLog

log = logging.getLogger("fraudgraph.worker")

# (path, json body, key scope) -> (HTTP status, json body)
Post = Callable[[str, Any, str], tuple[int, Any]]
EVENT_FIELDS = ("txn_id", "step", "type", "amount", "name_orig", "name_dest")
BATCH = 1_000
SAMPLE = 5_000
ALERT_QUANTILE = 0.98


@dataclass
class Reference:
    """The stream's first days as the monitors see them (model matrix, scores)."""

    key: tuple[int, int, str]
    inputs: InputDriftMonitor
    predictions: PredictionDriftMonitor
    alert_cut: float
    sample: dict[str, np.ndarray]


@dataclass
class Worker:
    log: DecisionLog
    settings: WorkerSettings
    bundles_root: Path
    reports_dir: Path
    post: Post | None = None
    events: pl.DataFrame | None = None
    seed: int = 0
    _champion: BundleModel | None = None
    _reference: Reference | None = None
    _perf: PerformanceMonitor | None = None
    _perf_upto: int | None = None
    _next_retrain: int | None = None
    _stream: tuple[tuple[Any, ...], pl.DataFrame] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ tick

    def tick(self) -> dict[str, Any]:
        """Run every job that is due; returns what each did (also kept in `history`)."""
        done: dict[str, Any] = {"replayed": self.replay()}
        latest = self.log.latest_step()
        if latest is not None and self.champion() is not None:
            done["drift_windows"] = self.monitor_drift(latest)
            done["performance_alarms"] = self.monitor_performance(latest)
            done["retrain"] = self.retrain(latest)
            done["judged"] = self.judge()
        self.history.append(done)
        return done

    def champion(self) -> BundleModel | None:
        rows = [m for m in self.log.models() if m.stage == "champion"]
        if not rows:
            return None
        if self._champion is None or self._champion.version != rows[0].version:
            self._champion = BundleModel(self.bundles_root / rows[0].bundle_uri)
            self._reference = None  # monitors follow the model being served
        return self._champion

    # ------------------------------------------------------------------ replayer

    def replay(self) -> list[int]:
        """Stream the next `steps_per_tick` steps if the simulation is running."""
        sim = self.log.sim()
        if sim is None or not sim.running or self.events is None or self.post is None:
            return []
        if sim.next_step > sim.last_step:
            self.log.set_sim(running=False)
            return []
        frame = self._drifted(sim.scenario, sim.scenario_start, sim.first_step, sim.last_step)
        steps = list(
            range(sim.next_step, min(sim.next_step + sim.steps_per_tick, sim.last_step + 1))
        )
        for step in steps:
            events = frame.filter(pl.col("step") == step).select(EVENT_FIELDS).to_dicts()
            for i in range(0, len(events), BATCH):
                self._send("/v1/score/batch", {"transactions": events[i : i + BATCH]}, step)
            labels = [
                {"txn_id": t, "observed_fraud": bool(f), "step": step, "source": "simulator"}
                for t, f in frame.filter(pl.col("label_step") == step)
                .select("txn_id", "observed_fraud")
                .iter_rows()
            ]
            for i in range(0, len(labels), BATCH):
                self._send("/v1/labels/batch", {"labels": labels[i : i + BATCH]}, step)
            self.log.set_sim(next_step=step + 1)  # progress survives a failed later step
        return steps

    def _send(self, path: str, body: Any, step: int, attempts: int = 8) -> None:
        """POST with backoff on rate limiting (the API's limits apply to the replayer too)."""
        assert self.post is not None
        for attempt in range(attempts):
            status, reply = self.post(path, body, "write")
            if status == 200:
                failed = reply.get("failed", 0) if isinstance(reply, dict) else 0
                if failed:  # per-item errors (e.g. out of order) do not fail the batch
                    first: dict[str, Any] = next(
                        (r for r in reply.get("results", []) if "error" in r), {}
                    )
                    log.warning(
                        "%s at step %s: %s of the batch rejected (%s)",
                        path,
                        step,
                        failed,
                        first.get("error", {}).get("code", "?"),
                    )
                return
            if status != 429:
                break
            time.sleep(min(2.0**attempt, 30.0))
        raise RuntimeError(f"{path} at step {step} failed: {status} {reply}")

    def _drifted(self, scenario_id: str, start: int, first: int, last: int) -> pl.DataFrame:
        """Events of [first, last] with the scenario applied from `start`, and labels."""
        key = (scenario_id, start, first, last)
        if self._stream is not None and self._stream[0] == key:
            return self._stream[1]
        assert self.events is not None
        events = self.events.filter(pl.col("step").is_between(first, last))
        scenario = scenario_by_id(scenario_id)
        if scenario.kind != "none":
            scenario = dataclasses.replace(scenario, start_step=start, end_step=last)
            events = apply(events, scenario).events
        labels = LabelConfig(delay_steps=self.settings.label_delay_steps, seed=self.seed)
        frame = label_schedule(events, labels)
        self._stream = (key, frame)
        return frame

    # ------------------------------------------------------------------ monitors

    def _matrix(self, features: list[dict[str, Any]], model: BundleModel) -> np.ndarray:
        """Stored features (null = no history) -> the model matrix with the shared fills."""
        if not features:
            return np.zeros((0, len(model.features)))
        rows = [
            row_to_vector(
                {k: float("nan") if f.get(k) is None else float(f[k]) for k in model.features},
                model.features,
                model.spec.horizon_steps,
            )[0]
            for f in features
        ]
        return np.vstack(rows)

    def reference(self, first: int, model: BundleModel) -> Reference | None:
        last = first + self.settings.reference_steps - 1
        key = (first, last, model.version)
        if self._reference is not None and self._reference.key == key:
            return self._reference
        rows = self.log.scored(first, last)
        if len(rows) < 100:
            return None
        steps = np.array([r[0] for r in rows])
        x = self._matrix([r[1] for r in rows], model)
        scores = np.array([r[2] for r in rows])
        cut = float(np.quantile(scores, ALERT_QUANTILE))
        rng = np.random.default_rng(self.seed)
        pick = rng.choice(len(rows), min(50_000, len(rows)), replace=False)
        sample = {f: x[pick, i] for i, f in enumerate(model.features)}
        s = self.settings
        self._reference = Reference(
            key=key,
            inputs=InputDriftMonitor(
                matrix_stats(x, model.features), sample, psi_alarm=s.input_psi
            ),
            predictions=PredictionDriftMonitor(
                scores,
                np.array(reference_alert_rates(scores, steps, first, last, cut) or [0.0]),
                psi_alarm=s.score_psi,
                z_alarm=s.alert_z,
            ),
            alert_cut=cut,
            sample=sample,
        )
        return self._reference

    def monitor_drift(self, latest: int) -> list[tuple[int, int]]:
        """Evaluate every window that is complete (a later step has been scored)."""
        model, first = self.champion(), self.log.first_step()
        if model is None or first is None:
            return []
        ref_end = first + self.settings.reference_steps - 1
        start = max((self.log.last_drift_end() or ref_end) + 1, ref_end + 1)
        size = self.settings.window_steps
        done = []
        while start + size - 1 < latest:
            reference = self.reference(first, model)
            if reference is None:
                break
            end = start + size - 1
            self._drift_window(start, end, model, reference)
            done.append((start, end))
            start = end + 1
        return done

    def _drift_window(self, start: int, end: int, model: BundleModel, ref: Reference) -> None:
        rows = self.log.scored(start, end)
        windows: list[dict[str, Any]] = []
        alarms: list[dict[str, Any]] = []
        if rows:
            x = self._matrix([r[1] for r in rows], model)
            scores = np.array([r[2] for r in rows])
            current = {f: x[:, i] for i, f in enumerate(model.features)}
            inputs = ref.inputs.evaluate(current)
            preds = ref.predictions.evaluate(scores, float(np.mean(scores >= ref.alert_cut)))
            for row in inputs["features"]:
                windows.append(
                    {
                        "start_step": start,
                        "end_step": end,
                        "feature": row["feature"],
                        "psi": row["psi"],
                        "ks_stat": row["stat"],
                        "ks_p": row["p"],
                        "alarm": row["alarm"],
                    }
                )
                if row["alarm"]:
                    alarms.append(
                        {
                            "kind": "input",
                            "detector": row["test"],
                            "feature": row["feature"],
                            "step": end,
                            "details": {"psi": row["psi"], "p": row["p"], "window": [start, end]},
                        }
                    )
            if preds["alarm"]:
                alarms.append(
                    {
                        "kind": "prediction",
                        "detector": "score_psi_alert_rate",
                        "feature": None,
                        "step": end,
                        "details": {**preds, "window": [start, end]},
                    }
                )
            self._evidently(start, end, ref, current, len(rows))
        windows.append(  # summary row: also marks the window as done when it was empty
            {
                "start_step": start,
                "end_step": end,
                "feature": "__score__",
                "psi": preds["score_psi"] if rows else None,
                "ks_stat": None,
                "ks_p": None,
                "alarm": bool(rows) and bool(preds["alarm"]),
            }
        )
        self.log.add_drift(windows, alarms)
        for a in alarms:
            log.warning("drift alarm %s %s at step %s", a["kind"], a["feature"] or "", a["step"])

    def _evidently(
        self, start: int, end: int, ref: Reference, current: dict[str, np.ndarray], n: int
    ) -> None:
        if not self.settings.evidently:
            return
        try:
            from fraudgraph.drift.evidently_reports import drift_report
        except ImportError:  # the slim image has no Evidently
            return
        rng = np.random.default_rng(start)
        pick = rng.choice(n, min(SAMPLE, n), replace=False)
        ref_n = len(next(iter(ref.sample.values())))
        ref_pick = rng.choice(ref_n, min(SAMPLE, ref_n), replace=False)
        try:
            drift_report(
                {k: v[ref_pick] for k, v in ref.sample.items()},
                {k: v[pick] for k, v in current.items()},
                self.reports_dir / "evidently" / f"window-{start}-{end}",
            )
        except Exception as exc:  # a report is a nice-to-have, never fatal
            log.warning("evidently report for %s-%s failed: %s", start, end, exc)

    def monitor_performance(self, latest: int) -> list[dict[str, Any]]:
        """Feed fraud labels known up to the previous step (later ones may still come)."""
        if self._perf is None:
            s = self.settings
            self._perf = PerformanceMonitor(adwin_delta=s.adwin_delta, ph_threshold=s.ph_threshold)
            # After a restart, start fresh instead of raising the old alarms again.
            seen = self.log.alarm_counts().get("performance", 0)
            first = self.log.first_step() or latest
            self._perf_upto = latest - 1 if seen else first - 1
        assert self._perf_upto is not None
        upto = latest - 1
        before = len(self._perf.alarms)
        for step, decision in self.log.fraud_labels(self._perf_upto, upto):
            self._perf.update(decision == "APPROVE", step)
        self._perf_upto = max(self._perf_upto, upto)
        fired = self._perf.alarms[before:]
        if fired:
            self.log.add_drift(
                [],
                [
                    {
                        "kind": "performance",
                        "detector": a["detector"],
                        "feature": None,
                        "step": int(a["step"]),
                        "details": {"labels_seen": a["labels_seen"]},
                    }
                    for a in fired
                ],
            )
            for a in fired:
                log.warning("performance alarm %s at step %s", a["detector"], int(a["step"]))
        return fired

    # ------------------------------------------------------------------ retrain

    def retrain(self, latest: int) -> dict[str, Any] | None:
        """Daily: a challenger from the last week of known labels (6.6)."""
        first = self.log.first_step()
        champion = self.champion()
        if first is None or champion is None:
            return None
        s = self.settings
        if self._next_retrain is None:
            self._next_retrain = first + s.retrain_every
        if latest < self._next_retrain:
            return None
        self._next_retrain = latest + s.retrain_every
        rows = self.log.labelled(latest - s.retrain_window, latest, known_by=latest)
        if not rows:
            return {"step": latest, "skipped": "no labels"}
        steps = np.array([r[0] for r in rows])
        y = np.array([r[3] for r in rows], dtype=bool)
        calib = steps > steps.max() - s.calib_steps
        fit = ~calib
        n_fit, n_calib = int(y[fit].sum()), int(y[calib].sum())
        if n_fit < s.min_fraud_labels or n_calib < 5:
            return {"step": latest, "skipped": f"too few fraud labels ({n_fit}, {n_calib})"}
        x = self._matrix([r[1] for r in rows], champion)
        rng = np.random.default_rng(self.seed + latest)
        booster = fit_like(champion.booster, x, y, fit, legit_keep=s.legit_keep, rng=rng)
        raw = np.asarray(booster.predict(x[calib]))
        calibrator = fit_calibrator(raw, y[calib].astype(int))
        p_calib = calibrator(raw)
        metrics = {
            "retrained_at_step": latest,
            "fit_rows": int(fit.sum()),
            "fit_fraud": int(y[fit].sum()),
            "calib_rows": int(calib.sum()),
            "calib_fraud": int(y[calib].sum()),
            "calib_ap": Ranked(y[calib].astype(int), p_calib).metrics()["ap"],
            "champion_calib_ap": Ranked(
                y[calib].astype(int),
                champion.calibrator(np.asarray(champion.booster.predict(x[calib]))),
            ).metrics()["ap"],
        }
        path = self._write_bundle(champion, booster, calibrator, p_calib, y[calib], x[fit], metrics)
        for old in self.log.challengers():  # only the newest challenger competes
            self.log.retire(old.version)
        version = json.loads((path / "manifest.json").read_text(encoding="utf-8"))["version"]
        self.log.register_challenger(version, metrics, path.name)
        log.info("challenger %s registered (calib AP %.3f)", version, metrics["calib_ap"])
        return {"step": latest, "challenger": version, **metrics}

    def _write_bundle(
        self,
        champion: BundleModel,
        booster: Any,
        calibrator: Any,
        p_calib: np.ndarray,
        y_calib: np.ndarray,
        x_fit: np.ndarray,
        metrics: dict[str, Any],
    ) -> Path:
        base = str(champion.manifest["model"]).split("-r")[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            booster.save_model(str(tmp_dir / "model.txt"))
            joblib.dump(calibrator, tmp_dir / "calibrator.joblib")
            conformal_path = None
            if champion.conformal is not None:
                c = champion.conformal
                model = conformal.fit(
                    p_calib, y_calib, alpha=c.alpha, score=c.score, mondrian=c.mondrian
                )
                conformal_path = tmp_dir / "conformal.json"
                conformal.save(model, conformal_path)
            spec = json.loads(
                (champion.bundle_dir / "feature_spec.json").read_text(encoding="utf-8")
            )
            return write_bundle(
                self.bundles_root,
                model=f"{base}-r{metrics['retrained_at_step']}",
                booster_path=tmp_dir / "model.txt",
                calibrator_path=tmp_dir / "calibrator.joblib",
                policy_path=champion.bundle_dir / "policy.json",
                conformal_path=conformal_path,
                explain_dir=champion.bundle_dir,
                feature_spec=spec,
                train_features=pl.DataFrame(
                    {f: x_fit[:, i] for i, f in enumerate(champion.features)}
                ),
                metrics=metrics,
                data_sha256="stream",
                git_sha=str(champion.manifest.get("git_sha", "unknown")),
            )

    # ------------------------------------------------------------------ promotion

    def judge(self) -> list[dict[str, Any]]:
        """Promote a challenger that beats the champion on labelled shadow traffic."""
        champion = self.champion()
        if champion is None:
            return []
        out = []
        for model in self.log.challengers():
            pairs = self.log.shadow_pairs(model.version, since_step=0)
            if len(pairs) < self.settings.min_shadow_labels:
                continue
            amount = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            champ = np.array([ACTIONS.index(p[2]) for p in pairs])
            chall = np.array([ACTIONS.index(p[3]) for p in pairs])
            cost_champion = float(realised_costs(champ, y, amount, champion.costs).sum())
            cost_challenger = float(realised_costs(chall, y, amount, champion.costs).sum())
            verdict = {
                "version": model.version,
                "labelled": len(pairs),
                "cost_champion": cost_champion,
                "cost_challenger": cost_challenger,
            }
            if cost_challenger < cost_champion and self.settings.auto_promote and self.post:
                status, body = self.post(f"/v1/models/{model.version}/promote", None, "admin")
                verdict["promoted"] = status == 200
                if status != 200:
                    log.error("promotion of %s failed: %s %s", model.version, status, body)
            elif cost_challenger >= cost_champion:
                self.log.retire(model.version)
                verdict["retired"] = True
            out.append(verdict)
        return out


def scenario_by_id(scenario_id: str) -> Scenario:
    for scenario in load_scenarios(REPO_ROOT / "configs" / "drift_scenarios.yaml"):
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(scenario_id)

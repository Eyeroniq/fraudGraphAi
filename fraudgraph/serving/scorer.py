"""Pure inference (task 2.1): bundle + feature engine -> decision.

Used by the API and by tests. The bundle is verified on load (every file's hash) and its
feature engine version must equal this code's, so a model can never be served with
features computed differently from how it was trained (D3).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from fraudgraph.decision import conformal
from fraudgraph.decision.conformal import CLASSES
from fraudgraph.decision.costs import ACTIONS, CostParams, expected_costs
from fraudgraph.decision.policy import OnlinePolicy, PolicyConfig
from fraudgraph.explain.reasons import ReasonCodes, contributions, gap_after_top_k
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.matrix import row_to_vector
from fraudgraph.features.spec import ENGINE_VERSION, Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore, StateStore
from fraudgraph.models.bundle import verify_bundle

MODELLED_TYPES = frozenset({"TRANSFER", "CASH_OUT"})
# Shown when a payment is not approved but no feature raised its risk (a policy decision).
POLICY_REASONS = {
    "model_uncertain": (
        "The model is not confident either way, so a check costs less than a mistake"
    ),
    "conformal_fraud": "The payment looks like confirmed fraud patterns with high confidence",
    "capacity_overflow": "The review team is at capacity today; the next cheapest action was used",
    "expected_cost": "A check is cheaper than the expected loss for this amount",
}


class BundleMismatchError(RuntimeError):
    """The bundle was built with a different feature engine than this code."""


@dataclass(frozen=True)
class Decision:
    txn_id: str
    model_version: str
    p_fraud: float
    decision: str
    decision_reason: str
    expected_cost: dict[str, float]
    prediction_set: list[str] | None = None  # conformal set; None without a conformal model
    reasons: list[dict[str, Any]] = field(default_factory=list)
    explanation_stable: bool | None = None  # None when no reasons were computed
    latency_ms: float = 0.0
    # Engine output the model saw (NaN = no history); empty for unmodelled types.
    features: dict[str, float] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Explanation:
    """Full attribution of one stored decision (A7 `/explain`)."""

    contributions: dict[str, float]  # per feature, log-odds
    bias: float
    groups: dict[str, float]  # per reason code
    reasons: list[dict[str, Any]]
    explanation_stable: bool


def check_engine_compatible(bundle_version: str, code_version: str) -> None:
    """Same MAJOR, and the code's MINOR at least the bundle's (it only adds features)."""
    b_major, b_minor = (int(x) for x in bundle_version.split(".")[:2])
    c_major, c_minor = (int(x) for x in code_version.split(".")[:2])
    if b_major != c_major or c_minor < b_minor:
        raise BundleMismatchError(f"bundle engine {bundle_version} vs code engine {code_version}")


class BundleModel:
    """A verified bundle that decides from engine features: the champion's model part, and
    on its own a shadow challenger scored on the champion's features (7.5)."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = verify_bundle(bundle_dir)
        spec_data = json.loads((bundle_dir / "feature_spec.json").read_text(encoding="utf-8"))
        check_engine_compatible(spec_data["engine_version"], ENGINE_VERSION)
        self.spec = FeatureSpec(
            windows=tuple(spec_data["windows"]),
            horizon_steps=spec_data["horizon_steps"],
            label_delay_steps=spec_data.get("label_delay_steps", 24),
        )
        self.version: str = self.manifest["version"]
        self.features: list[str] = self.manifest["features"]
        self.booster = lgb.Booster(model_file=str(bundle_dir / "model.txt"))
        self.calibrator = joblib.load(bundle_dir / "calibrator.joblib")
        policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
        self.costs = CostParams(**policy["costs"])
        conformal_path = bundle_dir / "conformal.json"
        self.conformal = conformal.load(conformal_path) if conformal_path.is_file() else None
        self.policy = OnlinePolicy(
            PolicyConfig(
                costs=self.costs,
                conformal=self.conformal,
                uncertain_action=ACTIONS.index(policy.get("uncertain_action", "REVIEW")),
                review_capacity_per_day=policy.get("review_capacity_per_day"),
            )
        )
        codes_path = bundle_dir / "reason_codes.yaml"
        self.reason_codes = (
            ReasonCodes.load(codes_path, self.features) if codes_path.is_file() else None
        )
        explain_path = bundle_dir / "explain.json"
        self.stability_gap = (
            float(json.loads(explain_path.read_text(encoding="utf-8"))["stability_gap_threshold"])
            if explain_path.is_file()
            else 0.0
        )

    def check_features(self, available: Sequence[str]) -> None:
        missing = set(self.features) - set(available)
        if missing:
            raise BundleMismatchError(f"engine does not compute {sorted(missing)}")

    def probability_from(self, feats: dict[str, float]) -> tuple[float, np.ndarray]:
        x = row_to_vector(feats, self.features, self.spec.horizon_steps)
        raw = np.asarray(self.booster.predict(x, num_threads=1))  # one row: no thread pool
        return float(self.calibrator(raw)[0]), x

    def decide(self, event: Event, feats: dict[str, float], *, explain: bool = True) -> Decision:
        """Decision for `event` from features computed before it (no state is touched).
        `explain=False` skips the reasons (shadow scoring only needs the decision)."""
        if event.type not in MODELLED_TYPES:
            return Decision(
                txn_id=event.txn_id,
                model_version=self.version,
                p_fraud=0.0,
                decision="APPROVE",
                decision_reason="type_not_modelled",
                expected_cost=self._cost_dict(np.zeros(1), np.array([event.amount])),
            )
        return self._decide(event, feats, explain)

    def explain(
        self, x: np.ndarray, feats: dict[str, float], event: Event
    ) -> tuple[list[dict[str, Any]], bool]:
        """Top reasons (grouped TreeSHAP) and whether the top-3 is stable."""
        assert self.reason_codes is not None
        grouped = self.reason_codes.grouped(contributions(self.booster, x)[0])
        values: dict[str, Any] = {**feats, "amount": event.amount, "type": event.type}
        reasons = [r.__dict__ for r in self.reason_codes.reasons(grouped[0], values)]
        stable = bool(gap_after_top_k(grouped)[0] >= self.stability_gap)
        return reasons, stable

    def explain_features(self, feats: dict[str, float], event: Event) -> Explanation:
        """Attribution of a decision from its stored features (no state is read)."""
        if self.reason_codes is None:
            raise BundleMismatchError("bundle has no reason codes")
        x = row_to_vector(feats, self.features, self.spec.horizon_steps)
        shap, bias = contributions(self.booster, x)
        grouped = self.reason_codes.grouped(shap)
        reasons, stable = self.explain(x, feats, event)
        return Explanation(
            contributions={f: float(v) for f, v in zip(self.features, shap[0], strict=True)},
            bias=float(bias[0]),
            groups={c: float(v) for c, v in zip(self.reason_codes.codes, grouped[0], strict=True)},
            reasons=reasons,
            explanation_stable=stable,
        )

    def _decide(self, event: Event, feats: dict[str, float], explain: bool = True) -> Decision:
        p, x = self.probability_from(feats)
        action, reason, sets = self.policy.decide(p, event.amount, event.step)
        reasons: list[dict[str, Any]] = []
        stable: bool | None = None
        if explain and action != 0 and self.reason_codes is not None:  # anything not approved
            reasons, stable = self.explain(x, feats, event)
            if not reasons:  # nothing raised the risk: say why the policy acted
                text = POLICY_REASONS.get(reason, reason)
                reasons = [{"code": reason.upper(), "text": text, "impact": 0.0}]
        prediction_set = (
            None if sets is None else [c for c, inside in zip(CLASSES, sets, strict=True) if inside]
        )
        return Decision(
            txn_id=event.txn_id,
            model_version=self.version,
            p_fraud=p,
            decision=ACTIONS[action],
            decision_reason=reason,
            expected_cost=self._cost_dict(np.array([p]), np.array([event.amount])),
            prediction_set=prediction_set,
            reasons=reasons,
            explanation_stable=stable,
            features=feats,
        )

    def _cost_dict(self, p: np.ndarray, amount: np.ndarray) -> dict[str, float]:
        """Expected cost of every enabled action (disabled ones are left out, not inf)."""
        costs = expected_costs(p, amount, self.costs)[0]
        return {ACTIONS[a]: round(float(costs[a]), 4) for a in self.costs.enabled_actions()}


class Scorer(BundleModel):
    """Bundle + feature engine: score an event, then record it in state."""

    def __init__(self, bundle_dir: Path, store: StateStore | None = None) -> None:
        super().__init__(bundle_dir)
        self.engine = FeatureEngine(self.spec, store or InMemoryStore(self.spec.horizon_steps))
        self.check_features(self.engine.feature_names)

    def score(self, event: Event, *, apply: bool = True) -> Decision:
        """Decide on `event`, then (by default) record it in state. Score first (D4)."""
        started = time.perf_counter()
        feats = self.engine.features(event) if event.type in MODELLED_TYPES else {}
        decision = self.decide(event, feats)
        if apply:
            self.engine.apply(event)
        latency = (time.perf_counter() - started) * 1_000
        return Decision(**{**decision.__dict__, "latency_ms": round(latency, 3)})

    def probability(self, event: Event) -> float:
        """Calibrated P(fraud) for a modelled event, without touching state."""
        return self.probability_from(self.engine.features(event))[0]

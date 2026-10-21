"""Pure inference (task 2.1): bundle + feature engine -> decision.

Used by the API and by tests. The bundle is verified on load (every file's hash) and its
feature engine version must equal this code's, so a model can never be served with
features computed differently from how it was trained (D3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from fraudgraph.decision.costs import ACTIONS, CostParams, expected_costs
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.matrix import row_to_vector
from fraudgraph.features.spec import ENGINE_VERSION, Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore, StateStore
from fraudgraph.models.bundle import verify_bundle

MODELLED_TYPES = frozenset({"TRANSFER", "CASH_OUT"})


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
    reasons: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class Scorer:
    def __init__(self, bundle_dir: Path, store: StateStore | None = None) -> None:
        self.manifest = verify_bundle(bundle_dir)
        spec_data = json.loads((bundle_dir / "feature_spec.json").read_text(encoding="utf-8"))
        if spec_data["engine_version"] != ENGINE_VERSION:
            raise BundleMismatchError(
                f"bundle engine {spec_data['engine_version']} != code engine {ENGINE_VERSION}"
            )
        self.spec = FeatureSpec(
            windows=tuple(spec_data["windows"]), horizon_steps=spec_data["horizon_steps"]
        )
        self.version: str = self.manifest["version"]
        self.features: list[str] = self.manifest["features"]
        self.booster = lgb.Booster(model_file=str(bundle_dir / "model.txt"))
        self.calibrator = joblib.load(bundle_dir / "calibrator.joblib")
        policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
        self.costs = CostParams(**policy["costs"])
        self.engine = FeatureEngine(self.spec, store or InMemoryStore(self.spec.horizon_steps))

    def score(self, event: Event, *, apply: bool = True) -> Decision:
        """Decide on `event`, then (by default) record it in state. Score first (D4)."""
        started = time.perf_counter()
        if event.type in MODELLED_TYPES:
            decision = self._decide(event)
        else:
            decision = Decision(
                txn_id=event.txn_id,
                model_version=self.version,
                p_fraud=0.0,
                decision="APPROVE",
                decision_reason="type_not_modelled",
                expected_cost={
                    "APPROVE": 0.0,
                    "REVIEW": self.costs.c_review,
                    "BLOCK": self.costs.c_friction,
                },
            )
        if apply:
            self.engine.apply(event)
        latency = (time.perf_counter() - started) * 1_000
        return Decision(**{**decision.__dict__, "latency_ms": round(latency, 3)})

    def probability(self, event: Event) -> float:
        """Calibrated P(fraud) for a modelled event, without touching state."""
        x = row_to_vector(self.engine.features(event), self.features, self.spec.horizon_steps)
        raw = np.asarray(self.booster.predict(x))
        return float(self.calibrator(raw)[0])

    def _decide(self, event: Event) -> Decision:
        p = self.probability(event)
        costs = expected_costs(np.array([p]), np.array([event.amount]), self.costs)[0]
        action = int(np.argmin(costs))
        return Decision(
            txn_id=event.txn_id,
            model_version=self.version,
            p_fraud=p,
            decision=ACTIONS[action],
            decision_reason="expected_cost",
            expected_cost={a: round(float(c), 4) for a, c in zip(ACTIONS, costs, strict=True)},
        )

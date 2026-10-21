"""The decision-time feature engine (A5, D3).

One code path for offline backfill and online serving:

    feats = engine.features(event)   # read-only: uses state from strictly earlier steps
    engine.apply(event)              # then record the event

Families (prefix = family, used later to group SHAP reasons):
- `t_*` transaction: what the request itself says.
- `o_*` origin history: the sender's earlier outgoing transactions within the horizon.
  PaySim origins almost never repeat (docs/data_audit.md), so these are expected to be
  mostly empty; they are built because the engine and the drift scenarios need them.

Graph families (G1) are added in task 3.2.
"""

from __future__ import annotations

import math

from fraudgraph.features.spec import FRAUD_AMOUNT_CAP, STEPS_PER_DAY, Event, FeatureSpec
from fraudgraph.features.state import StateStore

NAN = float("nan")


def origin_key(account: str) -> str:
    return f"o:{account}"


class FeatureEngine:
    def __init__(self, spec: FeatureSpec, store: StateStore) -> None:
        self.spec = spec
        self.store = store
        self.feature_names: tuple[str, ...] = (
            *TRANSACTION_FEATURES,
            *(f"o_n_{_label(w)}" for w in spec.windows),
            *(f"o_log_sum_{_label(w)}" for w in spec.windows),
            *ORIGIN_SCALARS,
        )

    def features(self, ev: Event) -> dict[str, float]:
        """Features for `ev`. Pure: reads state, never changes it."""
        out = transaction_features(ev)
        out.update(self._origin_features(ev))
        return out

    def apply(self, ev: Event) -> None:
        """Record `ev` in state. Call after scoring it."""
        self.store.add(origin_key(ev.name_orig), ev.step, ev.amount)

    def _origin_features(self, ev: Event) -> dict[str, float]:
        history = self.store.history(origin_key(ev.name_orig), ev.step)
        out: dict[str, float] = {}
        for w in self.spec.windows:
            recent = [b for b in history if b[0] >= ev.step - w]
            out[f"o_n_{_label(w)}"] = float(sum(b[1] for b in recent))
            out[f"o_log_sum_{_label(w)}"] = math.log1p(sum(b[2] for b in recent))

        n = sum(b[1] for b in history)
        out["o_first_seen"] = 1.0 if n == 0 else 0.0
        out["o_steps_since_last"] = ev.step - history[-1][0] if history else NAN
        if n >= 2:
            mean = sum(b[2] for b in history) / n
            var = max(sum(b[3] for b in history) / n - mean * mean, 0.0)
            std = math.sqrt(var)
            out["o_amount_z"] = (ev.amount - mean) / std if std > 0 else 0.0
        else:
            out["o_amount_z"] = NAN
        return out


TRANSACTION_FEATURES = ("t_log_amount", "t_is_transfer", "t_hour", "t_amount_at_cap")
ORIGIN_SCALARS = ("o_first_seen", "o_steps_since_last", "o_amount_z")


def transaction_features(ev: Event) -> dict[str, float]:
    return {
        "t_log_amount": math.log1p(ev.amount),
        "t_is_transfer": 1.0 if ev.type == "TRANSFER" else 0.0,
        # Simulator-inflated (legit has a daily cycle, fraud does not): always ablate.
        "t_hour": float(ev.step % STEPS_PER_DAY),
        "t_amount_at_cap": 1.0 if ev.amount == FRAUD_AMOUNT_CAP else 0.0,
    }


def _label(steps: int) -> str:
    if steps % (7 * STEPS_PER_DAY) == 0:
        return f"{steps // (7 * STEPS_PER_DAY)}w"
    if steps % STEPS_PER_DAY == 0:
        return f"{steps // STEPS_PER_DAY}d"
    return f"{steps}h"

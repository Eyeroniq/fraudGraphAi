"""The decision-time feature engine (A5, D3, D4).

One code path for offline backfill and online serving:

    feats = engine.features(event)   # read-only: uses state from strictly earlier steps
    engine.apply(event)              # then record the event
    engine.apply_label(dest, is_fraud, known_at_step)   # when a label arrives (delayed)

Families (prefix = family, used to group SHAP reasons):
- `t_*`  transaction: what the request itself says.
- `o_*`  origin history: the sender's earlier outgoing transactions (expected near-empty
         on PaySim: origins almost never repeat, docs/data_audit.md).
- `g_*`  graph (G1): the receiving account's earlier activity (G1-a, the core family),
         confirmed fraud it received (G1-b, label-delayed) and the sender->receiver pair
         (G1-c).

Every feature uses only events at strictly earlier steps within the horizon, and labels
only after they are known (D4): a label known at step s is visible from step s + 1.
"""

from __future__ import annotations

import math

from fraudgraph.features.spec import FRAUD_AMOUNT_CAP, STEPS_PER_DAY, Event, FeatureSpec
from fraudgraph.features.state import Bucket, StateStore

NAN = float("nan")


def origin_key(account: str) -> str:
    return f"o:{account}"


def dest_key(account: str) -> str:
    return f"d:{account}"


def fraud_key(account: str) -> str:
    return f"f:{account}"


class FeatureEngine:
    def __init__(self, spec: FeatureSpec, store: StateStore) -> None:
        self.spec = spec
        self.store = store
        w = spec.windows
        self.feature_names: tuple[str, ...] = (
            *TRANSACTION_FEATURES,
            *(f"o_n_{_label(x)}" for x in w),
            *(f"o_log_sum_{_label(x)}" for x in w),
            *ORIGIN_SCALARS,
            *(f"g_dest_in_n_{_label(x)}" for x in w),
            *(f"g_dest_senders_{_label(x)}" for x in w if x > 1),
            *GRAPH_SCALARS,
        )

    def features(self, ev: Event) -> dict[str, float]:
        """Features for `ev`. Pure: reads state, never changes it."""
        out = transaction_features(ev)
        out.update(self._origin_features(ev))
        out.update(self._graph_features(ev))
        return out

    def apply(self, ev: Event) -> bool:
        """Record `ev` in state. Call after scoring it. Returns False if `ev.txn_id` was
        already applied (a retried request), so state never double-counts."""
        return self.store.apply(
            ev.txn_id,
            [
                (origin_key(ev.name_orig), ev.step, ev.amount, ev.name_dest),
                (dest_key(ev.name_dest), ev.step, ev.amount, ev.name_orig),
            ],
        )

    def apply_label(
        self, dest: str, is_fraud: bool, known_at_step: int, txn_id: str | None = None
    ) -> None:
        """A label for a payment into `dest` became known at `known_at_step` (D4).

        Only confirmed fraud is stored (it is what G1-b counts). The caller is responsible
        for the delay: in backfill, a transaction at step s is labelled at s + L. With a
        `txn_id`, the same label posted twice is counted once.
        """
        if is_fraud:
            marker = None if txn_id is None else f"label:{txn_id}"
            self.store.apply(marker, [(fraud_key(dest), known_at_step, 1.0, None)])

    # ------------------------------------------------------------------ families

    def _origin_features(self, ev: Event) -> dict[str, float]:
        history = self.store.history(origin_key(ev.name_orig), ev.step)
        out: dict[str, float] = {}
        for w in self.spec.windows:
            recent = _since(history, ev.step - w)
            out[f"o_n_{_label(w)}"] = float(sum(b.n for b in recent))
            out[f"o_log_sum_{_label(w)}"] = math.log1p(sum(b.total for b in recent))

        n = sum(b.n for b in history)
        out["o_first_seen"] = 1.0 if n == 0 else 0.0
        out["o_steps_since_last"] = float(ev.step - history[-1].step) if history else NAN
        if n >= 2:
            mean = sum(b.total for b in history) / n
            var = max(sum(b.total_sq for b in history) / n - mean * mean, 0.0)
            std = math.sqrt(var)
            out["o_amount_z"] = (ev.amount - mean) / std if std > 0 else 0.0
        else:
            out["o_amount_z"] = NAN
        return out

    def _graph_features(self, ev: Event) -> dict[str, float]:
        incoming = self.store.history(dest_key(ev.name_dest), ev.step)
        outgoing = self.store.history(origin_key(ev.name_dest), ev.step)
        out: dict[str, float] = {}

        # G1-a: the receiving account.
        for w in self.spec.windows:
            recent = _since(incoming, ev.step - w)
            out[f"g_dest_in_n_{_label(w)}"] = float(sum(b.n for b in recent))
            if w > 1:
                out[f"g_dest_senders_{_label(w)}"] = float(len(_members(recent)))
        n_in = sum(b.n for b in incoming)
        n_in_day = sum(b.n for b in _since(incoming, ev.step - STEPS_PER_DAY))
        out["g_dest_is_new"] = 1.0 if n_in == 0 else 0.0
        out["g_dest_log_in_sum"] = math.log1p(sum(b.total for b in incoming))
        out["g_dest_out_n"] = float(sum(b.n for b in outgoing))
        first = [h[0].step for h in (incoming, outgoing) if h]
        out["g_dest_age"] = float(ev.step - min(first)) if first else NAN
        out["g_dest_in_share_1d"] = n_in_day / n_in if n_in else NAN

        # G1-b: confirmed fraud this account received (labels already known).
        known = self.store.history(fraud_key(ev.name_dest), ev.step)
        out["g_dest_known_fraud_in"] = float(sum(b.n for b in known))

        # G1-c: has this sender paid this receiver before?
        sender_history = self.store.history(origin_key(ev.name_orig), ev.step)
        out["g_pair_n"] = float(sum((b.members or []).count(ev.name_dest) for b in sender_history))
        return out


TRANSACTION_FEATURES = ("t_log_amount", "t_is_transfer", "t_hour", "t_amount_at_cap")
ORIGIN_SCALARS = ("o_first_seen", "o_steps_since_last", "o_amount_z")
GRAPH_SCALARS = (
    "g_dest_is_new",
    "g_dest_log_in_sum",
    "g_dest_out_n",
    "g_dest_age",
    "g_dest_in_share_1d",
    "g_dest_known_fraud_in",
    "g_pair_n",
)
FAMILIES = {"t": "transaction", "o": "origin history", "g": "graph"}


def transaction_features(ev: Event) -> dict[str, float]:
    return {
        "t_log_amount": math.log1p(ev.amount),
        "t_is_transfer": 1.0 if ev.type == "TRANSFER" else 0.0,
        # Simulator-inflated (legit has a daily cycle, fraud does not): always ablate.
        "t_hour": float(ev.step % STEPS_PER_DAY),
        "t_amount_at_cap": 1.0 if ev.amount == FRAUD_AMOUNT_CAP else 0.0,
    }


def _since(history: list[Bucket], first_step: int) -> list[Bucket]:
    return [b for b in history if b.step >= first_step]


def _members(buckets: list[Bucket]) -> set[str]:
    out: set[str] = set()
    for b in buckets:
        if b.members:
            out.update(b.members)
    return out


def _label(steps: int) -> str:
    if steps % (7 * STEPS_PER_DAY) == 0:
        return f"{steps // (7 * STEPS_PER_DAY)}w"
    if steps % STEPS_PER_DAY == 0:
        return f"{steps // STEPS_PER_DAY}d"
    return f"{steps}h"

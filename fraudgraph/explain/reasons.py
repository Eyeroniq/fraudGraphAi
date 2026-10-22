"""SHAP-based reason codes (task 5.1).

Attributions come from LightGBM's own TreeSHAP (`predict(..., pred_contrib=True)`), on the
margin (log-odds): the same values as `shap.TreeExplainer`, with no `shap` import, so the
serving image stays slim. Isotonic calibration is monotone, so the ranking is unchanged.

Feature attributions are summed per reason group (configs/reason_codes.yaml) and the
top risk-increasing groups become plain-language reasons with the payment's own values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import yaml

TOP_K = 3


@dataclass(frozen=True)
class Variant:
    text: str
    when: tuple[str, float] | None  # (feature, value) for "feature == value"


@dataclass(frozen=True)
class Group:
    code: str
    label: str
    features: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Reason:
    code: str
    text: str
    impact: float  # summed SHAP (log-odds) of the group; > 0 raises the fraud score


class ReasonCodes:
    def __init__(self, groups: list[Group], model_features: list[str]) -> None:
        self.model_features = model_features
        index = {name: i for i, name in enumerate(model_features)}
        self.groups = [g for g in groups if any(f in index for f in g.features)]
        self.membership = np.zeros((len(model_features), len(self.groups)))
        for j, g in enumerate(self.groups):
            for f in g.features:
                if f in index:
                    self.membership[index[f], j] = 1.0
        ungrouped = [f for f in model_features if self.membership[index[f]].sum() == 0]
        if ungrouped:
            raise ValueError(f"features without a reason group: {ungrouped}")
        self.codes = [g.code for g in self.groups]

    @classmethod
    def load(cls, path: Path, model_features: list[str]) -> ReasonCodes:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        groups = [
            Group(
                code=g["code"],
                label=g["label"],
                features=tuple(g["features"]),
                variants=tuple(
                    Variant(v["text"], _parse_when(v.get("when"))) for v in g["variants"]
                ),
            )
            for g in data["groups"]
        ]
        return cls(groups, model_features)

    def grouped(self, contributions: np.ndarray) -> np.ndarray:
        """Per-feature SHAP [n, F] -> per-group SHAP [n, G]."""
        return np.asarray(contributions @ self.membership)

    def reasons(
        self, group_shap: np.ndarray, values: dict[str, Any], k: int = TOP_K
    ) -> list[Reason]:
        """Top-k risk-increasing groups for one payment, as text."""
        order = np.argsort(-group_shap, kind="stable")
        out = []
        for j in order[:k]:
            if group_shap[j] <= 0:
                break
            out.append(
                Reason(self.codes[j], self._text(self.groups[j], values), float(group_shap[j]))
            )
        return out

    def _text(self, group: Group, values: dict[str, Any]) -> str:
        for variant in group.variants:
            if variant.when is None or _matches(variant.when, values):
                try:
                    return variant.text.format(**values)
                except (KeyError, ValueError):
                    return group.label
        return group.label


def contributions(booster: lgb.Booster, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-feature SHAP [n, F], bias [n]) on the margin, via LightGBM's TreeSHAP."""
    threads: dict[str, Any] = {"num_threads": 1} if len(x) == 1 else {}  # serving: one row
    raw = np.asarray(
        booster.predict(
            x, pred_contrib=True, num_iteration=booster.best_iteration or None, **threads
        )
    )
    return raw[:, :-1], raw[:, -1]


def gap_after_top_k(group_shap: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """|SHAP| gap between the k-th and (k+1)-th group: small gaps mean fragile top-k."""
    s = -np.sort(-np.abs(group_shap), axis=1)
    if s.shape[1] <= k:
        return np.full(len(s), np.inf)
    return np.asarray(s[:, k - 1] - s[:, k])


def _parse_when(text: str | None) -> tuple[str, float] | None:
    if text is None:
        return None
    left, sep, right = text.partition("==")
    if not sep:
        raise ValueError(f"only 'feature == value' conditions are supported: {text!r}")
    return left.strip(), float(right)


def _matches(cond: tuple[str, float], values: dict[str, Any]) -> bool:
    value = values.get(cond[0])
    return (
        value is not None
        and not (isinstance(value, float) and math.isnan(value))
        and float(value) == cond[1]
    )

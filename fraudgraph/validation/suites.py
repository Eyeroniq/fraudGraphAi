"""Validation gates (task 9.1): `fgx validate`.

Three suites in the shape of deepchecks' tabular suites, each check with an explicit
condition and a severity:

- **data_integrity** - the raw events and the engine features are well formed;
- **train_test_validation** - the splits are time-ordered and disjoint, the test set has no
  categories the model never saw, labels and features did not shift beyond limits;
- **model_evaluation** - the bundle's model clears its AP floor, beats prevalence, is
  calibrated better than a constant, and is not much weaker on any segment.

A FAIL condition stops CI (`fgx validate` exits 1); a WARN is reported only. Feature drift
is a warning: PaySim's destination-volume features drift with calendar time by design
(docs/drift_study.md), which is a fact about the data, not a defect.

deepchecks itself is not used: its latest release (0.19) fails at run time against the
project's scikit-learn 1.9 (removed scorer names) and needs a setuptools pin, so these
checks are written directly with the same suite structure and HTML/JSON reports.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from fraudgraph.drift.psi_ks import bin_shares, matrix_stats, psi
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.models.compare import Ranked
from fraudgraph.serving.scorer import MODELLED_TYPES, BundleModel

Severity = Literal["fail", "warn"]
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
CORE = ("txn_id", "step", "type", "amount", "name_orig", "name_dest", "is_fraud")


@dataclass
class Check:
    name: str
    condition: str
    passed: bool
    value: Any
    severity: Severity = "fail"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "pass" if self.passed else self.severity


@dataclass(frozen=True)
class Limits:
    """Conditions' thresholds (configs/validation.yaml)."""

    ap_floor: float = 0.3
    min_lift: float = 10.0  # AP over prevalence
    max_label_shift: float = 4.0  # test / train prevalence ratio, either way
    feature_psi_warn: float = 0.25
    segment_ap_share: float = 0.5  # segment AP >= share x overall AP
    segment_min_fraud: int = 20
    max_nan_share: float = 0.999
    max_prevalence: float = 0.05


# ------------------------------------------------------------------ suites


def data_integrity(
    events: pl.DataFrame, table: pl.DataFrame, features: list[str], lim: Limits
) -> list[Check]:
    n = events.height
    dupes = n - events["txn_id"].n_unique()
    nulls = {c: int(events[c].null_count()) for c in CORE}
    types = sorted(set(events["type"].unique().to_list()))
    fraud_types = sorted(set(events.filter(pl.col("is_fraud"))["type"].unique().to_list()))
    x = frame_to_matrix(table, features, 168)
    raw = table.select(features).to_numpy().astype(np.float64)
    nan_share = {f: float(np.mean(np.isnan(raw[:, i]))) for i, f in enumerate(features)}
    constant = [f for i, f in enumerate(features) if np.unique(x[:, i]).size < 2]
    prevalence = float(events["is_fraud"].cast(pl.Float64).to_numpy().mean())
    return [
        Check("unique_txn_id", "no duplicate transaction IDs", dupes == 0, dupes),
        Check("required_not_null", "no nulls in core columns", sum(nulls.values()) == 0, nulls),
        Check(
            "amount_non_negative",
            "every amount >= 0",
            int((events["amount"] < 0).sum()) == 0,
            int((events["amount"] < 0).sum()),
        ),
        Check(
            "step_in_range",
            "1 <= step <= 744",
            bool(events["step"].min() >= 1 and events["step"].max() <= 744),  # type: ignore[operator]
            [events["step"].min(), events["step"].max()],
        ),
        Check(
            "known_types", f"types are a subset of {list(TYPES)}", set(types) <= set(TYPES), types
        ),
        Check(
            "fraud_only_on_modelled_types",
            "fraud only on TRANSFER / CASH_OUT (what the model scores)",
            set(fraud_types) <= MODELLED_TYPES,
            fraud_types,
        ),
        Check(
            "features_finite",
            "no +-inf in engine features",
            bool(np.isfinite(x).all()),
            int((~np.isfinite(x)).sum()),
        ),
        Check(
            "features_not_empty",
            f"no feature is NaN on more than {lim.max_nan_share:.1%} of rows",
            max(nan_share.values(), default=0.0) <= lim.max_nan_share,
            {f: round(v, 4) for f, v in sorted(nan_share.items(), key=lambda kv: -kv[1])[:5]},
            severity="warn",
        ),
        Check(
            "features_not_constant",
            "every feature takes at least two values",
            not constant,
            constant,
            severity="warn",
        ),
        Check(
            "prevalence_plausible",
            f"0 < fraud rate < {lim.max_prevalence:.0%}",
            0 < prevalence < lim.max_prevalence,
            round(prevalence, 6),
        ),
    ]


def train_test_validation(
    train: pl.DataFrame, test: pl.DataFrame, features: list[str], label_delay: int, lim: Limits
) -> list[Check]:
    gap = int(test["step"].min()) - int(train["step"].max())  # type: ignore[arg-type]
    overlap = len(set(train["txn_id"].to_list()) & set(test["txn_id"].to_list()))
    new_types = (
        sorted(set(test["type"].unique().to_list()) - set(train["type"].unique().to_list()))
        if "type" in train.columns
        else []
    )
    p_train = float(train["is_fraud"].cast(pl.Float64).to_numpy().mean())
    p_test = float(test["is_fraud"].cast(pl.Float64).to_numpy().mean())
    ratio = (max(p_test, 1e-9) / max(p_train, 1e-9)) if p_train > 0 else float("inf")
    x_train = frame_to_matrix(train, features, 168)
    x_test = frame_to_matrix(test, features, 168)
    stats = matrix_stats(x_train, features)
    drift = {
        f: round(
            psi(
                np.r_[stats[f]["shares"], stats[f]["nan_share"]],
                bin_shares(x_test[:, i], np.asarray(stats[f]["edges"])),
            ),
            4,
        )
        for i, f in enumerate(features)
    }
    drifted = {f: v for f, v in drift.items() if v > lim.feature_psi_warn}
    return [
        Check(
            "time_ordered",
            f"test starts more than the label delay ({label_delay}) after train ends",
            gap > label_delay,
            gap,
        ),
        Check(
            "no_shared_transactions", "no transaction in both train and test", overlap == 0, overlap
        ),
        Check(
            "no_new_categories",
            "test has no payment type unseen in train",
            not new_types,
            new_types,
        ),
        Check(
            "label_drift",
            f"test fraud rate within x{lim.max_label_shift:g} of train",
            1 / lim.max_label_shift <= ratio <= lim.max_label_shift,
            {"train": round(p_train, 6), "test": round(p_test, 6), "ratio": round(ratio, 3)},
        ),
        Check(
            "feature_drift",
            f"feature PSI train -> test <= {lim.feature_psi_warn}",
            not drifted,
            dict(sorted(drifted.items(), key=lambda kv: -kv[1])),
            severity="warn",
            details={"psi": drift},
        ),
    ]


def model_evaluation(
    model: BundleModel,
    test: pl.DataFrame,
    amount: np.ndarray,
    lim: Limits,
    champion: str | None = None,
) -> list[Check]:
    modelled = (
        test["type"].is_in(list(MODELLED_TYPES)).to_numpy()
        if "type" in test.columns
        else np.ones(test.height, bool)
    )
    t = test.filter(pl.Series(modelled))
    amount = amount[modelled]
    y = t["is_fraud"].cast(pl.Int8).to_numpy()
    x = frame_to_matrix(t, model.features, model.spec.horizon_steps)
    # Ranking is judged on the raw score: isotonic calibration is a step function, and its
    # ties lower AP (0.467 -> 0.415 on PaySim test). Calibration is judged on p itself.
    raw = np.asarray(model.booster.predict(x))
    p = np.asarray(model.calibrator(raw))
    prevalence = float(y.mean())
    ap = Ranked(y, raw).metrics()["ap"] if y.sum() else float("nan")
    brier = float(np.mean((p - y) ** 2))
    constant = prevalence * (1 - prevalence)
    segments: dict[str, dict[str, float]] = {}
    weak: dict[str, float] = {}
    quartiles = np.quantile(amount, [0.25, 0.5, 0.75])
    bucket = np.searchsorted(quartiles, amount, side="right")
    groups = {f"type={v}": t["type"].to_numpy() == v for v in sorted(set(t["type"].to_list()))}
    groups |= {f"amount=Q{q + 1}": bucket == q for q in range(4)}
    for name, mask in groups.items():
        n_fraud = int(y[mask].sum())
        if n_fraud < lim.segment_min_fraud:
            continue
        seg_ap = Ranked(y[mask], raw[mask]).metrics()["ap"]
        segments[name] = {"ap": round(seg_ap, 4), "fraud": n_fraud, "rows": int(mask.sum())}
        if seg_ap < lim.segment_ap_share * ap:
            weak[name] = round(seg_ap, 4)
    checks = [
        Check("ap_floor", f"test AP >= {lim.ap_floor}", ap >= lim.ap_floor, round(ap, 4)),
        Check(
            "beats_prevalence",
            f"test AP >= {lim.min_lift:g} x fraud rate",
            ap >= lim.min_lift * prevalence,
            {
                "ap": round(ap, 4),
                "prevalence": round(prevalence, 6),
                "lift": round(ap / max(prevalence, 1e-12), 1),
            },
        ),
        Check(
            "calibration_skill",
            "Brier score below a constant predictor's",
            brier < constant,
            {
                "brier": round(brier, 7),
                "constant": round(constant, 7),
                "skill": round(1 - brier / constant, 4),
            },
        ),
        Check(
            "weak_segments",
            f"no segment (type, amount quartile; >= {lim.segment_min_fraud} fraud) "
            f"below {lim.segment_ap_share:.0%} of overall AP",
            not weak,
            weak,
            severity="warn",
            details={"segments": segments},
        ),
    ]
    if champion is not None:
        checks.append(
            Check(
                "serves_champion",
                f"bundle model is the configured champion ({champion})",
                model.manifest["model"] == champion,
                model.manifest["model"],
            )
        )
    return checks


# ------------------------------------------------------------------ report


def summarise(suites: dict[str, list[Check]]) -> dict[str, Any]:
    counts = {
        s: sum(c.status == s for checks in suites.values() for c in checks)
        for s in ("pass", "warn", "fail")
    }
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "passed": counts["fail"] == 0,
        "counts": counts,
        "suites": {
            name: [asdict(c) | {"status": c.status} for c in checks]
            for name, checks in suites.items()
        },
    }


def _fmt(value: Any) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return html.escape(text if len(text) < 400 else text[:400] + " ...")


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;background:#161719;color:#f4f4f5;margin:2rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:.5rem .75rem;border-bottom:1px solid #333;text-align:left;vertical-align:top}}
th{{color:#6ee7b7;text-transform:uppercase;letter-spacing:.2em;font-size:.75rem}}
code{{color:#a1a1aa}}
</style>
<h1>{title}</h1><p>{summary}</p><table>{rows}</table></html>
"""


def write_reports(summary: dict[str, Any], out: Path, title: str) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "validation.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    color = {"pass": "#34d399", "warn": "#fbbf24", "fail": "#fb7185"}
    rows = []
    for suite, checks in summary["suites"].items():
        rows.append(f'<tr><th colspan="4">{html.escape(suite)}</th></tr>')
        for c in checks:
            rows.append(
                f'<tr><td><b style="color:{color[c["status"]]}">{c["status"].upper()}</b></td>'
                f"<td>{html.escape(c['name'])}</td><td>{html.escape(c['condition'])}</td>"
                f"<td><code>{_fmt(c['value'])}</code></td></tr>"
            )
    counts = summary["counts"]
    page = PAGE.format(
        title=html.escape(title),
        summary=(
            f"{counts['pass']} passed · {counts['warn']} warnings · {counts['fail']} failed · "
            f"{html.escape(summary['generated_at'])}"
        ),
        rows="".join(rows),
    )
    html_path = out / "validation.html"
    html_path.write_text(page, encoding="utf-8")
    return json_path, html_path

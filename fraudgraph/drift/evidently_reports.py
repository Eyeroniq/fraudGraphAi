"""Evidently reports (task 6.5): data-drift report per window, saved as HTML + JSON, with
Evidently's per-column PSI pulled out to cross-check our own monitor. The two PSIs differ
mainly in binning: ours uses the bundle's fixed train_fit quantile bins (and a NaN bin);
Evidently bins each comparison afresh from the reference data."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_COLUMN = re.compile(r"ValueDrift\(column=([^,]+),method=psi")


def drift_report(
    reference: dict[str, np.ndarray], current: dict[str, np.ndarray], out_stem: Path
) -> dict[str, float]:
    """Evidently DataDriftPreset(psi): save `<stem>.html` + `.json`, return PSI per column."""
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref = pd.DataFrame({k: np.nan_to_num(v, nan=-1.0) for k, v in reference.items()})
    cur = pd.DataFrame({k: np.nan_to_num(v, nan=-1.0) for k, v in current.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset(method="psi")]).run(current_data=cur, reference_data=ref)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_stem.with_suffix(".html")))
    snapshot.save_json(str(out_stem.with_suffix(".json")))
    out: dict[str, float] = {}
    for metric in snapshot.dict()["metrics"]:
        match = _COLUMN.match(metric.get("metric_name", ""))
        if match:
            out[match.group(1)] = float(metric["value"])
    return out


def compare(ours: dict[str, float], evidently: dict[str, float]) -> dict[str, Any]:
    names = sorted(set(ours) & set(evidently))
    a = np.array([ours[n] for n in names])
    b = np.array([evidently[n] for n in names])
    from scipy.stats import spearmanr

    return {
        "features": len(names),
        "spearman": float(spearmanr(a, b).statistic) if len(names) > 2 else float("nan"),
        "both_above_0.2": int(np.sum((a > 0.2) & (b > 0.2))),
        "only_ours_above_0.2": int(np.sum((a > 0.2) & (b <= 0.2))),
        "only_evidently_above_0.2": int(np.sum((a <= 0.2) & (b > 0.2))),
        "per_feature": {n: {"ours": ours[n], "evidently": evidently[n]} for n in names},
    }

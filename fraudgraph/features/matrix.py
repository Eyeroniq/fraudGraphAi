"""Turn feature values into the model's input row, identically offline and online.

"No history" features are NaN in the engine output. Before a model sees them they get
fixed fills, defined here once so training (`models.baselines.matrix`) and serving
(`serving.scorer`) cannot drift apart.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def fill_values(horizon_steps: int) -> dict[str, float]:
    """Feature -> value used when the feature is NaN."""
    return {
        "o_steps_since_last": float(horizon_steps + 1),  # "longer ago than we remember"
        "o_amount_z": 0.0,  # no history: not unusual
        "g_dest_age": float(horizon_steps + 1),  # never seen: older than we remember
        "g_dest_in_share_1d": 0.0,  # nothing received at all
    }


def frame_to_matrix(table: pl.DataFrame, features: list[str], horizon_steps: int) -> np.ndarray:
    fills = fill_values(horizon_steps)
    filled = table.select(features).with_columns(
        [pl.col(c).fill_nan(fills[c]) for c in features if c in fills]
    )
    return filled.to_numpy().astype(np.float64)


def row_to_vector(values: dict[str, float], features: list[str], horizon_steps: int) -> np.ndarray:
    fills = fill_values(horizon_steps)
    row = [
        fills[name] if name in fills and math.isnan(values[name]) else values[name]
        for name in features
    ]
    return np.asarray([row], dtype=np.float64)

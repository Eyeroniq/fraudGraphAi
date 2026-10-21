"""Destination novelty, computed in bulk for the B-rule baseline (1.3).

`dest_is_new` = the destination received no transaction (of any type) at an earlier
step within the horizon. Same-step transactions do not count (D4). This is the batch
form of the G1-a feature the engine gains in 3.2; a parity test will tie them together.
"""

from __future__ import annotations

import polars as pl


def destination_novelty(frame: pl.DataFrame, horizon_steps: int) -> pl.DataFrame:
    """`txn_id, dest_is_new` for every row of a prepared PaySim frame."""
    previous_step = (
        frame.select("name_dest", "step")
        .unique()
        .sort("name_dest", "step")
        .with_columns(prev_step=pl.col("step").shift(1).over("name_dest"))
    )
    return (
        frame.select("txn_id", "name_dest", "step")
        .join(previous_step, on=["name_dest", "step"], how="left")
        .select(
            "txn_id",
            dest_is_new=(
                pl.col("prev_step").is_null()
                | (pl.col("step") - pl.col("prev_step") > horizon_steps)
            ),
        )
    )

"""Drift scenario injection. For now only D3 (mule rings), used by experiment E4; the
full D0-D5 library with sudden/gradual modes comes with task 6.2."""

from __future__ import annotations

import numpy as np
import polars as pl


def inject_mule_rings(
    events: pl.DataFrame,
    *,
    first_step: int,
    last_step: int,
    rings: int = 20,
    destinations_per_ring: int = 3,
    senders_per_destination: int = 15,
    burst_steps: int = 6,
    seed: int = 0,
) -> tuple[pl.DataFrame, list[str]]:
    """Add rings of brand-new destination accounts, each receiving TRANSFERs from many new
    senders within a short burst. Ring payments do not drain the sender (so only graph
    features can see them) and are labelled fraud. Returns (events, injected txn_ids)."""
    if rings > 999 or destinations_per_ring > 99 or senders_per_destination > 999 or seed > 99:
        raise ValueError(
            "ring ids are fixed-width: rings <= 999, destinations <= 99, senders <= 999"
        )
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(rings):
        start = int(rng.integers(first_step, last_step - burst_steps + 2))
        for d in range(destinations_per_ring):
            dest = f"C8{seed:02d}{r:03d}{d:02d}000"
            for s in range(senders_per_destination):
                balance = float(np.round(rng.lognormal(np.log(60_000), 0.8), 2))
                amount = float(np.round(balance * rng.uniform(0.2, 0.8), 2))
                rows.append(
                    {
                        "txn_id": f"R{r:03d}{d:02d}{s:03d}",
                        "step": int(rng.integers(start, start + burst_steps)),
                        "type": "TRANSFER",
                        "amount": amount,
                        "name_orig": f"C9{seed:02d}{r:03d}{d:02d}{s:03d}",
                        "oldbalance_orig": balance,
                        "newbalance_orig": round(balance - amount, 2),
                        "name_dest": dest,
                        "oldbalance_dest": 0.0,
                        "newbalance_dest": amount,
                        "is_fraud": True,
                        "is_flagged_fraud": False,
                    }
                )
    injected = pl.DataFrame(rows).cast({c: events.schema[c] for c in rows[0]})
    merged = pl.concat([events, injected.select(events.columns)]).sort("step", maintain_order=True)
    return merged, injected["txn_id"].to_list()

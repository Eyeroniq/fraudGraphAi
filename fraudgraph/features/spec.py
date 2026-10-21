"""What the feature engine computes, and the event it computes it from."""

from __future__ import annotations

from dataclasses import dataclass

# Bump when feature definitions change: a model bundle refuses to load with an engine
# whose version differs from the one it was trained with (2.1).
ENGINE_VERSION = "1.2.0"

STEPS_PER_DAY = 24
FRAUD_AMOUNT_CAP = 10_000_000.0


@dataclass(frozen=True)
class Event:
    """A transaction as known at authorisation time. No balance columns (D2)."""

    txn_id: str
    step: int
    type: str
    amount: float
    name_orig: str
    name_dest: str


@dataclass(frozen=True)
class FeatureSpec:
    windows: tuple[int, ...] = (1, 24, 168)  # steps: 1 h, 24 h, 7 d
    horizon_steps: int = 168  # state TTL; history beyond this is forgotten
    version: str = ENGINE_VERSION

    def __post_init__(self) -> None:
        if not self.windows or max(self.windows) > self.horizon_steps:
            raise ValueError("every window must fit inside horizon_steps")

"""Performance drift from delayed labels (task 6.4).

When a fraud label arrives, we learn whether the model *caught* it at the time
(score at or above the review threshold) or missed it. ADWIN and Page-Hinkley watch that
miss stream: a sustained rise means recall has dropped, i.e. the model is being beaten.
Only fraud labels matter here (legit labels say nothing about recall), and labels that
never arrive (missing chargebacks) are invisible to this monitor by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceMonitor:
    adwin_delta: float = 0.002
    ph_threshold: float = 20.0
    ph_delta: float = 0.005
    min_labels: int = 30
    alarms: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from river import drift

        self._adwin: Any = drift.ADWIN(  # type: ignore[no-untyped-call]
            delta=self.adwin_delta
        )
        self._ph: Any = drift.PageHinkley(
            threshold=self.ph_threshold, delta=self.ph_delta, mode="up"
        )
        self._seen = 0

    def update(self, missed: bool, step: int) -> list[str]:
        """Feed one arriving fraud label; returns the detectors that fired."""
        self._seen += 1
        fired = []
        value = 1.0 if missed else 0.0
        self._adwin.update(value)
        self._ph.update(value)
        if self._seen >= self.min_labels:
            if self._adwin.drift_detected:
                fired.append("adwin")
            if self._ph.drift_detected:
                fired.append("page_hinkley")
        for name in fired:
            self.alarms.append(
                {"detector": name, "step": float(step), "labels_seen": float(self._seen)}
            )
        return fired

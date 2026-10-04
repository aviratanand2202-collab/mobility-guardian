"""
Caregiver dismissal tracking + anti-alarm-fatigue governance (spec §2.6).

Two separate mechanisms, both scoped per H3 grid cell (resolution 9):

1. Dismissal decay clock: a caregiver marking "User is Safe" suppresses
   future alerts for that grid cell, but the suppression:
     - decays back to zero over a FIXED 30-day window
     - the clock starts on the FIRST dismissal and does NOT reset on
       subsequent dismissals within that window (prevents indefinite
       suppression via repeated clicking)
     - caps at 3 dismissals per cell per 30-day window; the 4th forces a
       formal recalibration flow instead of a quick-dismiss

2. MAD baseline quarantine: this module only tracks dismissal metadata for
   the alert/UI layer. The actual exclusion of dismissed/high-entropy
   trips from the rolling baseline distribution is computed on the ML
   side (see ml/features/baseline.py) - this module is the source of
   which trips were dismissed, which the ML pipeline reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

DECAY_WINDOW_DAYS = 30
MAX_DISMISSALS_PER_WINDOW = 3


@dataclass
class CellDismissalState:
    grid_cell_id: str
    decay_clock_start: datetime | None = None
    dismissal_timestamps: list[datetime] = field(default_factory=list)

    def _prune_expired(self, now: datetime) -> None:
        if self.decay_clock_start is None:
            return
        if now - self.decay_clock_start > timedelta(days=DECAY_WINDOW_DAYS):
            # Window fully expired - reset clean.
            self.decay_clock_start = None
            self.dismissal_timestamps = []

    def register_dismissal(self, now: datetime) -> tuple[bool, int]:
        """
        Record a caregiver "User is Safe" dismissal for this cell.

        Returns (allowed, count_in_window):
            allowed=False means the UI should refuse quick-dismiss and
            force the formal safe-zone recalibration flow instead.
        """
        self._prune_expired(now)

        if self.decay_clock_start is None:
            # First dismissal in a fresh window - fixes the clock.
            self.decay_clock_start = now

        if len(self.dismissal_timestamps) >= MAX_DISMISSALS_PER_WINDOW:
            return False, len(self.dismissal_timestamps)

        self.dismissal_timestamps.append(now)
        return True, len(self.dismissal_timestamps)

    def sensitivity_multiplier(self) -> float:
        """
        Returns the anomaly-score multiplier for this cell, floored at
        0.70 (max 30% suppression) per spec §2.6 - caregiver dismissals
        can never fully silence a cell; extreme kinematic entropy still
        punches through.
        """
        if not self.dismissal_timestamps:
            return 1.0
        # Linear scaling: each dismissal contributes up to -10%, floor -30%.
        suppression = min(0.30, 0.10 * len(self.dismissal_timestamps))
        return 1.0 - suppression


# Per-cell state registry. Replace with Redis/DB before production.
_cell_states: dict[str, CellDismissalState] = {}


def get_or_create_cell_state(grid_cell_id: str) -> CellDismissalState:
    if grid_cell_id not in _cell_states:
        _cell_states[grid_cell_id] = CellDismissalState(grid_cell_id=grid_cell_id)
    return _cell_states[grid_cell_id]

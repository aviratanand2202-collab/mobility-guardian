"""
Pedestrian Dead Reckoning (PDR) hysteresis state machine.

Implements the Schmitt-trigger thresholds from spec §2.5 to prevent state
churn when net displacement hovers near a boundary. Activated when
signal_status.state == DEGRADED_SIGNAL (HDOP > 20m), using fused
accelerometer + gyroscope net displacement in place of GPS.

Thresholds (fixed, do not tune without updating docs/limitations.md and
re-deriving the detection-delay bound in §4):

    Tier 0 (INDOOR_PACING)         -> escalate to Tier 1 at  > 27m
    Tier 1 (ZONE_TRANSITION)       -> de-escalate to Tier 0 at < 23m
                                       escalate to Tier 2 at   > 52m
    Tier 2 (UNTRACKED_DISPLACEMENT)-> de-escalate to Tier 1 at < 48m

Max blind-spot duration is 15 minutes (enforced by the caller - see
degraded_signal.py - not by this state machine).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PDRTier(str, Enum):
    INDOOR_PACING = "INDOOR_PACING"
    ZONE_TRANSITION = "ZONE_TRANSITION"
    UNTRACKED_DISPLACEMENT = "UNTRACKED_DISPLACEMENT"


# Hysteresis bounds (meters), per spec §2.5
ESCALATE_TIER0_TO_TIER1_M = 27.0
DEESCALATE_TIER1_TO_TIER0_M = 23.0
ESCALATE_TIER1_TO_TIER2_M = 52.0
DEESCALATE_TIER2_TO_TIER1_M = 48.0


@dataclass
class PDRState:
    tier: PDRTier = PDRTier.INDOOR_PACING

    def update(self, net_displacement_m: float) -> PDRTier:
        """
        Apply the latest net displacement reading and return the (possibly
        unchanged) tier. Pure function of current tier + new reading - call
        this once per telemetry update while signal is degraded.
        """
        if self.tier == PDRTier.INDOOR_PACING:
            if net_displacement_m > ESCALATE_TIER0_TO_TIER1_M:
                self.tier = PDRTier.ZONE_TRANSITION

        elif self.tier == PDRTier.ZONE_TRANSITION:
            if net_displacement_m > ESCALATE_TIER1_TO_TIER2_M:
                self.tier = PDRTier.UNTRACKED_DISPLACEMENT
            elif net_displacement_m < DEESCALATE_TIER1_TO_TIER0_M:
                self.tier = PDRTier.INDOOR_PACING
            # else: stays in ZONE_TRANSITION (inside the hysteresis band)

        elif self.tier == PDRTier.UNTRACKED_DISPLACEMENT:
            if net_displacement_m < DEESCALATE_TIER2_TO_TIER1_M:
                self.tier = PDRTier.ZONE_TRANSITION
            # else: stays in UNTRACKED_DISPLACEMENT

        return self.tier

    def should_alert_caregiver(self) -> bool:
        """Tier 2 wakes GPS for a forced lock and alerts the caregiver."""
        return self.tier == PDRTier.UNTRACKED_DISPLACEMENT


# Per-user state registry. Replace with a real store (Redis / DB) before
# production - this in-memory dict is only for local dev / Sprint 1-2.
_user_states: dict[str, PDRState] = {}


def get_or_create_state(user_id: str) -> PDRState:
    if user_id not in _user_states:
        _user_states[user_id] = PDRState()
    return _user_states[user_id]

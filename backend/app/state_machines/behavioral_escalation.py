"""
Behavioral trajectory risk escalation state machine.

Bridges the ML behavioral classifier (PACING, RANDOM_DRIFT) to operational
risk evaluation with safety invariants:
1. Multi-window persistence requirement (>= 2 distinct windows) prevents single-window false positives.
2. Distinct window tracking by window_id prevents sliding-window duplicate counting.
3. Explicit counter resets on changed behavior (NORMAL, LAPPING), dwell / insufficient evidence,
   stale telemetry gaps (> 180s), degraded GPS (PDR fallback), or monitoring disconnection.
4. Respects safe-area context: pacing inside designated safe area is suppressed from emergency escalation.
5. Additive clamping only: elevates baseline operational score to SUSPICIOUS (>= 65.0),
   never overwrites higher operational scores (e.g. CRITICAL high speed) and never copies raw ML scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
from typing import Any, Dict, Optional, Tuple

from app.models import RiskTier, SignalState, TelemetryPayload

logger = logging.getLogger(__name__)

# Constants
REQUIRED_CONSECUTIVE_WINDOWS: int = 2
MIN_BEHAVIORAL_CONFIDENCE: float = 0.75
MAX_STALE_GAP_SECONDS: float = 180.0
MAX_CONSECUTIVE_ML_ERRORS: int = 3
ESCALATED_MIN_RISK_SCORE: float = 65.0
SUPPORTED_ANOMALOUS_PATTERNS = {"PACING", "RANDOM_DRIFT"}

# In-memory user state registry protected by thread lock
_user_states: Dict[str, BehavioralEscalationState] = {}
_state_lock = threading.Lock()


@dataclass
class BehavioralEscalationState:
    user_id: str
    consecutive_anomalous_windows: int = 0
    last_evaluated_window_id: Optional[str] = None
    last_observation_time: Optional[datetime] = None
    last_confirmed_anomalous_time: Optional[datetime] = None
    consecutive_ml_errors: int = 0
    last_pattern: Optional[str] = None
    last_confidence: Optional[float] = None
    is_escalated: bool = False
    escalation_reason: Optional[str] = None
    is_hr_corroborated: bool = False
    hr_corroboration_detail: Optional[str] = None
    target_tier: Optional[RiskTier] = None
    target_score: Optional[float] = None

    def reset(self, reason: str = "") -> None:
        """Reset consecutive counter and escalation state."""
        if self.consecutive_anomalous_windows > 0 or self.is_escalated:
            logger.info(f"[BehavioralEscalation] Resetting state for user '{self.user_id}': {reason}")
        self.consecutive_anomalous_windows = 0
        self.last_evaluated_window_id = None
        self.last_confirmed_anomalous_time = None
        self.consecutive_ml_errors = 0
        self.is_escalated = False
        self.last_pattern = None
        self.last_confidence = None
        self.escalation_reason = None
        self.is_hr_corroborated = False
        self.hr_corroboration_detail = None
        self.target_tier = None
        self.target_score = None


def get_or_create_behavioral_state(user_id: str) -> BehavioralEscalationState:
    """Retrieve or initialize the BehavioralEscalationState for a specific user."""
    with _state_lock:
        if user_id not in _user_states:
            _user_states[user_id] = BehavioralEscalationState(user_id=user_id)
        return _user_states[user_id]


def reset_behavioral_state(user_id: Optional[str] = None) -> None:
    """Reset behavioral escalation state for a specific user, or all users if user_id is None."""
    with _state_lock:
        if user_id is not None:
            if user_id in _user_states:
                _user_states[user_id].reset("Explicit reset requested")
        else:
            for state in _user_states.values():
                state.reset("Global reset requested")
            _user_states.clear()


def evaluate_behavioral_escalation(
    payload: TelemetryPayload,
    ml_eval: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Evaluate whether incoming telemetry warrants operational risk escalation.

    Returns:
        Tuple of (is_escalated: bool, explanation: Optional[str]).
    """
    user_id = payload.user_id
    state = get_or_create_behavioral_state(user_id)

    # 1. Degraded GPS check: if signal is degraded, bypass ML behavioral escalation
    if payload.signal_status.state == SignalState.DEGRADED_SIGNAL:
        state.reset("GPS signal degraded; PDR displacement fallback active")
        return False, None

    # Parse and normalize timestamp
    ts = payload.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 2. Stale telemetry / temporal gap check & out-of-order protection (Scenario J)
    if state.last_observation_time is not None:
        gap_sec = (ts - state.last_observation_time).total_seconds()
        if gap_sec < 0:
            logger.info(f"[BehavioralEscalation] Out-of-order telemetry for user '{user_id}' (delta: {gap_sec:.1f}s); ignoring without counter change")
            return state.is_escalated, state.escalation_reason
        elif gap_sec == 0:
            logger.debug(f"[BehavioralEscalation] Duplicate telemetry timestamp for user '{user_id}'; ignoring duplicate")
            return state.is_escalated, state.escalation_reason
        elif gap_sec > MAX_STALE_GAP_SECONDS:
            state.reset(f"Temporal telemetry gap of {gap_sec:.1f}s exceeded threshold ({MAX_STALE_GAP_SECONDS}s)")
        state.last_observation_time = ts
    else:
        state.last_observation_time = ts

    # 3. Check for stale evidence expiration on active escalation
    # Stale evidence cannot silently sustain an escalation indefinitely.
    if state.is_escalated and state.last_confirmed_anomalous_time is not None:
        evidence_gap_sec = (ts - state.last_confirmed_anomalous_time).total_seconds()
        if evidence_gap_sec > MAX_STALE_GAP_SECONDS:
            state.reset(
                f"Behavioral escalation expired: no confirming anomalous evidence for {evidence_gap_sec:.1f}s "
                f"(threshold {MAX_STALE_GAP_SECONDS}s)"
            )
            return False, None

    # 4. Missing or errored ML evaluation
    if not ml_eval or not isinstance(ml_eval, dict):
        if state.is_escalated:
            state.consecutive_ml_errors += 1
            if state.consecutive_ml_errors >= MAX_CONSECUTIVE_ML_ERRORS:
                state.reset(
                    f"Behavioral escalation reset: ML inference missing for {state.consecutive_ml_errors} consecutive readings"
                )
                return False, None
        return state.is_escalated, state.escalation_reason

    engine_status = ml_eval.get("engine_status")
    evidence_status = ml_eval.get("evidence_status")

    if engine_status != "ACTIVE" or evidence_status == "ERROR":
        # Engine error or timeout - do not escalate; track consecutive errors if already escalated
        if state.is_escalated:
            state.consecutive_ml_errors += 1
            if state.consecutive_ml_errors >= MAX_CONSECUTIVE_ML_ERRORS:
                state.reset(
                    f"Behavioral escalation reset: ML inference errored ({engine_status}/{evidence_status}) "
                    f"for {state.consecutive_ml_errors} consecutive readings"
                )
                return False, None
        return state.is_escalated, state.escalation_reason

    # ML evaluation is active and healthy: reset consecutive ML error counter
    state.consecutive_ml_errors = 0

    # 5. Safe-Area Policy: movement inside designated safe area is suppressed from emergency escalation
    safe_area_state = str(ml_eval.get("safe_area_state") or "").upper()
    if safe_area_state == "INSIDE_SAFE_AREA":
        state.reset("Movement is contained within designated safe area")
        return False, None

    # Extract heart rate evidence for physiological corroboration
    hr_ev = ml_eval.get("heart_rate_evidence")
    is_hr_corroborating = bool(hr_ev and hr_ev.get("is_corroborating_risk"))
    hr_bpm = hr_ev.get("current_hr_bpm") if hr_ev else None
    hr_z = hr_ev.get("z_score") if hr_ev else None

    # 6. Check if a distinct window completed
    window_id = ml_eval.get("window_id")
    if evidence_status != "WINDOW_EVALUATED" or not window_id:
        # Intermediate point while window is accumulating; maintain current status without incrementing
        return state.is_escalated, state.escalation_reason

    if window_id == state.last_evaluated_window_id:
        # Same sliding window re-evaluated; do not count as new evidence
        return state.is_escalated, state.escalation_reason

    # Genuinely distinct window completed
    state.last_evaluated_window_id = window_id

    pattern = str(ml_eval.get("behavioral_indicator") or "NORMAL").upper()
    conf_raw = ml_eval.get("behavioral_confidence")
    try:
        conf = float(conf_raw) if conf_raw is not None else 0.0
    except (ValueError, TypeError):
        conf = 0.0

    # 7. Evaluate behavioral indicator, confidence, and physiological corroboration
    if pattern in SUPPORTED_ANOMALOUS_PATTERNS and conf >= MIN_BEHAVIORAL_CONFIDENCE:
        state.consecutive_anomalous_windows += 1
        state.last_pattern = pattern
        state.last_confidence = conf
        state.last_confirmed_anomalous_time = ts

        pattern_desc = (
            "pacing (repetitive linear directional reversals)"
            if pattern == "PACING"
            else "random drift (elevated directional entropy and meandering)"
        )
        duration_mins = (state.consecutive_anomalous_windows * 120) // 60

        if is_hr_corroborating:
            state.is_hr_corroborated = True
            state.hr_corroboration_detail = (
                f"Concomitant physiological arousal: elevated heart rate ({hr_bpm} bpm, z={hr_z:+.2f}) "
                f"observed during low-movement context relative to personal baseline."
            )

            # Physiological Corroboration Escalation Rules:
            # Multi-modal evidence: 1 window of confirmed anomalous kinematics + physiological arousal
            # satisfies immediate escalation to SUSPICIOUS.
            # Sustained multi-modal evidence (>= 2 windows) escalates to CRITICAL.
            if state.consecutive_anomalous_windows >= REQUIRED_CONSECUTIVE_WINDOWS:
                state.is_escalated = True
                state.target_tier = RiskTier.CRITICAL
                state.target_score = 85.0
                state.escalation_reason = (
                    f"Sustained {pattern_desc} across {state.consecutive_anomalous_windows} windows "
                    f"strongly corroborated by persistent physiological arousal ({hr_bpm} bpm, z={hr_z:+.2f})."
                )
            else:
                state.is_escalated = True
                state.target_tier = RiskTier.SUSPICIOUS
                state.target_score = 70.0
                state.escalation_reason = (
                    f"Repetitive {pattern_desc} detected across observation window corroborated by "
                    f"significant physiological arousal ({hr_bpm} bpm, z={hr_z:+.2f} relative to personal baseline)."
                )

            logger.info(
                f"[BehavioralEscalation] User '{user_id}' escalated to {state.target_tier.value}: "
                f"{state.escalation_reason}"
            )
            return True, state.escalation_reason

        else:
            state.is_hr_corroborated = False
            state.hr_corroboration_detail = None

            # Pure Kinematics Escalation Rules:
            # Pacing alone without physiological corroboration requires >= 2 distinct windows (persistence gate).
            if state.consecutive_anomalous_windows >= REQUIRED_CONSECUTIVE_WINDOWS:
                state.is_escalated = True
                state.target_tier = RiskTier.SUSPICIOUS
                state.target_score = ESCALATED_MIN_RISK_SCORE
                state.escalation_reason = (
                    f"Sustained {pattern_desc} detected across {state.consecutive_anomalous_windows} "
                    f"distinct observation windows ({duration_mins} minutes) with {int(conf * 100)}% confidence."
                )
                logger.info(
                    f"[BehavioralEscalation] User '{user_id}' escalated to SUSPICIOUS: "
                    f"{state.escalation_reason}"
                )
                return True, state.escalation_reason
            else:
                # First anomalous window without physiological corroboration: persistence requirement not yet satisfied
                state.is_escalated = False
                state.target_tier = None
                state.target_score = None
                state.escalation_reason = None
                return False, None

    else:
        # Normal behavior, lapping, dwell / insufficient evidence, or weak confidence: reset counter
        state.reset(f"Observed pattern '{pattern}' (confidence: {conf:.2f}) does not meet anomalous persistence criteria")
        return False, None

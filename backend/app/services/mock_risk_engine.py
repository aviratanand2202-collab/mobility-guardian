"""
MOCK IMPLEMENTATION — replace compute_risk() internals with a call
to the real trained model / ML service when available. Do not change
the function signature without updating all callers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Optional

from pydantic import BaseModel

from app.models import (
    PDRTierState,
    PollingInstruction,
    PollingMode,
    RiskScoreOutput,
    RiskTier,
    SignalState,
    TelemetryPayload,
)
from app.state_machines.battery_pulse import compute_polling_instruction
from app.state_machines.dismissal_quarantine import CellDismissalState


# ---------- Nested schema structures strictly matching /shared/schema.json ----------

class KinematicFeatures(BaseModel):
    tortuosity_index: float
    entropy_value: float
    distance_from_anchor_m: float
    step_speed_variance: float


class TriggerState(BaseModel):
    window_seconds: int = 120
    consecutive_anomalous_windows: int
    required_k: int
    percentile_bound_crossed: Optional[str] = None


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    human_readable: str


class Explainability(BaseModel):
    top_features: list[TopFeature]


class MockRiskScoreOutput(RiskScoreOutput):
    """
    Subclasses RiskScoreOutput to attach kinematic_features, trigger_state,
    and explainability strictly matching /shared/schema.json without altering
    the frozen app/models.py definitions.
    """
    pdr_tier: Optional[str] = None
    kinematic_features: Optional[KinematicFeatures] = None
    trigger_state: Optional[TriggerState] = None
    explainability: Optional[Explainability] = None


# ---------- Risk tier to polling tier mapping per spec §2.6 ----------

_RISK_TO_POLLING_TIER = {
    RiskTier.QUIESCENT: 0,
    RiskTier.NORMAL_TRANSIT: 1,
    RiskTier.SUSPICIOUS: 2,
    RiskTier.CRITICAL: 3,
}


def _normalize_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC for cross-environment determinism."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_risk(
    telemetry: TelemetryPayload,
    pdr_tier: PDRTierState | None = None,
    cell_dismissal_state: CellDismissalState | None = None,
) -> RiskScoreOutput:
    """
    Deterministic mock risk engine.

    Derives risk from PDR state during degraded signal, or from a deterministic
    speed + pseudo-random tortuosity heuristic during valid signal. Applies
    sensitivity multiplier from the cell dismissal quarantine state machine
    to suppress risk score in previously-dismissed cells.
    """
    ts_utc = _normalize_utc(telemetry.timestamp)

    # 2-minute non-overlapping window epoch calculated strictly in UTC
    epoch_seconds = int(ts_utc.timestamp())
    window_epoch = epoch_seconds // 120

    # Deterministic pseudo-random seed from (user_id, window_epoch)
    seed_input = f"{telemetry.user_id}:{window_epoch}".encode("utf-8")
    seed_hash = hashlib.sha256(seed_input).digest()
    pseudo_rand = int.from_bytes(seed_hash[:4], "big") / 0xFFFFFFFF  # in [0.0, 1.0)

    pdr_val = None
    if pdr_tier is not None:
        pdr_val = pdr_tier.value if hasattr(pdr_tier, "value") else str(pdr_tier)

    # Branch A: DEGRADED_SIGNAL or active PDR tier
    # ML risk scoring freezes during degraded signal to prevent spurious triggers;
    # risk_tier is derived directly from the PDR state machine.
    if telemetry.signal_status.state == SignalState.DEGRADED_SIGNAL or pdr_val is not None:
        if pdr_val == "INDOOR_PACING":
            raw_score = 15.0
        elif pdr_val == "ZONE_TRANSITION":
            raw_score = 50.0
        elif pdr_val == "UNTRACKED_DISPLACEMENT":
            raw_score = 75.0
        else:
            raw_score = 15.0
        pdr_tier_str = pdr_val

    # Branch B: VALID signal
    # Heuristic based on speed + synthetic tortuosity from the deterministic seed
    else:
        pdr_tier_str = None
        speed = max(0.0, float(telemetry.sensor_metrics.speed_mps))
        # Speed contribution up to 60 points (5.0 m/s reaches 60.0)
        speed_comp = min(60.0, speed * 12.0)
        # Tortuosity contribution up to 40 points
        tortuosity_comp = pseudo_rand * 40.0
        raw_score = min(100.0, max(0.0, speed_comp + tortuosity_comp))

    # Apply sensitivity suppression multiplier (floored at 0.70 inside CellDismissalState)
    multiplier = (
        cell_dismissal_state.sensitivity_multiplier()
        if cell_dismissal_state is not None
        else 1.0
    )
    risk_score = round(min(100.0, max(0.0, raw_score * multiplier)), 1)

    # Map to risk_tier per specification boundaries:
    # 0-34: QUIESCENT, 35-64: NORMAL_TRANSIT, 65-84: SUSPICIOUS, 85-100: CRITICAL
    if risk_score <= 34.0:
        risk_tier = RiskTier.QUIESCENT
    elif risk_score <= 64.0:
        risk_tier = RiskTier.NORMAL_TRANSIT
    elif risk_score <= 84.0:
        risk_tier = RiskTier.SUSPICIOUS
    else:
        risk_tier = RiskTier.CRITICAL

    # Map risk_tier -> polling_tier (0-3)
    polling_tier = _RISK_TO_POLLING_TIER[risk_tier]

    # Polling instruction calculation via existing battery_pulse state machine
    battery_pct = telemetry.sensor_metrics.battery_pct
    polling_inst_dataclass = compute_polling_instruction(risk_tier.value, battery_pct)
    battery_override_active = polling_inst_dataclass.mode != "CONTINUOUS"
    polling_instruction = PollingInstruction(
        mode=PollingMode(polling_inst_dataclass.mode),
        burst_seconds=polling_inst_dataclass.burst_seconds,
        burst_hz=polling_inst_dataclass.burst_hz,
        sleep_seconds=polling_inst_dataclass.sleep_seconds,
    )

    # Kinematic features mock matching /shared/schema.json
    kinematic_features = KinematicFeatures(
        tortuosity_index=round(1.0 + pseudo_rand * 2.0, 2),
        entropy_value=round(0.1 + (risk_score / 100.0) * 0.9, 2),
        distance_from_anchor_m=150.0,
        step_speed_variance=round(0.05 + (risk_score / 250.0), 3),
    )

    # Trigger state matching /shared/schema.json
    trigger_state = TriggerState(
        window_seconds=120,
        consecutive_anomalous_windows=1 if risk_tier in (RiskTier.SUSPICIOUS, RiskTier.CRITICAL) else 0,
        required_k=2 if risk_tier == RiskTier.CRITICAL else 3,
        percentile_bound_crossed="P99" if risk_tier == RiskTier.CRITICAL else ("P95" if risk_tier == RiskTier.SUSPICIOUS else None),
    )

    # Explainability TreeSHAP mock matching /shared/schema.json
    top_features = [
        TopFeature(
            feature="path_tortuosity",
            shap_value=round(0.15 + (risk_score / 200.0), 2),
            human_readable="Erratic turning pattern detected" if risk_score >= 35.0 else "Regular straight-line heading",
        ),
        TopFeature(
            feature="step_speed_variance",
            shap_value=round(0.08 + (risk_score / 300.0), 2),
            human_readable="Velocity entropy elevated" if risk_score >= 65.0 else "Stable locomotive pace",
        ),
        TopFeature(
            feature="distance_from_anchor",
            shap_value=0.12,
            human_readable="Displacement beyond known safe anchor cluster",
        ),
    ]
    explainability = Explainability(top_features=top_features)

    # Predicted lead time (seconds) only when SUSPICIOUS or CRITICAL
    if risk_tier in (RiskTier.SUSPICIOUS, RiskTier.CRITICAL):
        predicted_lead_time_sec = round(max(240.0, 900.0 - (risk_score - 65.0) * 15.0), 1)
    else:
        predicted_lead_time_sec = None

    return MockRiskScoreOutput(
        user_id=telemetry.user_id,
        timestamp=ts_utc,
        risk_tier=risk_tier,
        risk_score=risk_score,
        polling_tier=polling_tier,
        predicted_lead_time_sec=predicted_lead_time_sec,
        battery_override_active=battery_override_active,
        polling_instruction=polling_instruction,
        location=telemetry.location,
        pdr_tier=pdr_tier_str,
        kinematic_features=kinematic_features,
        trigger_state=trigger_state,
        explainability=explainability,
    )

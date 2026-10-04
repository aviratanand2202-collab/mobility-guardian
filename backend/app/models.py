"""
Pydantic models mirroring /shared/schema.json.

These must stay in sync with the JSON Schema contract. If you add/change a
field here, update schema.json too (and tell Person A).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Telemetry ----------

class SignalState(str, Enum):
    VALID = "VALID"
    DEGRADED_SIGNAL = "DEGRADED_SIGNAL"


class ActivityType(str, Enum):
    STATIONARY = "STATIONARY"
    WALKING = "WALKING"
    IN_VEHICLE = "IN_VEHICLE"
    UNKNOWN = "UNKNOWN"


class PDRTierState(str, Enum):
    INDOOR_PACING = "INDOOR_PACING"
    ZONE_TRANSITION = "ZONE_TRANSITION"
    UNTRACKED_DISPLACEMENT = "UNTRACKED_DISPLACEMENT"


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    altitude_m: Optional[float] = None


class SensorMetrics(BaseModel):
    horizontal_accuracy_m: float
    speed_mps: float = Field(..., ge=0)
    heading_deg: float = Field(..., ge=0, le=360)
    activity_type: ActivityType = ActivityType.UNKNOWN
    battery_pct: int = Field(..., ge=0, le=100)


class IMUMetrics(BaseModel):
    step_count_since_last_gps: int = Field(..., ge=0)
    net_displacement_m: float
    pdr_tier_state: PDRTierState


class SignalStatus(BaseModel):
    state: SignalState
    degraded_since: Optional[datetime] = None


class TelemetryPayload(BaseModel):
    user_id: str
    timestamp: datetime
    location: Location
    sensor_metrics: SensorMetrics
    imu_metrics: Optional[IMUMetrics] = None
    signal_status: SignalStatus


# ---------- Risk output (consumed from ML side) ----------

class RiskTier(str, Enum):
    QUIESCENT = "QUIESCENT"
    NORMAL_TRANSIT = "NORMAL_TRANSIT"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


class RiskScoreOutput(BaseModel):
    user_id: str
    timestamp: datetime
    risk_tier: RiskTier
    risk_score: float = Field(..., ge=0, le=100)
    polling_tier: int = Field(..., ge=0, le=3)
    predicted_lead_time_sec: Optional[float] = None
    battery_override_active: bool = False
    # kinematic_features, trigger_state, explainability intentionally loose
    # here (dict passthrough) since Person A owns their internal shape -
    # backend only needs to read risk_tier / risk_score / polling_tier to
    # drive its own state machines.


# ---------- Alerts ----------

class AlertStatus(str, Enum):
    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED_SAFE = "DISMISSED_SAFE"
    ESCALATED = "ESCALATED"
    AUTO_RESOLVED = "AUTO_RESOLVED"


class Dismissal(BaseModel):
    dismissed_at: datetime
    decay_clock_start: datetime
    dismissal_count_in_window: int = Field(..., le=3)


class AlertRecord(BaseModel):
    alert_id: str
    user_id: str
    grid_cell_id: str
    timestamp: datetime
    alert_tier: int = Field(..., ge=2, le=3)
    status: AlertStatus
    dismissal: Optional[Dismissal] = None


# ---------- Consent ----------

class GuardianConsent(BaseModel):
    granted: bool
    granted_at: datetime
    guardian_id: str


class EndUserConsent(BaseModel):
    granted: bool
    granted_at: datetime


class ConsentRecord(BaseModel):
    user_id: str
    guardian_consent: GuardianConsent
    end_user_consent_applicable: bool
    end_user_consent: Optional[EndUserConsent] = None
    data_retention_ack: bool = False

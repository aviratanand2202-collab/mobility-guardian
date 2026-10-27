"""
Pydantic models mirroring /shared/schema.json.

These must stay in sync with the JSON Schema contract. If you add/change a
field here, update schema.json too (and tell Person A).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

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


class WearState(str, Enum):
    ON_WRIST = "ON_WRIST"
    OFF_WRIST = "OFF_WRIST"
    UNKNOWN = "UNKNOWN"


class WearableMetrics(BaseModel):
    heart_rate_bpm: int = Field(..., ge=30, le=250, description="Heart rate reading in beats per minute (30-250)")
    timestamp: datetime
    wear_state: WearState = WearState.UNKNOWN
    sensor_status: Optional[str] = None


class TelemetryPayload(BaseModel):
    user_id: str
    timestamp: datetime
    location: Location
    sensor_metrics: SensorMetrics
    imu_metrics: Optional[IMUMetrics] = None
    signal_status: SignalStatus
    wearable_metrics: Optional[WearableMetrics] = None


# ---------- Risk output (consumed from ML side) ----------

class RiskTier(str, Enum):
    QUIESCENT = "QUIESCENT"
    NORMAL_TRANSIT = "NORMAL_TRANSIT"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


class PollingMode(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    PULSE = "PULSE"
    LAST_GASP = "LAST_GASP"


class PollingInstruction(BaseModel):
    mode: PollingMode
    burst_seconds: Optional[int] = None
    burst_hz: Optional[int] = None
    sleep_seconds: Optional[int] = None


class RiskScoreOutput(BaseModel):
    user_id: str
    timestamp: datetime
    risk_tier: RiskTier
    risk_score: float = Field(..., ge=0, le=100)
    polling_tier: int = Field(..., ge=0, le=3)
    predicted_lead_time_sec: Optional[float] = None
    battery_override_active: bool = False
    polling_instruction: Optional[PollingInstruction] = None
    location: Optional[Location] = None
    ml_evaluation: Optional[dict[str, Any]] = None
    kinematic_features: Optional[dict[str, Any]] = None
    trigger_state: Optional[dict[str, Any]] = None
    explainability: Optional[dict[str, Any]] = None
    wearable_metrics: Optional[WearableMetrics] = None
    sensor_metrics: Optional[SensorMetrics] = None


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
    zone_id: Optional[str] = None


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


# ---------- Trusted Companion Mode ----------

class CompanionSessionStatus(str, Enum):
    PENDING_INVITE = "PENDING_INVITE"
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class AccompanimentStatus(str, Enum):
    WITH_COMPANION = "With Companion"
    SEPARATION_PENDING = "Separation Pending"
    SEPARATED = "Separated"
    REJOINING = "Rejoining"
    LOCATION_UNAVAILABLE = "Location Unavailable"
    ACCOMPANIMENT_UNVERIFIED = "Accompaniment Unverified"


class CompanionSessionCreate(BaseModel):
    patient_id: str
    caregiver_id: str = "caregiver_primary"
    validity_minutes: int = Field(default=60, ge=5, le=1440)
    companion_id: Optional[str] = None
    companion_alias: Optional[str] = None
    relationship: Optional[str] = "family"


class CompanionConsentRequest(BaseModel):
    granted: bool
    companion_alias: Optional[str] = "Trusted Companion"


class CompanionLocationUpdate(BaseModel):
    companion_token: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    accuracy_m: Optional[float] = None
    timestamp: Optional[datetime] = None


class CompanionSessionSummary(BaseModel):
    session_id: str
    patient_id: str
    companion_user_id: Optional[str] = None
    relationship: Optional[str] = "trusted_companion"
    status: CompanionSessionStatus
    invite_token: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    consent_granted: bool
    consent_granted_at: Optional[datetime] = None
    companion_alias: str = "Trusted Companion"
    last_companion_lat: Optional[float] = None
    last_companion_lng: Optional[float] = None
    last_companion_accuracy_m: Optional[float] = None
    last_companion_timestamp: Optional[datetime] = None
    accompaniment_status: Optional[AccompanimentStatus] = None
    separation_distance_m: Optional[float] = None
    is_location_stale: bool = True
    safe_zone_departure_accompanied: Optional[bool] = False
    safe_zone: Optional[dict[str, Any]] = None

# ---------- Geofence Boundary Transitions ----------

class BoundaryState(str, Enum):
    INSIDE = "INSIDE"
    EXIT_PENDING = "EXIT_PENDING"
    BREACHED = "BREACHED"
    REENTRY_PENDING = "REENTRY_PENDING"
    RESOLVED = "RESOLVED"
    OUTSIDE = "OUTSIDE"


class GeofenceEvaluationStatus(str, Enum):
    NORMAL = "NORMAL"
    BREACH_ACTIVE = "BREACH_ACTIVE"
    NO_ZONES_CONFIGURED = "NO_ZONES_CONFIGURED"
    INSUFFICIENT_GPS = "INSUFFICIENT_GPS"


# ---------- Safe Zones ----------

class SafeZoneGeometryType(str, Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"


class SafeZoneCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable zone label e.g. Home, Company")
    description: Optional[str] = Field(None, max_length=500)
    geometry_type: SafeZoneGeometryType = SafeZoneGeometryType.CIRCLE
    center_lat: Optional[float] = Field(None, ge=-90, le=90)
    center_lng: Optional[float] = Field(None, ge=-180, le=180)
    radius_m: Optional[float] = Field(None, gt=0, le=50000)
    polygon_vertices: Optional[list[list[float]]] = Field(
        None,
        description="List of [latitude, longitude] pairs defining polygon vertices. Minimum 3 points."
    )
    is_enabled: bool = True


class SafeZoneUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    geometry_type: Optional[SafeZoneGeometryType] = None
    center_lat: Optional[float] = Field(None, ge=-90, le=90)
    center_lng: Optional[float] = Field(None, ge=-180, le=180)
    radius_m: Optional[float] = Field(None, gt=0, le=50000)
    polygon_vertices: Optional[list[list[float]]] = None
    is_enabled: Optional[bool] = None


class SafeZoneResponse(BaseModel):
    zone_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    geometry_type: SafeZoneGeometryType
    center_lat: Optional[float] = None
    center_lng: Optional[float] = None
    radius_m: Optional[float] = None
    polygon_vertices: Optional[list[list[float]]] = None
    is_enabled: bool
    created_at: datetime
    updated_at: datetime

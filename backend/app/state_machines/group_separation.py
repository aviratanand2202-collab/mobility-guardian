"""Group & Companion Separation State Machine (Phase 4).

Manages the lifecycle of group accompaniment, debounced separation,
physiological corroboration (wearable heart rate), and caregiver alert deduplication.

Key Safety Invariants:
1. Authorization: Only operates on explicitly consented patients and active companion sessions.
2. GPS Quality & Debouncing: Requires fresh GPS (<=60s) with accuracy <= 25m.
   Requires configurable consecutive readings (default: 2) to confirm separation or reunification.
3. Physiological Corroboration: When separated, evaluates the separated patient's heart rate
   against empirical personal baseline (z-score >= +2.0 or resting BPM >= 100).
   Escalates Tier 2 (SUSPICIOUS) -> Tier 3 (CRITICAL) if physiological arousal is present.
4. Non-Suppression Rule: Normal, resting, or missing heart rate NEVER suppresses
   the separation alert; it remains active at Tier 2.
5. Deduplication & Auto-Resolution: Maintains a single persistent AlertRecord while separated;
   auto-resolves cleanly upon confirmed group reunification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import select, desc

from app.db import (
    async_session,
    init_db,
    AlertRecordRow,
    CompanionSessionRow,
    GroupSeparationStateRow,
    RiskScoreHistoryRow,
    WearableReadingRow,
)
from app.models import AlertRecord, AlertStatus, AccompanimentStatus
from app.routers.consent import is_consented
from app.routers.risk_history import _extract_location
from app.services.geospatial import location_to_h3_cell
from app.websocket.risk_stream import push_alert_update

logger = logging.getLogger(__name__)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two coordinates."""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 1)


@dataclass(frozen=True)
class GroupSeparationConfig:
    """Configurable thresholds for group separation and physiological escalation."""

    separation_distance_m: float = 50.0
    """Distance in meters beyond which group members are considered separating."""

    persistence_readings: int = 2
    """Consecutive readings beyond distance threshold to confirm separation."""

    reunite_readings: int = 2
    """Consecutive readings within distance threshold to confirm group reunification."""

    max_gps_accuracy_m: float = 25.0
    """Accuracy threshold; readings with accuracy worse than this hold state."""

    max_reading_age_sec: float = 60.0
    """Maximum allowable age of companion or wearable telemetry before flagged stale."""

    corroborating_z_score_threshold: float = 2.0
    """Heart rate z-score threshold relative to empirical baseline indicating physiological arousal."""

    tachycardia_bpm_threshold: int = 100
    """Absolute BPM threshold indicating significant resting elevation."""


DEFAULT_SEPARATION_CONFIG = GroupSeparationConfig()


@dataclass
class GroupSeparationResult:
    """Outcome of evaluating patient and companion locations and biometrics."""

    status: str
    patient_id: str
    companion_id: Optional[str]
    distance_m: Optional[float]
    action: str  # "NONE", "ALERT_TRIGGERED", "ALERT_UPDATED", "ALERT_RESOLVED"
    alert_id: Optional[str]
    alert_tier: Optional[int]
    heart_rate_evaluated: bool
    heart_rate_bpm: Optional[int]
    heart_rate_provenance: Optional[str]
    heart_rate_z_score: Optional[float]
    is_physiological_arousal: bool
    consecutive_separated: int
    consecutive_united: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def evaluate_group_separation(
    patient_id: str,
    patient_lat: float,
    patient_lng: float,
    patient_accuracy_m: float,
    patient_timestamp: datetime,
    patient_is_degraded: bool = False,
    config: GroupSeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> GroupSeparationResult:
    """
    Evaluate group separation and wearable physiological corroboration.
    Manages persistent alert lifecycle for companion separation.
    """
    if patient_timestamp.tzinfo is None:
        patient_timestamp = patient_timestamp.replace(tzinfo=timezone.utc)

    # 1. Enforce patient authorization / consent
    if not await is_consented(patient_id):
        return GroupSeparationResult(
            status="CONSENT_NOT_GRANTED",
            patient_id=patient_id,
            companion_id=None,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message=f"Consent not granted for patient '{patient_id}'. Companion monitoring disabled.",
        )

    # 2. Fetch active companion session
    async with async_session() as session:
        comp_res = await session.execute(
            select(CompanionSessionRow).where(
                CompanionSessionRow.patient_id == patient_id,
                CompanionSessionRow.status == "ACTIVE",
            ).order_by(desc(CompanionSessionRow.created_at)).limit(1)
        )
        comp_row = comp_res.scalar_one_or_none()

    if comp_row is None:
        return GroupSeparationResult(
            status="NO_ACTIVE_COMPANION_SESSION",
            patient_id=patient_id,
            companion_id=None,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message=f"No active companion session found for patient '{patient_id}'.",
        )

    companion_id = comp_row.companion_user_id or comp_row.companion_alias

    # 3. GPS Evidence Quality Check for Patient
    if patient_is_degraded or patient_accuracy_m > config.max_gps_accuracy_m:
        return GroupSeparationResult(
            status="INSUFFICIENT_GPS",
            patient_id=patient_id,
            companion_id=companion_id,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message=(
                f"Patient GPS accuracy ({patient_accuracy_m:.1f}m) exceeds threshold ({config.max_gps_accuracy_m}m) "
                "or signal is degraded. Separation state held to prevent false alarms."
            ),
        )

    # 4. Sync and verify Companion location
    comp_lat = comp_row.last_companion_lat
    comp_lng = comp_row.last_companion_lng
    comp_ts = comp_row.last_companion_timestamp
    if comp_ts is not None and comp_ts.tzinfo is None:
        comp_ts = comp_ts.replace(tzinfo=timezone.utc)
    comp_acc = comp_row.last_companion_accuracy_m or 5.0

    if comp_row.companion_user_id:
        async with async_session() as session:
            comp_hist_res = await session.execute(
                select(RiskScoreHistoryRow)
                .where(RiskScoreHistoryRow.user_id == comp_row.companion_user_id)
                .order_by(desc(RiskScoreHistoryRow.timestamp))
                .limit(1)
            )
            comp_hist = comp_hist_res.scalar_one_or_none()
            if comp_hist:
                c_loc = _extract_location(comp_hist.raw_output_json)
                if c_loc and c_loc.lat is not None and c_loc.lng is not None:
                    hist_ts = comp_hist.timestamp if comp_hist.timestamp.tzinfo is not None else comp_hist.timestamp.replace(tzinfo=timezone.utc)
                    if comp_ts is None or hist_ts >= comp_ts:
                        comp_lat = c_loc.lat
                        comp_lng = c_loc.lng
                        comp_ts = hist_ts

    now = datetime.now(timezone.utc)
    if comp_lat is None or comp_lng is None or comp_ts is None:
        return GroupSeparationResult(
            status="ACCOMPANIMENT_UNVERIFIED",
            patient_id=patient_id,
            companion_id=companion_id,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message="Companion location is missing or not yet reported.",
        )

    if comp_ts.tzinfo is None:
        comp_ts = comp_ts.replace(tzinfo=timezone.utc)

    comp_age_sec = (now - comp_ts).total_seconds()
    if comp_age_sec > config.max_reading_age_sec:
        return GroupSeparationResult(
            status="ACCOMPANIMENT_UNVERIFIED",
            patient_id=patient_id,
            companion_id=companion_id,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message=f"Companion location is stale ({comp_age_sec:.1f}s > {config.max_reading_age_sec}s). Separation state held.",
        )

    if comp_acc > config.max_gps_accuracy_m:
        return GroupSeparationResult(
            status="INSUFFICIENT_GPS",
            patient_id=patient_id,
            companion_id=companion_id,
            distance_m=None,
            action="NONE",
            alert_id=None,
            alert_tier=None,
            heart_rate_evaluated=False,
            heart_rate_bpm=None,
            heart_rate_provenance=None,
            heart_rate_z_score=None,
            is_physiological_arousal=False,
            consecutive_separated=0,
            consecutive_united=0,
            message=f"Companion GPS accuracy ({comp_acc:.1f}m) exceeds threshold ({config.max_gps_accuracy_m}m). Separation state held.",
        )

    # 5. Calculate Haversine distance
    distance_m = haversine_meters(patient_lat, patient_lng, comp_lat, comp_lng)
    grid_cell_id = location_to_h3_cell(patient_lat, patient_lng)

    # 6. Evaluate state machine & debouncing
    action = "NONE"
    alert_tier = None
    alert_id = None
    hr_evaluated = False
    hr_bpm = None
    hr_provenance = None
    hr_z_score = None
    is_arousal = False

    async with async_session() as session:
        state_res = await session.execute(
            select(GroupSeparationStateRow).where(GroupSeparationStateRow.patient_id == patient_id)
        )
        state_row = state_res.scalar_one_or_none()

        if state_row is None:
            state_row = GroupSeparationStateRow(
                patient_id=patient_id,
                session_id=comp_row.session_id,
                status="WITH_COMPANION" if distance_m <= config.separation_distance_m else "SEPARATION_PENDING",
                consecutive_separated=0,
                consecutive_united=0,
                last_distance_m=distance_m,
                active_alert_id=None,
                last_evaluated_at=patient_timestamp,
                updated_at=now,
            )
            session.add(state_row)
            await session.flush()

        # Update distance and timestamp
        state_row.last_distance_m = distance_m
        state_row.last_evaluated_at = patient_timestamp
        state_row.updated_at = now

        # Debouncing logic
        is_separated_reading = distance_m > config.separation_distance_m

        if is_separated_reading:
            state_row.consecutive_separated += 1
            state_row.consecutive_united = 0
            if state_row.consecutive_separated >= config.persistence_readings:
                state_row.status = "SEPARATED"
            else:
                state_row.status = "SEPARATION_PENDING"
        else:
            state_row.consecutive_united += 1
            state_row.consecutive_separated = 0
            if state_row.status in ("SEPARATED", "REUNITE_PENDING"): 
                if state_row.consecutive_united >= config.reunite_readings:
                    state_row.status = "WITH_COMPANION"
                else:
                    state_row.status = "REUNITE_PENDING"
            else:
                state_row.status = "WITH_COMPANION"
        # -------------------------------------------------------------
        # Case A: Confirmed SEPARATED -> Check Biometrics & Alert Lifecycle
        # -------------------------------------------------------------
        if state_row.status == "SEPARATED":
            # 7. Evaluate separated person's heart rate
            wear_res = await session.execute(
                select(WearableReadingRow)
                .where(WearableReadingRow.user_id == patient_id)
                .order_by(desc(WearableReadingRow.timestamp))
                .limit(1)
            )
            latest_wear = wear_res.scalar_one_or_none()

            target_tier = 2  # Default SUSPICIOUS for separation without arousal

            if latest_wear and latest_wear.heart_rate_bpm is not None:
                wear_ts = latest_wear.timestamp if latest_wear.timestamp.tzinfo is not None else latest_wear.timestamp.replace(tzinfo=timezone.utc)
                wear_age_sec = (now - wear_ts).total_seconds()
                hr_provenance = latest_wear.sensor_status or "NOMINAL"
                hr_bpm = latest_wear.heart_rate_bpm

                if wear_age_sec <= config.max_reading_age_sec and latest_wear.wear_state == "ON_WRIST":
                    hr_evaluated = True

                    # Query baseline readings
                    hist_res = await session.execute(
                        select(WearableReadingRow.heart_rate_bpm)
                        .where(
                            WearableReadingRow.user_id == patient_id,
                            WearableReadingRow.wear_state == "ON_WRIST",
                        )
                        .order_by(desc(WearableReadingRow.timestamp))
                        .limit(50)
                    )
                    hr_values = hist_res.scalars().all()

                    if len(hr_values) >= 5:
                        mean_hr = sum(hr_values) / len(hr_values)
                        variance = sum((x - mean_hr) ** 2 for x in hr_values) / len(hr_values)
                        std_hr = max(math.sqrt(variance), 2.0)
                        hr_z_score = round((hr_bpm - mean_hr) / std_hr, 2)

                        if hr_z_score >= config.corroborating_z_score_threshold or hr_bpm >= config.tachycardia_bpm_threshold:
                            is_arousal = True
                            target_tier = 3
                    elif hr_bpm >= config.tachycardia_bpm_threshold:
                        is_arousal = True
                        target_tier = 3

            # Non-suppression rule check: even if HR is normal or missing, target_tier >= 2
            alert_tier = target_tier

            # Manage persistent AlertRecord
            active_alert = None
            if state_row.active_alert_id:
                alert_res = await session.execute(
                    select(AlertRecordRow).where(AlertRecordRow.alert_id == state_row.active_alert_id)
                )
                active_alert = alert_res.scalar_one_or_none()

            if active_alert and active_alert.status == AlertStatus.PENDING.value:
                # Update existing pending alert location and escalate if physiological arousal
                active_alert.timestamp = patient_timestamp
                active_alert.grid_cell_id = grid_cell_id
                if target_tier > active_alert.alert_tier:
                    active_alert.alert_tier = target_tier
                action = "ALERT_UPDATED"
                alert_id = active_alert.alert_id
                alert_rec = AlertRecord(
                    alert_id=active_alert.alert_id,
                    user_id=active_alert.user_id,
                    grid_cell_id=active_alert.grid_cell_id,
                    timestamp=active_alert.timestamp,
                    alert_tier=active_alert.alert_tier,
                    status=AlertStatus(active_alert.status),
                    zone_id=active_alert.zone_id,
                )
                await push_alert_update(patient_id, alert_rec.model_dump(mode="json"))
            else:
                # Create single persistent alert
                new_alert_id = uuid4().hex
                new_alert = AlertRecordRow(
                    alert_id=new_alert_id,
                    user_id=patient_id,
                    grid_cell_id=grid_cell_id,
                    timestamp=patient_timestamp,
                    alert_tier=target_tier,
                    status=AlertStatus.PENDING.value,
                    zone_id="group_separation",
                )
                session.add(new_alert)
                state_row.active_alert_id = new_alert_id
                action = "ALERT_TRIGGERED"
                alert_id = new_alert_id
                alert_rec = AlertRecord(
                    alert_id=new_alert.alert_id,
                    user_id=new_alert.user_id,
                    grid_cell_id=new_alert.grid_cell_id,
                    timestamp=new_alert.timestamp,
                    alert_tier=new_alert.alert_tier,
                    status=AlertStatus(new_alert.status),
                    zone_id=new_alert.zone_id,
                )
                await push_alert_update(patient_id, alert_rec.model_dump(mode="json"))

        # -------------------------------------------------------------
        # Case B: Confirmed Reunification (WITH_COMPANION) -> Auto-Resolve Alert
        # -------------------------------------------------------------
        elif state_row.status == "WITH_COMPANION":
            if state_row.active_alert_id:
                alert_res = await session.execute(
                    select(AlertRecordRow).where(AlertRecordRow.alert_id == state_row.active_alert_id)
                )
                active_alert = alert_res.scalar_one_or_none()
                if active_alert and active_alert.status == AlertStatus.PENDING.value:
                    active_alert.status = AlertStatus.AUTO_RESOLVED.value
                    active_alert.timestamp = patient_timestamp
                    action = "ALERT_RESOLVED"
                    alert_id = active_alert.alert_id
                    alert_rec = AlertRecord(
                        alert_id=active_alert.alert_id,
                        user_id=active_alert.user_id,
                        grid_cell_id=active_alert.grid_cell_id,
                        timestamp=active_alert.timestamp,
                        alert_tier=active_alert.alert_tier,
                        status=AlertStatus(active_alert.status),
                        zone_id=active_alert.zone_id,
                    )
                    await push_alert_update(patient_id, alert_rec.model_dump(mode="json"))
                state_row.active_alert_id = None

        await session.commit()

        status_text = state_row.status
        consec_sep = state_row.consecutive_separated
        consec_uni = state_row.consecutive_united

    msg = (
        f"Group status: {status_text} (Distance: {distance_m:.1f}m)."
        + (f" Physiological arousal corroborated ({hr_bpm} BPM, z={hr_z_score})." if is_arousal else "")
        + (f" Alert {action}." if action != "NONE" else "")
    )

    return GroupSeparationResult(
        status=status_text,
        patient_id=patient_id,
        companion_id=companion_id,
        distance_m=distance_m,
        action=action,
        alert_id=alert_id,
        alert_tier=alert_tier,
        heart_rate_evaluated=hr_evaluated,
        heart_rate_bpm=hr_bpm,
        heart_rate_provenance=hr_provenance,
        heart_rate_z_score=hr_z_score,
        is_physiological_arousal=is_arousal,
        consecutive_separated=consec_sep,
        consecutive_united=consec_uni,
        message=msg,
    )

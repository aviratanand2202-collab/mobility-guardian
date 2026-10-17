"""
Trusted Companion Mode endpoints.

Requirements & Privacy Architecture:
1. Caregiver creates an expiring companion invitation link for an authorized patient.
2. The invited companion explicitly consents to temporary location sharing before joining.
3. Companion does NOT become a permanent family member; personal ID is NEVER exposed
   on the caregiver dashboard (a privacy-preserving alias like "Trusted Companion" is used).
4. Timestamped companion location updates are verified against patient location to produce:
   - "With Companion" (distance <= threshold)
   - "Separated" (distance > threshold)
   - "Accompaniment Unverified" (missing, stale > 60s, or unavailable)
5. Caregiver can end the session at any time; companion can stop sharing at any time.
"""
from __future__ import annotations

import math
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, desc

from app.db import async_session, CompanionSessionRow, RiskScoreHistoryRow
from app.models import (
    AccompanimentStatus,
    CompanionConsentRequest,
    CompanionLocationUpdate,
    CompanionSessionCreate,
    CompanionSessionStatus,
    CompanionSessionSummary,
)
from app.routers.consent import is_consented
from app.routers.risk_history import _extract_location

router = APIRouter()


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


@router.post("/sessions/create", response_model=CompanionSessionSummary)
async def create_companion_session(payload: CompanionSessionCreate):
    """
    Caregiver starts a temporary companion session or associates an authorized family member.
    """
    # Enforce patient authorization/consent check
    if not await is_consented(payload.patient_id):
        raise HTTPException(
            status_code=403,
            detail="Consent not granted for this patient. Cannot invite companion without authorized patient consent.",
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=payload.validity_minutes)
    session_id = uuid4().hex
    invite_token = secrets.token_urlsafe(32)

    is_direct_family_link = bool(payload.companion_id and payload.companion_id.strip())
    companion_uid = payload.companion_id.strip() if is_direct_family_link else None
    rel = payload.relationship or ("family" if is_direct_family_link else "trusted_companion")
    alias = (
        payload.companion_alias.strip()
        if payload.companion_alias and payload.companion_alias.strip()
        else (f"Family Member ({companion_uid})" if is_direct_family_link else "Trusted Companion")
    )

    init_lat = None
    init_lng = None
    init_ts = None

    async with async_session() as session:
        # End any previous active/pending session for this patient to ensure a single active association
        existing_res = await session.execute(
            select(CompanionSessionRow).where(
                CompanionSessionRow.patient_id == payload.patient_id,
                CompanionSessionRow.status.in_([
                    CompanionSessionStatus.ACTIVE.value,
                    CompanionSessionStatus.PENDING_INVITE.value,
                ]),
            )
        )
        for old_row in existing_res.scalars().all():
            old_row.status = CompanionSessionStatus.ENDED.value

        if companion_uid:
            comp_hist_res = await session.execute(
                select(RiskScoreHistoryRow)
                .where(RiskScoreHistoryRow.user_id == companion_uid)
                .order_by(desc(RiskScoreHistoryRow.timestamp))
                .limit(1)
            )
            comp_hist = comp_hist_res.scalar_one_or_none()
            if comp_hist:
                loc = _extract_location(comp_hist.raw_output_json)
                if loc and loc.lat is not None and loc.lng is not None:
                    init_lat = loc.lat
                    init_lng = loc.lng
                    init_ts = (
                        comp_hist.timestamp
                        if comp_hist.timestamp.tzinfo is not None
                        else comp_hist.timestamp.replace(tzinfo=timezone.utc)
                    )

        row = CompanionSessionRow(
            session_id=session_id,
            patient_id=payload.patient_id,
            caregiver_id=payload.caregiver_id,
            invite_token=invite_token,
            status=CompanionSessionStatus.ACTIVE.value if is_direct_family_link else CompanionSessionStatus.PENDING_INVITE.value,
            created_at=now,
            expires_at=expires_at,
            consent_granted=True if is_direct_family_link else False,
            consent_granted_at=now if is_direct_family_link else None,
            companion_alias=alias,
            companion_user_id=companion_uid,
            relationship=rel,
            last_companion_lat=init_lat,
            last_companion_lng=init_lng,
            last_companion_timestamp=init_ts,
        )

        session.add(row)
        await session.commit()

    return CompanionSessionSummary(
        session_id=row.session_id,
        patient_id=row.patient_id,
        companion_user_id=row.companion_user_id,
        relationship=row.relationship,
        status=CompanionSessionStatus(row.status),
        invite_token=row.invite_token,
        created_at=row.created_at,
        expires_at=row.expires_at,
        consent_granted=row.consent_granted,
        consent_granted_at=row.consent_granted_at,
        companion_alias=row.companion_alias,
        last_companion_lat=row.last_companion_lat,
        last_companion_lng=row.last_companion_lng,
        last_companion_timestamp=row.last_companion_timestamp,
        is_location_stale=True if init_ts is None else (now - init_ts).total_seconds() > 60.0,
    )


@router.get("/invite/{invite_token}")
async def get_invite_details(invite_token: str):
    """
    Companion views invitation details before granting explicit consent.
    Ensures token is valid and not expired.
    """
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow).where(CompanionSessionRow.invite_token == invite_token)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="Invitation link is invalid or does not exist.")

        # Check expiration
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if now > expires_at:
            row.status = CompanionSessionStatus.EXPIRED.value
            await session.commit()
            raise HTTPException(status_code=410, detail="This invitation link has expired.")

        if row.status in (CompanionSessionStatus.ENDED.value, CompanionSessionStatus.REVOKED.value):
            raise HTTPException(status_code=400, detail="This companion session has already concluded.")

        return {
            "session_id": row.session_id,
            "patient_id": row.patient_id,
            "status": row.status,
            "expires_at": expires_at.isoformat(),
            "companion_alias": row.companion_alias,
            "privacy_disclosure": (
                "You are invited as a temporary trusted companion for this walking session. "
                "You do not become a permanent family member, and your personal ID will never "
                "be exposed on the caregiver dashboard. Location sharing will automatically "
                "cease when the session ends or when you click Stop Sharing."
            ),
        }


@router.post("/invite/{invite_token}/consent")
async def record_companion_consent(invite_token: str, payload: CompanionConsentRequest):
    """
    Companion explicitly grants or declines consent to share their location.
    """
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow).where(CompanionSessionRow.invite_token == invite_token)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="Invitation link is invalid.")

        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if now > expires_at:
            row.status = CompanionSessionStatus.EXPIRED.value
            await session.commit()
            raise HTTPException(status_code=410, detail="This invitation link has expired.")

        if row.status in (CompanionSessionStatus.ENDED.value, CompanionSessionStatus.REVOKED.value):
            raise HTTPException(status_code=400, detail="This companion session has already concluded.")

        if not payload.granted:
            row.status = CompanionSessionStatus.REVOKED.value
            row.consent_granted = False
            await session.commit()
            return {"status": "declined", "session_id": row.session_id}

        row.status = CompanionSessionStatus.ACTIVE.value
        row.consent_granted = True
        row.consent_granted_at = now
        if payload.companion_alias and payload.companion_alias.strip():
            row.companion_alias = payload.companion_alias.strip()

        await session.commit()

        return {
            "status": "ACTIVE",
            "session_id": row.session_id,
            "companion_token": row.invite_token,
            "companion_alias": row.companion_alias,
            "message": "Consent recorded. You are now sharing temporary location for this session.",
        }


@router.post("/sessions/{session_id}/location")
async def update_companion_location(session_id: str, payload: CompanionLocationUpdate):
    """
    Consenting companion transmits actual, timestamped device location updates.
    Enforces authorization, consent, and session active status.
    """
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow).where(CompanionSessionRow.session_id == session_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        # Enforce token match
        if row.invite_token != payload.companion_token:
            raise HTTPException(status_code=401, detail="Unauthorized companion token.")

        # Enforce consent and active status
        if not row.consent_granted or row.status != CompanionSessionStatus.ACTIVE.value:
            raise HTTPException(
                status_code=403,
                detail="Location updates rejected: Companion consent has not been granted or session is inactive.",
            )

        # Enforce expiration
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if now > expires_at:
            row.status = CompanionSessionStatus.EXPIRED.value
            await session.commit()
            raise HTTPException(status_code=410, detail="Session has expired. Location updates are discontinued.")

        ts = payload.timestamp or now
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        row.last_companion_lat = payload.lat
        row.last_companion_lng = payload.lng
        row.last_companion_accuracy_m = payload.accuracy_m
        row.last_companion_timestamp = ts

        await session.commit()

        return {
            "status": "recorded",
            "session_id": row.session_id,
            "timestamp": ts.isoformat(),
        }


def get_patient_configured_safe_zone(patient_id: str) -> tuple[Any, dict[str, Any]]:
    """
    Retrieve configured safe area boundary for a patient if one exists.
    Returns (safe_boundary_obj, safe_zone_info_dict).
    If no safe zone is configured for this patient, returns (None, {"configured": False, ...}).
    NEVER invents or falls back to hardcoded default coordinates.
    """
    try:
        from app.services.inference_service import get_inference_engine
        engine = get_inference_engine()
        boundary = engine.get_user_safe_area(patient_id)
        if boundary is None:
            engine.load_user_profile(patient_id)
            boundary = engine.get_user_safe_area(patient_id)

        if boundary and getattr(boundary, "zones", None):
            zone = boundary.zones[0]
            from ml.src.geofence import CircularSafeArea, PolygonSafeArea
            if isinstance(zone, CircularSafeArea):
                info = {
                    "configured": True,
                    "name": zone.name,
                    "type": "circle",
                    "center": [zone.center_latitude, zone.center_longitude],
                    "radius_m": zone.radius_m,
                }
            elif isinstance(zone, PolygonSafeArea):
                info = {
                    "configured": True,
                    "name": zone.name,
                    "type": "polygon",
                    "vertices": [[lat, lon] for lat, lon in zone.vertices],
                }
            else:
                info = {
                    "configured": True,
                    "name": getattr(zone, "name", "Configured Safe Zone"),
                    "type": "custom",
                }
            return boundary, info
    except Exception:
        pass

    return None, {
        "configured": False,
        "name": None,
        "description": f"No safe zone configured for patient {patient_id}",
    }


@router.get("/sessions/active/{patient_id}")
async def get_active_companion_session(
    patient_id: str,
    patient_lat: Optional[float] = Query(None, description="Current patient latitude"),
    patient_lng: Optional[float] = Query(None, description="Current patient longitude"),
    threshold_meters: float = Query(50.0, description="Cohesion distance threshold in meters"),
):
    """
    Retrieve current active or pending companion session for the selected patient.
    Evaluates real-time Accompaniment Status:
    - 'With Companion': Both fresh, distance <= threshold
    - 'Separated': Both fresh, distance > threshold
    - 'Accompaniment Unverified': Missing, stale (>60s), or unavailable
    """
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow)
            .where(
                CompanionSessionRow.patient_id == patient_id,
                CompanionSessionRow.status.in_([
                    CompanionSessionStatus.ACTIVE.value,
                    CompanionSessionStatus.PENDING_INVITE.value,
                ]),
            )
            .order_by(desc(CompanionSessionRow.created_at))
        )
        row = result.scalars().first()

        if row is None:
            return {"active_session": None}

        # Check expiration
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if now > expires_at:
            row.status = CompanionSessionStatus.EXPIRED.value
            await session.commit()
            return {"active_session": None}

        # If companion_user_id is set, sync their latest persisted telemetry from risk_score_history
        if row.companion_user_id:
            comp_hist_res = await session.execute(
                select(RiskScoreHistoryRow)
                .where(RiskScoreHistoryRow.user_id == row.companion_user_id)
                .order_by(desc(RiskScoreHistoryRow.timestamp))
                .limit(1)
            )
            comp_hist = comp_hist_res.scalar_one_or_none()
            if comp_hist:
                loc = _extract_location(comp_hist.raw_output_json)
                if loc and loc.lat is not None and loc.lng is not None:
                    hist_ts = (
                        comp_hist.timestamp
                        if comp_hist.timestamp.tzinfo is not None
                        else comp_hist.timestamp.replace(tzinfo=timezone.utc)
                    )
                    curr_ts = (
                        row.last_companion_timestamp
                        if row.last_companion_timestamp is None or row.last_companion_timestamp.tzinfo is not None
                        else row.last_companion_timestamp.replace(tzinfo=timezone.utc)
                    )
                    if curr_ts is None or hist_ts >= curr_ts:
                        row.last_companion_lat = loc.lat
                        row.last_companion_lng = loc.lng
                        row.last_companion_timestamp = hist_ts

        # If patient coordinates were not passed in query, pull patient's latest persisted location
        if patient_lat is None or patient_lng is None:
            pat_hist_res = await session.execute(
                select(RiskScoreHistoryRow)
                .where(RiskScoreHistoryRow.user_id == patient_id)
                .order_by(desc(RiskScoreHistoryRow.timestamp))
                .limit(1)
            )
            pat_hist = pat_hist_res.scalar_one_or_none()
            if pat_hist:
                p_loc = _extract_location(pat_hist.raw_output_json)
                if p_loc and p_loc.lat is not None and p_loc.lng is not None:
                    patient_lat = p_loc.lat
                    patient_lng = p_loc.lng

        # Evaluate accompaniment status
        accompaniment_status = AccompanimentStatus.ACCOMPANIMENT_UNVERIFIED
        separation_distance_m = None
        is_stale = True
        safe_zone_departure_accompanied = False

        # Retrieve actual configured safe zone for patient (NO hardcoded San Francisco coordinates)
        safe_boundary, safe_zone_info = get_patient_configured_safe_zone(row.patient_id)

        if row.status == CompanionSessionStatus.ACTIVE.value and row.last_companion_lat is not None and row.last_companion_lng is not None:
            comp_ts = row.last_companion_timestamp
            if comp_ts is not None:
                if comp_ts.tzinfo is None:
                    comp_ts = comp_ts.replace(tzinfo=timezone.utc)
                age_sec = (now - comp_ts).total_seconds()
                is_stale = age_sec > 60.0

            if not is_stale and patient_lat is not None and patient_lng is not None:
                dist = haversine_meters(
                    patient_lat,
                    patient_lng,
                    row.last_companion_lat,
                    row.last_companion_lng,
                )
                separation_distance_m = dist

                outside_zone_together = False
                if safe_boundary is not None:
                    from ml.src.geofence import SafeAreaState
                    pat_state, _, _ = safe_boundary.evaluate(patient_lat, patient_lng)
                    comp_state, _, _ = safe_boundary.evaluate(row.last_companion_lat, row.last_companion_lng)
                    outside_zone_together = (
                        pat_state == SafeAreaState.OUTSIDE_SAFE_AREA
                        and comp_state == SafeAreaState.OUTSIDE_SAFE_AREA
                        and dist <= threshold_meters
                    )

                if dist <= threshold_meters:
                    accompaniment_status = AccompanimentStatus.WITH_COMPANION
                    safe_zone_departure_accompanied = outside_zone_together
                else:
                    accompaniment_status = AccompanimentStatus.SEPARATED
            else:
                accompaniment_status = AccompanimentStatus.ACCOMPANIMENT_UNVERIFIED

        return {
            "active_session": {
                "session_id": row.session_id,
                "patient_id": row.patient_id,
                "companion_user_id": row.companion_user_id,
                "relationship": row.relationship,
                "status": row.status,
                "invite_token": row.invite_token,
                "created_at": row.created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "consent_granted": row.consent_granted,
                "consent_granted_at": row.consent_granted_at.isoformat() if row.consent_granted_at else None,
                "companion_alias": row.companion_alias,
                "last_companion_lat": row.last_companion_lat,
                "last_companion_lng": row.last_companion_lng,
                "last_companion_accuracy_m": row.last_companion_accuracy_m,
                "last_companion_timestamp": row.last_companion_timestamp.isoformat() if row.last_companion_timestamp else None,
                "accompaniment_status": accompaniment_status.value,
                "separation_distance_m": separation_distance_m,
                "is_location_stale": is_stale,
                "safe_zone_departure_accompanied": safe_zone_departure_accompanied,
                "safe_zone": safe_zone_info,
            }
        }


@router.post("/sessions/{session_id}/end")
async def end_companion_session(session_id: str):
    """
    Caregiver concludes the companion session.
    """
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow).where(CompanionSessionRow.session_id == session_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        row.status = CompanionSessionStatus.ENDED.value
        await session.commit()

        return {"status": "ended", "session_id": row.session_id}


@router.post("/sessions/{session_id}/stop-sharing")
async def companion_stop_sharing(session_id: str, companion_token: str = Query(...)):
    """
    Companion exercises their right to stop sharing location and leave the session at any time.
    """
    async with async_session() as session:
        result = await session.execute(
            select(CompanionSessionRow).where(CompanionSessionRow.session_id == session_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        if row.invite_token != companion_token:
            raise HTTPException(status_code=401, detail="Unauthorized token.")

        row.status = CompanionSessionStatus.REVOKED.value
        row.consent_granted = False
        await session.commit()

        return {"status": "revoked", "session_id": row.session_id, "message": "Location sharing stopped."}

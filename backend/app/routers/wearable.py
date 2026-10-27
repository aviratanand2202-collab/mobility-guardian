"""
Patient Wearable & Biometrics Router.

Provides endpoints to retrieve the latest and historical wearable biometric readings
(e.g., heart rate, wear state, sensor status) and associated physiological baseline evaluations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc

from app.db import async_session, WearableReadingRow
from app.models import WearableMetrics, WearState
from ml.src.heart_rate import (
    evaluate_heart_rate,
    load_user_baseline_from_db,
    DEFAULT_HR_CONFIG,
)

router = APIRouter()


class LatestWearableResponse(BaseModel):
    status: str
    user_id: str
    reading: Optional[WearableMetrics] = None
    raw_reading: Optional[Dict[str, Any]] = None
    age_seconds: Optional[float] = None
    is_stale: bool = False
    evidence: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


@router.get("/{user_id}/latest", response_model=LatestWearableResponse)
async def get_latest_wearable_reading(user_id: str):
    """
    Retrieve the most recent biometric wearable reading and physiological evaluation for a patient.
    """
    async with async_session() as session:
        query = (
            select(WearableReadingRow)
            .where(WearableReadingRow.user_id == user_id)
            .order_by(desc(WearableReadingRow.timestamp))
            .limit(1)
        )
        result = await session.execute(query)
        row = result.scalar_one_or_none()

    if not row:
        return LatestWearableResponse(
            status="NO_DATA",
            user_id=user_id,
            reading=None,
            raw_reading=None,
            age_seconds=None,
            is_stale=False,
            evidence=None,
            message=f"No wearable readings recorded for patient '{user_id}'."
        )

    now_utc = datetime.now(timezone.utc)
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_sec = max(0.0, (now_utc - ts).total_seconds())
    is_stale = age_sec > DEFAULT_HR_CONFIG.freshness_timeout_sec

    # Evaluate physiological evidence against personal baseline
    baseline = load_user_baseline_from_db(user_id)
    evidence = evaluate_heart_rate(
        reading={
            "heart_rate_bpm": row.heart_rate_bpm,
            "wear_state": row.wear_state,
            "timestamp": ts,
        },
        current_speed_mps=0.0,
        baseline=baseline,
        current_timestamp=now_utc,
    )

    wear_enum = WearState.UNKNOWN
    try:
        wear_enum = WearState(row.wear_state)
    except Exception:
        pass

    reading_model = WearableMetrics(
        heart_rate_bpm=row.heart_rate_bpm,
        timestamp=ts,
        wear_state=wear_enum,
        sensor_status=row.sensor_status,
    )

    raw_reading = {
        "heart_rate_bpm": row.heart_rate_bpm,
        "timestamp": ts.isoformat(),
        "wear_state": row.wear_state,
        "sensor_status": row.sensor_status,
    }

    return LatestWearableResponse(
        status="SUCCESS",
        user_id=user_id,
        reading=reading_model,
        raw_reading=raw_reading,
        age_seconds=round(age_sec, 1),
        is_stale=is_stale,
        evidence=evidence.to_dict() if evidence else None,
        message=None,
    )


@router.get("/{user_id}/history", response_model=List[Dict[str, Any]])
async def get_wearable_history(
    user_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max readings to return"),
):
    """
    Retrieve historical biometric readings for a patient ordered from newest to oldest.
    """
    async with async_session() as session:
        query = (
            select(WearableReadingRow)
            .where(WearableReadingRow.user_id == user_id)
            .order_by(desc(WearableReadingRow.timestamp))
            .limit(limit)
        )
        result = await session.execute(query)
        rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "timestamp": r.timestamp.isoformat(),
            "heart_rate_bpm": r.heart_rate_bpm,
            "wear_state": r.wear_state,
            "sensor_status": r.sensor_status,
        }
        for r in rows
    ]

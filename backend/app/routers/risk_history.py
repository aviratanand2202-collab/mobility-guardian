"""
Risk history query router.

Provides endpoints to query a patient's historical risk trend line
and inspect full raw RiskScoreOutput payloads.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.db import async_session, RiskScoreHistoryRow
from app.models import Location, WearableMetrics

router = APIRouter()


class RiskHistoryEntry(BaseModel):
    id: int
    timestamp: datetime
    risk_tier: str
    risk_score: float
    polling_tier: int
    grid_cell_id: Optional[str] = None
    location: Optional[Location] = None
    ml_evaluation: Optional[dict[str, Any]] = None
    wearable_metrics: Optional[WearableMetrics] = None


def _extract_ml_evaluation(raw_json: Optional[str]) -> Optional[dict[str, Any]]:
    """Extract ml_evaluation dict from raw_output_json if present.

    Returns None gracefully if raw_json is missing, invalid JSON,
    or predates the addition of shadow ML evaluation.
    """
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
        if isinstance(data, dict):
            ml_eval = data.get("ml_evaluation")
            if isinstance(ml_eval, dict):
                return ml_eval
    except Exception:
        return None
    return None


def _extract_wearable_metrics(raw_json: Optional[str]) -> Optional[WearableMetrics]:
    """Extract WearableMetrics from raw_output_json if present."""
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
        if isinstance(data, dict):
            w = data.get("wearable_metrics")
            if isinstance(w, dict):
                return WearableMetrics(**w)
    except Exception:
        return None
    return None


def _extract_location(raw_json: Optional[str]) -> Optional[Location]:
    """
    Extract Location model from raw_output_json if present.
    Returns None gracefully if raw_json is missing, invalid JSON,
    or predates the addition of the location field.
    """
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return None
        loc_data = data.get("location")
        if loc_data and isinstance(loc_data, dict):
            lat = loc_data.get("lat")
            lng = loc_data.get("lng")
            if lat is not None and lng is not None:
                return Location(
                    lat=lat,
                    lng=lng,
                    altitude_m=loc_data.get("altitude_m"),
                )
    except Exception:
        return None
    return None


def _parse_datetime(val: Optional[str], default: Optional[datetime] = None) -> Optional[datetime]:
    if val is None or not val.strip():
        return default
    try:
        cleaned = val.strip().replace(" ", "+")
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ISO datetime format: '{val}'. Expected ISO-8601 string.",
        )


@router.get("/{user_id}", response_model=List[RiskHistoryEntry])
async def get_user_risk_history(
    user_id: str,
    start: Optional[str] = Query(None, description="Start timestamp (ISO-8601 UTC)"),
    end: Optional[str] = Query(None, description="End timestamp (ISO-8601 UTC)"),
    limit: int = Query(500, description="Max results (default 500, capped at 2000)"),
) -> List[RiskHistoryEntry]:
    """
    Retrieve lightweight historical risk records for a patient over a time range,
    ordered ascending by timestamp for efficient trend line visualization.
    """
    start_dt = _parse_datetime(start, default=None)
    end_dt = _parse_datetime(end, default=None)

    # Respect and cap limit between 1 and 2000
    capped_limit = max(1, min(limit, 2000))

    async with async_session() as session:
        stmt = select(RiskScoreHistoryRow).where(RiskScoreHistoryRow.user_id == user_id)
        if start_dt is not None:
            stmt = stmt.where(RiskScoreHistoryRow.timestamp >= start_dt)
        if end_dt is not None:
            stmt = stmt.where(RiskScoreHistoryRow.timestamp <= end_dt)

        if start_dt is None:
            # When start is not specified, fetch the most recent records up to capped_limit
            # and sort ascending so the result is chronological
            stmt = stmt.order_by(RiskScoreHistoryRow.timestamp.desc()).limit(capped_limit)
            result = await session.execute(stmt)
            rows = list(reversed(result.scalars().all()))
        else:
            stmt = stmt.order_by(RiskScoreHistoryRow.timestamp.asc()).limit(capped_limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()

    return [
        RiskHistoryEntry(
            id=row.id,
            timestamp=(
                row.timestamp
                if row.timestamp.tzinfo is not None
                else row.timestamp.replace(tzinfo=timezone.utc)
            ),
            risk_tier=row.risk_tier,
            risk_score=row.risk_score,
            polling_tier=row.polling_tier,
            grid_cell_id=row.grid_cell_id,
            location=_extract_location(row.raw_output_json),
            ml_evaluation=_extract_ml_evaluation(row.raw_output_json),
            wearable_metrics=_extract_wearable_metrics(row.raw_output_json),
        )
        for row in rows
    ]


@router.get("/{user_id}/{history_id}/raw")
async def get_user_risk_history_raw(user_id: str, history_id: int):
    """
    Retrieve and return the full deserialized RiskScoreOutput JSON for a specific
    historical record, including kinematic_features, trigger_state, and explainability.
    """
    async with async_session() as session:
        result = await session.execute(
            select(RiskScoreHistoryRow).where(
                RiskScoreHistoryRow.id == history_id,
                RiskScoreHistoryRow.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Risk history record {history_id} not found for user {user_id}",
        )

    try:
        return json.loads(row.raw_output_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted raw history JSON")

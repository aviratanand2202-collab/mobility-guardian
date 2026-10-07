"""
Alert lifecycle endpoints: creation (typically pushed internally when risk
crosses Tier 2/3), caregiver acknowledgement, and dismissal.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import async_session, AlertRecordRow
from app.models import AlertRecord, AlertStatus, Dismissal
from app.state_machines.dismissal_quarantine import (
    get_or_create_cell_state,
    save_cell_state,
)

router = APIRouter()


@router.get("/{alert_id}")
async def get_alert(alert_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(AlertRecordRow).where(AlertRecordRow.alert_id == alert_id)
        )
        row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return _row_to_alert_record(row)


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(alert_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(AlertRecordRow).where(AlertRecordRow.alert_id == alert_id)
        )
        row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="alert not found")

    cell_state = await get_or_create_cell_state(row.grid_cell_id)
    now = datetime.now(timezone.utc)
    allowed, count = cell_state.register_dismissal(now)
    await save_cell_state(cell_state)

    if not allowed:
        # 4th+ dismissal in window - force formal recalibration instead
        # of quick-dismiss.
        return {
            "status": "recalibration_required",
            "message": "Routine change detected. Add this route to the "
                       "Safe Mobility Profile permanently?",
        }

    row.status = AlertStatus.DISMISSED_SAFE.value
    row.dismissed_at = now
    row.dismissal_decay_clock_start = cell_state.decay_clock_start
    row.dismissal_count_in_window = count

    async with async_session() as session:
        await session.merge(row)
        await session.commit()

    return {
        "status": "dismissed",
        "dismissal_count_in_window": count,
        "cell_sensitivity_multiplier": cell_state.sensitivity_multiplier(),
    }


def _row_to_alert_record(row: AlertRecordRow) -> AlertRecord:
    """Reconstruct an AlertRecord Pydantic model from an SQLite row."""
    dismissal = None
    if row.dismissed_at is not None:
        decay_clock = row.dismissal_decay_clock_start
        if decay_clock is not None and decay_clock.tzinfo is None:
            decay_clock = decay_clock.replace(tzinfo=timezone.utc)
        dismissed_at = row.dismissed_at
        if dismissed_at.tzinfo is None:
            dismissed_at = dismissed_at.replace(tzinfo=timezone.utc)
        dismissal = Dismissal(
            dismissed_at=dismissed_at,
            decay_clock_start=decay_clock,
            dismissal_count_in_window=row.dismissal_count_in_window or 0,
        )
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return AlertRecord(
        alert_id=row.alert_id,
        user_id=row.user_id,
        grid_cell_id=row.grid_cell_id,
        timestamp=ts,
        alert_tier=row.alert_tier,
        status=AlertStatus(row.status),
        dismissal=dismissal,
    )

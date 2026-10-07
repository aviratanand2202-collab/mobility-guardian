"""
Telemetry ingestion endpoint. Client (mobile app / simulator) POSTs here
on every GPS/sensor reading per shared/schema.json TelemetryPayload.
"""
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import async_session, AlertRecordRow, save_risk_history
from app.models import (
    AlertRecord,
    AlertStatus,
    PDRTierState,
    RiskTier,
    SignalState,
    TelemetryPayload,
)
from app.routers.consent import is_consented
from app.services.geospatial import location_to_h3_cell
from app.services.mock_risk_engine import compute_risk
from app.state_machines.dismissal_quarantine import get_or_create_cell_state
from app.state_machines.pdr_hysteresis import get_or_create_state, save_state
from app.websocket.risk_stream import push_alert_update, push_risk_update

router = APIRouter()


@router.post("/ingest")
async def ingest_telemetry(payload: TelemetryPayload):
    # Reject telemetry for any user without a granted consent record.
    if not await is_consented(payload.user_id):
        raise HTTPException(
            status_code=403,
            detail="Consent not granted for this user. "
                   "Record consent before sending telemetry.",
        )

    # Derive H3 grid cell ID (resolution 9) once from telemetry coordinates
    grid_cell_id = location_to_h3_cell(payload.location.lat, payload.location.lng)
    cell_state = await get_or_create_cell_state(grid_cell_id)

    pdr_tier = None
    if payload.signal_status.state == SignalState.DEGRADED_SIGNAL:
        if payload.imu_metrics is None:
            raise HTTPException(
                status_code=422,
                detail="imu_metrics required when signal_status.state == DEGRADED_SIGNAL",
            )
        pdr_state = await get_or_create_state(payload.user_id)
        tier = pdr_state.update(payload.imu_metrics.net_displacement_m)
        await save_state(payload.user_id, pdr_state)
        pdr_tier = PDRTierState(tier.value)

    # Compute risk with suppression multiplier applied from cell dismissal state
    risk_output = compute_risk(payload, pdr_tier, cell_dismissal_state=cell_state)
    await push_risk_update(payload.user_id, risk_output.model_dump(mode="json"))

    # If risk is Tier 2 (SUSPICIOUS) or Tier 3 (CRITICAL), create or update an AlertRecord
    if risk_output.risk_tier in (RiskTier.SUSPICIOUS, RiskTier.CRITICAL):
        alert_tier_num = 2 if risk_output.risk_tier == RiskTier.SUSPICIOUS else 3

        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(
                    AlertRecordRow.user_id == payload.user_id,
                    AlertRecordRow.grid_cell_id == grid_cell_id,
                    AlertRecordRow.status == AlertStatus.PENDING.value,
                )
            )
            existing_alert = result.scalar_one_or_none()

            if existing_alert is not None:
                # Update existing pending alert's timestamp instead of creating a duplicate
                existing_alert.timestamp = payload.timestamp
                existing_alert.alert_tier = alert_tier_num
                await session.commit()
                alert_row = existing_alert
            else:
                new_alert = AlertRecordRow(
                    alert_id=uuid4().hex,
                    user_id=payload.user_id,
                    grid_cell_id=grid_cell_id,
                    timestamp=payload.timestamp,
                    alert_tier=alert_tier_num,
                    status=AlertStatus.PENDING.value,
                )
                session.add(new_alert)
                await session.commit()
                alert_row = new_alert

        alert_record = AlertRecord(
            alert_id=alert_row.alert_id,
            user_id=alert_row.user_id,
            grid_cell_id=alert_row.grid_cell_id,
            timestamp=alert_row.timestamp,
            alert_tier=alert_row.alert_tier,
            status=AlertStatus(alert_row.status),
            dismissal=None,
        )
        await push_alert_update(payload.user_id, alert_record.model_dump(mode="json"))

    # Persist computed risk score output to history (Stage 6)
    await save_risk_history(
        user_id=payload.user_id,
        timestamp=risk_output.timestamp,
        risk_tier=risk_output.risk_tier.value,
        risk_score=risk_output.risk_score,
        polling_tier=risk_output.polling_tier,
        grid_cell_id=grid_cell_id,
        raw_output_json=risk_output.model_dump_json(),
    )

    return risk_output

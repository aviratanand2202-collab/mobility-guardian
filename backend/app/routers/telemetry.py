"""
Telemetry ingestion endpoint. Client (mobile app / simulator) POSTs here
on every GPS/sensor reading per shared/schema.json TelemetryPayload.
"""
from fastapi import APIRouter, HTTPException

from app.models import TelemetryPayload, SignalState, PDRTierState
from app.routers.consent import is_consented
from app.services.mock_risk_engine import compute_risk
from app.state_machines.pdr_hysteresis import get_or_create_state, save_state
from app.websocket.risk_stream import push_risk_update

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
        # TODO: if pdr_state.should_alert_caregiver(), wake GPS for forced
        # lock and push a Tier 2 alert (see alerts.py / risk_stream.py).

    risk_output = compute_risk(payload, pdr_tier)
    await push_risk_update(payload.user_id, risk_output.model_dump(mode="json"))
    return risk_output


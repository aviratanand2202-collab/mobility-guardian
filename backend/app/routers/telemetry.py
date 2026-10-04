"""
Telemetry ingestion endpoint. Client (mobile app / simulator) POSTs here
on every GPS/sensor reading per shared/schema.json TelemetryPayload.
"""
from fastapi import APIRouter, HTTPException

from app.models import TelemetryPayload, SignalState
from app.state_machines.pdr_hysteresis import get_or_create_state

router = APIRouter()


@router.post("/ingest")
def ingest_telemetry(payload: TelemetryPayload):
    # TODO: verify ConsentRecord exists and is granted for payload.user_id
    # before accepting telemetry (see routers/consent.py).

    if payload.signal_status.state == SignalState.DEGRADED_SIGNAL:
        if payload.imu_metrics is None:
            raise HTTPException(
                status_code=422,
                detail="imu_metrics required when signal_status.state == DEGRADED_SIGNAL",
            )
        pdr_state = get_or_create_state(payload.user_id)
        tier = pdr_state.update(payload.imu_metrics.net_displacement_m)
        # TODO: if pdr_state.should_alert_caregiver(), wake GPS for forced
        # lock and push a Tier 2 alert (see alerts.py / risk_stream.py).
        return {"status": "ingested", "pdr_tier": tier.value}

    # TODO: forward valid-signal telemetry to the ML risk engine (either
    # via direct function call if colocated, or an internal HTTP/queue
    # call once ml/ is deployed as its own service) and push the
    # resulting RiskScoreOutput to the WebSocket stream.
    return {"status": "ingested", "pdr_tier": None}

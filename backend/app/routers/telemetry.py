"""
Telemetry ingestion endpoint. Client (mobile app / simulator) POSTs here
on every GPS/sensor reading per shared/schema.json TelemetryPayload.
"""
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import async_session, AlertRecordRow, save_risk_history, save_wearable_reading

from app.models import (
    AlertRecord,
    AlertStatus,
    PDRTierState,
    PollingInstruction,
    PollingMode,
    RiskTier,
    SignalState,
    TelemetryPayload,
)
try:
    from app.config import is_behavioral_risk_escalation_enabled
except ImportError:
    def is_behavioral_risk_escalation_enabled() -> bool:
        import os
        return os.getenv("ENABLE_BEHAVIORAL_RISK_ESCALATION", "true").lower() in ("1", "true", "yes")
from app.routers.consent import is_consented
from app.services.geospatial import location_to_h3_cell
from app.services.inference_service import evaluate_shadow_inference
from app.services.mock_risk_engine import compute_risk, TopFeature
from app.state_machines.battery_pulse import compute_polling_instruction
from app.state_machines.behavioral_escalation import (
    evaluate_behavioral_escalation,
    get_or_create_behavioral_state,
)
from app.state_machines.dismissal_quarantine import get_or_create_cell_state
from app.state_machines.pdr_hysteresis import get_or_create_state, save_state

from app.state_machines.geofence_boundary import evaluate_patient_geofence_boundaries
from app.state_machines.loop_lifecycle import evaluate_loop_alert_lifecycle
from app.state_machines.group_separation import evaluate_group_separation

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

    # Persist optional wearable metrics reading if provided (decoupled from GPS risk engine)
    wearable_m = getattr(payload, "wearable_metrics", None)
    if wearable_m is not None and save_wearable_reading is not None:
        await save_wearable_reading(
            user_id=payload.user_id,
            timestamp=wearable_m.timestamp,
            heart_rate_bpm=wearable_m.heart_rate_bpm,
            wear_state=(
                wearable_m.wear_state.value
                if hasattr(wearable_m.wear_state, "value")
                else str(wearable_m.wear_state)
            ),
            sensor_status=wearable_m.sensor_status,
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

    # Attach live wearable and sensor metrics to risk_output so downstream WS clients receive them
    if wearable_m is not None and hasattr(risk_output, "wearable_metrics"):
        risk_output.wearable_metrics = wearable_m
    if getattr(payload, "sensor_metrics", None) is not None and hasattr(risk_output, "sensor_metrics"):
        risk_output.sensor_metrics = payload.sensor_metrics

    # Observational shadow-mode ML inference (governed by ENABLE_ML_INFERENCE_ENGINE)
    ml_eval = await evaluate_shadow_inference(payload)
    if ml_eval is not None:
        risk_output.ml_evaluation = ml_eval

    # Behavioral operational bridge (governed by ENABLE_BEHAVIORAL_RISK_ESCALATION)
    if is_behavioral_risk_escalation_enabled():
        escalated, reason = evaluate_behavioral_escalation(payload, ml_eval)
        if escalated:
            b_state = get_or_create_behavioral_state(payload.user_id)
            target_tier = getattr(b_state, "target_tier", RiskTier.SUSPICIOUS) or RiskTier.SUSPICIOUS
            target_score = getattr(b_state, "target_score", 65.0) or 65.0

            if target_tier == RiskTier.CRITICAL:
                risk_output.risk_tier = RiskTier.CRITICAL
                risk_output.risk_score = max(risk_output.risk_score, target_score)
                risk_output.polling_tier = max(risk_output.polling_tier, 3)
            elif risk_output.risk_tier in (RiskTier.QUIESCENT, RiskTier.NORMAL_TRANSIT):
                risk_output.risk_tier = RiskTier.SUSPICIOUS
                risk_output.risk_score = max(risk_output.risk_score, target_score)
                risk_output.polling_tier = max(risk_output.polling_tier, 2)

            if risk_output.predicted_lead_time_sec is None:
                risk_output.predicted_lead_time_sec = round(
                    max(240.0, 900.0 - (risk_output.risk_score - 65.0) * 15.0), 1
                )

            # Recompute polling instruction to ensure consistency with final operational tier and battery state
            battery_pct = payload.sensor_metrics.battery_pct
            polling_inst_dc = compute_polling_instruction(risk_output.risk_tier.value, battery_pct)
            risk_output.battery_override_active = polling_inst_dc.mode != "CONTINUOUS"
            risk_output.polling_instruction = PollingInstruction(
                mode=PollingMode(polling_inst_dc.mode),
                burst_seconds=polling_inst_dc.burst_seconds,
                burst_hz=polling_inst_dc.burst_hz,
                sleep_seconds=polling_inst_dc.sleep_seconds,
            )

            # Prepend behavioral and physiological top features for explainability if present
            if hasattr(risk_output, "explainability") and risk_output.explainability:
                top_feats = getattr(risk_output.explainability, "top_features", None)
                if isinstance(top_feats, list):
                    if not any(f.feature == "behavioral_trajectory_anomaly" for f in top_feats):
                        top_feats.insert(
                            0,
                            TopFeature(
                                feature="behavioral_trajectory_anomaly",
                                shap_value=0.45,
                                human_readable=reason or "Sustained anomalous movement pattern detected.",
                            ),
                        )
                    if getattr(b_state, "is_hr_corroborated", False) and not any(f.feature == "physiological_arousal_corroboration" for f in top_feats):
                        hr_detail = getattr(b_state, "hr_corroboration_detail", "")
                        top_feats.insert(
                            1,
                            TopFeature(
                                feature="physiological_arousal_corroboration",
                                shap_value=0.35,
                                human_readable=hr_detail or "Concomitant physiological arousal relative to baseline.",
                            ),
                        )

    # Phase 2: Evaluate safe-zone boundary transition state machine
    geofence_eval = None
    if evaluate_patient_geofence_boundaries is not None:
        geofence_eval = await evaluate_patient_geofence_boundaries(
            user_id=payload.user_id,
            lat=payload.location.lat,
            lng=payload.location.lng,
            accuracy_m=payload.sensor_metrics.horizontal_accuracy_m,
            timestamp=payload.timestamp,
            is_signal_degraded=(payload.signal_status.state == SignalState.DEGRADED_SIGNAL),
        )

    # Phase 3: Evaluate trajectory persistent looping lifecycle
    if evaluate_loop_alert_lifecycle is not None:
        loop_ev = None
        hr_ev = None
        if getattr(risk_output, "ml_evaluation", None):
            loop_ev = risk_output.ml_evaluation.get("loop_evidence")
            hr_ev = risk_output.ml_evaluation.get("heart_rate_evidence")

        is_inside_safe = False
        if geofence_eval and hasattr(geofence_eval, "zone_states"):
            is_inside_safe = any(zs.get("state") == "INSIDE" for zs in geofence_eval.zone_states)

        await evaluate_loop_alert_lifecycle(
            user_id=payload.user_id,
            lat=payload.location.lat,
            lng=payload.location.lng,
            timestamp=payload.timestamp,
            loop_evidence=loop_ev,
            hr_evidence=hr_ev,
            is_inside_safe_zone=is_inside_safe,
        )

    # Phase 4: Evaluate group/companion separation lifecycle and physiological escalation
    if evaluate_group_separation is not None:
        await evaluate_group_separation(
            patient_id=payload.user_id,
            patient_lat=payload.location.lat,
            patient_lng=payload.location.lng,
            patient_accuracy_m=payload.sensor_metrics.horizontal_accuracy_m,
            patient_timestamp=payload.timestamp,
            patient_is_degraded=(payload.signal_status.state == SignalState.DEGRADED_SIGNAL),
        )

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
                alert_zone_id = None
                if is_behavioral_risk_escalation_enabled():
                    b_state = get_or_create_behavioral_state(payload.user_id)
                    if getattr(b_state, "is_escalated", False):
                        alert_zone_id = "behavioral_escalation"

                new_alert = AlertRecordRow(
                    alert_id=uuid4().hex,
                    user_id=payload.user_id,
                    grid_cell_id=grid_cell_id,
                    timestamp=payload.timestamp,
                    alert_tier=alert_tier_num,
                    status=AlertStatus.PENDING.value,
                    zone_id=alert_zone_id,
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
            zone_id=getattr(alert_row, "zone_id", None),
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

"""
Stage 3 smoke & integration tests — verifies the Client Polling Instruction.

Run with:  python -m pytest backend/tests/test_stage3_polling_instruction.py -v
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure backend directory is in sys.path when run from repo root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import app
from app.db import (
    init_db,
    dispose_engine,
    async_session,
    ConsentRecordRow,
    CellDismissalStateRow,
    PDRStateRow,
    AlertRecordRow,
)
from app.models import (
    TelemetryPayload,
    Location,
    SensorMetrics,
    SignalStatus,
    SignalState,
    PDRTierState,
    RiskTier,
    PollingMode,
)
from app.services.mock_risk_engine import compute_risk
from app.state_machines.battery_pulse import (
    PULSE_BURST_SECONDS,
    PULSE_BURST_HZ,
    PULSE_SLEEP_SECONDS,
)


@pytest.fixture(scope="session", autouse=True)
def _db():
    """Initialize tables once for the test session and clean up on exit."""
    asyncio.run(init_db())
    yield
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def _clean_tables():
    """Clean all tables before each test to ensure test isolation."""
    async def _clean():
        async with async_session() as session:
            await session.execute(delete(ConsentRecordRow))
            await session.execute(delete(CellDismissalStateRow))
            await session.execute(delete(PDRStateRow))
            await session.execute(delete(AlertRecordRow))
            await session.commit()
    asyncio.run(_clean())
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _grant_consent(client: TestClient, user_id: str) -> None:
    consent = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-04T12:00:00Z",
            "guardian_id": "guardian_stage3",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


# ---------------------------------------------------------------
# Check 1: SUSPICIOUS + battery_pct = 10 -> PULSE + battery_override_active = True
#          Verified on raw HTTP response JSON (r.json())
# ---------------------------------------------------------------

def test_suspicious_low_battery_pulse_instruction_http_json(client: TestClient):
    """
    Check 1: SUSPICIOUS risk_tier + battery_pct = 10 (below 15% threshold)
    returns polling_instruction with mode='PULSE', burst_seconds=5, burst_hz=1,
    sleep_seconds=175, AND battery_override_active=True.
    Explicitly asserts polling_instruction is present in raw deserialized r.json().
    """
    user_id = "patient_stage3_pulse"
    _grant_consent(client, user_id)

    # 1. Step to ZONE_TRANSITION (30m)
    p1 = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:02:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 25.0,
            "speed_mps": 0.5,
            "heading_deg": 90.0,
            "battery_pct": 10,
        },
        "imu_metrics": {
            "step_count_since_last_gps": 35,
            "net_displacement_m": 30.0,
            "pdr_tier_state": "ZONE_TRANSITION",
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL",
            "degraded_since": "2026-10-04T12:00:00Z",
        },
    }
    r1 = client.post("/api/telemetry/ingest", json=p1)
    assert r1.status_code == 200

    # 2. Step to UNTRACKED_DISPLACEMENT (60m) -> maps to SUSPICIOUS
    p2 = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:05:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 28.0,
            "speed_mps": 0.2,
            "heading_deg": 90.0,
            "battery_pct": 10,
        },
        "imu_metrics": {
            "step_count_since_last_gps": 75,
            "net_displacement_m": 60.0,
            "pdr_tier_state": "ZONE_TRANSITION",
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL",
            "degraded_since": "2026-10-04T12:00:00Z",
        },
    }
    r2 = client.post("/api/telemetry/ingest", json=p2)
    assert r2.status_code == 200
    data = r2.json()

    assert data["risk_tier"] == "SUSPICIOUS"
    assert data["battery_override_active"] is True

    # Assert polling_instruction is present and correctly structured in raw HTTP response JSON
    assert "polling_instruction" in data
    pi = data["polling_instruction"]
    assert pi["mode"] == "PULSE"
    assert pi["burst_seconds"] == PULSE_BURST_SECONDS == 5
    assert pi["burst_hz"] == PULSE_BURST_HZ == 1
    assert pi["sleep_seconds"] == PULSE_SLEEP_SECONDS == 175


# ---------------------------------------------------------------
# Check 2: CRITICAL + battery_pct = 50 -> CONTINUOUS, battery_override_active = False
# ---------------------------------------------------------------

def test_critical_normal_battery_continuous_instruction():
    """
    Check 2: CRITICAL risk_tier + battery_pct = 50 ->
    polling_instruction.mode == 'CONTINUOUS', battery_override_active == False.
    """
    telemetry = TelemetryPayload(
        user_id="user_crit_ok",
        timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
        location=Location(lat=51.5, lng=-0.1),
        sensor_metrics=SensorMetrics(
            horizontal_accuracy_m=4.0,
            speed_mps=7.0,
            heading_deg=180.0,
            battery_pct=50,
        ),
        signal_status=SignalStatus(state=SignalState.VALID),
    )

    out = compute_risk(telemetry, None)
    # Ensure risk tier is CRITICAL
    assert out.risk_tier == RiskTier.CRITICAL
    assert out.battery_override_active is False
    assert out.polling_instruction is not None
    assert out.polling_instruction.mode == PollingMode.CONTINUOUS
    assert out.polling_instruction.burst_seconds is None
    assert out.polling_instruction.sleep_seconds is None


# ---------------------------------------------------------------
# Check 3: CRITICAL + battery_pct = 3 -> LAST_GASP, battery_override_active = True
# ---------------------------------------------------------------

def test_critical_critically_low_battery_last_gasp():
    """
    Check 3: CRITICAL risk_tier + battery_pct = 3 (< 5% threshold) ->
    polling_instruction.mode == 'LAST_GASP', battery_override_active == True.
    """
    telemetry = TelemetryPayload(
        user_id="user_crit_low",
        timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
        location=Location(lat=51.5, lng=-0.1),
        sensor_metrics=SensorMetrics(
            horizontal_accuracy_m=4.0,
            speed_mps=7.0,
            heading_deg=180.0,
            battery_pct=3,
        ),
        signal_status=SignalStatus(state=SignalState.VALID),
    )

    out = compute_risk(telemetry, None)
    assert out.risk_tier == RiskTier.CRITICAL
    assert out.battery_override_active is True
    assert out.polling_instruction is not None
    assert out.polling_instruction.mode == PollingMode.LAST_GASP


# ---------------------------------------------------------------
# Check 4: QUIESCENT / NORMAL_TRANSIT regardless of battery -> CONTINUOUS
# ---------------------------------------------------------------

def test_quiescent_and_normal_transit_continuous_regardless_of_battery():
    """
    Check 4: QUIESCENT and NORMAL_TRANSIT risk tiers regardless of battery
    (even at 2% or 10%) return polling_instruction.mode == 'CONTINUOUS',
    battery_override_active == False.
    """
    base_loc = Location(lat=51.5, lng=-0.1)
    base_signal = SignalStatus(state=SignalState.DEGRADED_SIGNAL)

    for battery in [2, 10, 50, 100]:
        # QUIESCENT (INDOOR_PACING)
        t_quiescent = TelemetryPayload(
            user_id="u_q",
            timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
            location=base_loc,
            sensor_metrics=SensorMetrics(
                horizontal_accuracy_m=25.0,
                speed_mps=0.0,
                heading_deg=0.0,
                battery_pct=battery,
            ),
            signal_status=base_signal,
        )
        out_q = compute_risk(t_quiescent, PDRTierState.INDOOR_PACING)
        assert out_q.risk_tier == RiskTier.QUIESCENT
        assert out_q.battery_override_active is False
        assert out_q.polling_instruction.mode == PollingMode.CONTINUOUS

        # NORMAL_TRANSIT (ZONE_TRANSITION)
        t_transit = TelemetryPayload(
            user_id="u_t",
            timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
            location=base_loc,
            sensor_metrics=SensorMetrics(
                horizontal_accuracy_m=25.0,
                speed_mps=0.5,
                heading_deg=0.0,
                battery_pct=battery,
            ),
            signal_status=base_signal,
        )
        out_t = compute_risk(t_transit, PDRTierState.ZONE_TRANSITION)
        assert out_t.risk_tier == RiskTier.NORMAL_TRANSIT
        assert out_t.battery_override_active is False
        assert out_t.polling_instruction.mode == PollingMode.CONTINUOUS


# ---------------------------------------------------------------
# Check: WebSocket stream receives polling_instruction
# ---------------------------------------------------------------

def test_websocket_receives_polling_instruction(client: TestClient):
    """
    Verify that WebSocket clients on /ws/risk/{user_id} receive the
    polling_instruction object matching the HTTP response.
    """
    user_id = "patient_stage3_ws"
    _grant_consent(client, user_id)

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:20:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 3.0,
            "speed_mps": 1.0,
            "heading_deg": 90.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }

    with client.websocket_connect(f"/ws/risk/{user_id}") as ws:
        r = client.post("/api/telemetry/ingest", json=payload)
        assert r.status_code == 200
        http_data = r.json()

        ws_data = ws.receive_json()
        assert "polling_instruction" in ws_data
        assert ws_data["polling_instruction"] == http_data["polling_instruction"]

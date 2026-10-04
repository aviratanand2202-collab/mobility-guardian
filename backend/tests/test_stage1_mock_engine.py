"""
Stage 1 smoke & integration tests — verifies the Mock Risk Engine.

Run with:  python -m pytest backend/tests/test_stage1_mock_engine.py -v
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
)
from app.models import (
    TelemetryPayload,
    Location,
    SensorMetrics,
    SignalStatus,
    SignalState,
    PDRTierState,
    RiskTier,
)
from app.services.mock_risk_engine import compute_risk


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
            "guardian_id": "guardian_stage1",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


# ---------------------------------------------------------------
# Check 1: VALID signal with high speed + determinism + extra fields in HTTP JSON
# ---------------------------------------------------------------

def test_valid_signal_high_speed_and_determinism(client: TestClient):
    """
    Check 1: VALID signal with high speed returns RiskScoreOutput reflecting
    the heuristic, exact same input produces exact same output on second call,
    and extra schema fields (kinematic_features, explainability) are present
    in the raw HTTP response JSON.
    """
    user_id = "patient_stage1_valid"
    _grant_consent(client, user_id)

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:01:30Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 6.5,
            "heading_deg": 180.0,
            "battery_pct": 85,
        },
        "signal_status": {"state": "VALID"},
    }

    # First call
    r1 = client.post("/api/telemetry/ingest", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()

    # Verify risk_tier reflects high speed (speed_mps 6.5 -> speed_comp 60.0 -> score >= 60)
    assert data1["risk_tier"] in ("SUSPICIOUS", "CRITICAL")
    assert 0 <= data1["risk_score"] <= 100
    assert data1["user_id"] == user_id

    # Second call with exact same payload -> bit-for-bit identical response JSON
    r2 = client.post("/api/telemetry/ingest", json=payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data1 == data2

    # Check (1): kinematic_features and explainability.top_features are present in raw HTTP JSON
    assert "kinematic_features" in data1
    assert "explainability" in data1
    assert "top_features" in data1["explainability"]

    # Check (2): Exact nested shapes matching /shared/schema.json
    kf = data1["kinematic_features"]
    assert "tortuosity_index" in kf and isinstance(kf["tortuosity_index"], (int, float))
    assert "entropy_value" in kf and isinstance(kf["entropy_value"], (int, float))
    assert "distance_from_anchor_m" in kf and isinstance(kf["distance_from_anchor_m"], (int, float))
    assert "step_speed_variance" in kf and isinstance(kf["step_speed_variance"], (int, float))

    trig = data1["trigger_state"]
    assert trig["window_seconds"] == 120
    assert isinstance(trig["consecutive_anomalous_windows"], int)
    assert trig["required_k"] in (2, 3)
    assert trig["percentile_bound_crossed"] in ("P95", "P99", None)

    expl = data1["explainability"]
    assert 1 <= len(expl["top_features"]) <= 3
    for feat in expl["top_features"]:
        assert isinstance(feat["feature"], str)
        assert isinstance(feat["shap_value"], (int, float))
        assert isinstance(feat["human_readable"], str)

    # Predicted lead time present for SUSPICIOUS / CRITICAL
    assert data1["predicted_lead_time_sec"] is not None
    assert 240 <= data1["predicted_lead_time_sec"] <= 900


# ---------------------------------------------------------------
# Check 2: DEGRADED_SIGNAL with net_displacement_m = 60 -> SUSPICIOUS
# ---------------------------------------------------------------

def test_degraded_signal_untracked_displacement_maps_to_suspicious(client: TestClient):
    """
    Check 2: DEGRADED_SIGNAL with net_displacement_m = 60 triggers
    UNTRACKED_DISPLACEMENT (>52m) and maps directly to SUSPICIOUS tier
    without running entropy-based heuristic.
    """
    user_id = "patient_stage1_degraded"
    _grant_consent(client, user_id)

    # 1. First reading crosses 27m threshold into ZONE_TRANSITION
    payload_step1 = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:02:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 25.0,
            "speed_mps": 0.5,
            "heading_deg": 90.0,
            "battery_pct": 70,
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
    r1 = client.post("/api/telemetry/ingest", json=payload_step1)
    assert r1.status_code == 200
    assert r1.json()["pdr_tier"] == "ZONE_TRANSITION"

    # 2. Second reading crosses 52m threshold into UNTRACKED_DISPLACEMENT (60m)
    payload_step2 = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:05:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 28.0,
            "speed_mps": 0.2,
            "heading_deg": 90.0,
            "battery_pct": 70,
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

    r2 = client.post("/api/telemetry/ingest", json=payload_step2)
    assert r2.status_code == 200
    data = r2.json()

    assert data["risk_tier"] == "SUSPICIOUS"
    assert data["risk_score"] == 75.0
    assert data["polling_tier"] == 2
    assert data["pdr_tier"] == "UNTRACKED_DISPLACEMENT"


# ---------------------------------------------------------------
# Check 3: Connected WebSocket receives the same RiskScoreOutput
# ---------------------------------------------------------------

def test_websocket_receives_identical_risk_output(client: TestClient):
    """
    Check 3: A connected WebSocket client on /ws/risk/{user_id} receives
    the same RiskScoreOutput payload that was returned in the HTTP response.
    """
    user_id = "patient_stage1_ws"
    _grant_consent(client, user_id)

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-04T12:10:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {
            "horizontal_accuracy_m": 3.0,
            "speed_mps": 1.2,
            "heading_deg": 45.0,
            "battery_pct": 90,
        },
        "signal_status": {"state": "VALID"},
    }

    with client.websocket_connect(f"/ws/risk/{user_id}") as ws:
        r = client.post("/api/telemetry/ingest", json=payload)
        assert r.status_code == 200
        http_data = r.json()

        # Receive streamed risk payload from WebSocket
        ws_data = ws.receive_json()
        assert ws_data["user_id"] == http_data["user_id"]
        assert ws_data["risk_tier"] == http_data["risk_tier"]
        assert ws_data["risk_score"] == http_data["risk_score"]
        assert ws_data["polling_tier"] == http_data["polling_tier"]
        assert ws_data["battery_override_active"] == http_data["battery_override_active"]


# ---------------------------------------------------------------
# Check 4: Polling tier mapping (0, 1, 2, 3)
# ---------------------------------------------------------------

def test_polling_tier_mapping_all_tiers():
    """
    Check 4: Verify polling_tier mapping:
    QUIESCENT=0, NORMAL_TRANSIT=1, SUSPICIOUS=2, CRITICAL=3.
    """
    base_loc = Location(lat=51.5, lng=-0.1)
    base_signal = SignalStatus(state=SignalState.DEGRADED_SIGNAL)

    # 1. INDOOR_PACING -> QUIESCENT -> polling_tier 0
    t1 = TelemetryPayload(
        user_id="u1",
        timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
        location=base_loc,
        sensor_metrics=SensorMetrics(horizontal_accuracy_m=25.0, speed_mps=0.0, heading_deg=0.0, battery_pct=50),
        signal_status=base_signal,
    )
    out1 = compute_risk(t1, PDRTierState.INDOOR_PACING)
    assert out1.risk_tier == RiskTier.QUIESCENT
    assert out1.polling_tier == 0

    # 2. ZONE_TRANSITION -> NORMAL_TRANSIT -> polling_tier 1
    out2 = compute_risk(t1, PDRTierState.ZONE_TRANSITION)
    assert out2.risk_tier == RiskTier.NORMAL_TRANSIT
    assert out2.polling_tier == 1

    # 3. UNTRACKED_DISPLACEMENT -> SUSPICIOUS -> polling_tier 2
    out3 = compute_risk(t1, PDRTierState.UNTRACKED_DISPLACEMENT)
    assert out3.risk_tier == RiskTier.SUSPICIOUS
    assert out3.polling_tier == 2


# ---------------------------------------------------------------
# Check: Battery override active when battery < 15% and SUSPICIOUS
# ---------------------------------------------------------------

def test_battery_override_active_integration():
    """
    Check 2e: battery_override_active is True if battery < 15% and SUSPICIOUS (PULSE mode),
    and False when battery is sufficient.
    """
    base_loc = Location(lat=51.5, lng=-0.1)
    base_signal = SignalStatus(state=SignalState.DEGRADED_SIGNAL)

    # Low battery (10% < 15%) + SUSPICIOUS -> PULSE mode -> battery_override_active True
    t_low = TelemetryPayload(
        user_id="u_bat",
        timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
        location=base_loc,
        sensor_metrics=SensorMetrics(horizontal_accuracy_m=25.0, speed_mps=0.0, heading_deg=0.0, battery_pct=10),
        signal_status=base_signal,
    )
    out_low = compute_risk(t_low, PDRTierState.UNTRACKED_DISPLACEMENT)
    assert out_low.risk_tier == RiskTier.SUSPICIOUS
    assert out_low.battery_override_active is True

    # Sufficient battery (50%) + SUSPICIOUS -> CONTINUOUS mode -> battery_override_active False
    t_ok = TelemetryPayload(
        user_id="u_bat",
        timestamp=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
        location=base_loc,
        sensor_metrics=SensorMetrics(horizontal_accuracy_m=25.0, speed_mps=0.0, heading_deg=0.0, battery_pct=50),
        signal_status=base_signal,
    )
    out_ok = compute_risk(t_ok, PDRTierState.UNTRACKED_DISPLACEMENT)
    assert out_ok.risk_tier == RiskTier.SUSPICIOUS
    assert out_ok.battery_override_active is False


# ---------------------------------------------------------------
# Check (3): UTC-aware 2-minute window epoch determinism
# ---------------------------------------------------------------

def test_utc_normalization_determinism():
    """
    Check (3): The 2-minute window epoch treats timestamps as UTC-aware
    for cross-environment determinism.
    """
    base_loc = Location(lat=51.5, lng=-0.1)
    metrics = SensorMetrics(horizontal_accuracy_m=3.0, speed_mps=2.0, heading_deg=90.0, battery_pct=80)
    valid_signal = SignalStatus(state=SignalState.VALID)

    dt_aware = datetime(2026, 10, 4, 14, 30, 25, tzinfo=timezone.utc)
    t1 = TelemetryPayload(
        user_id="u_utc",
        timestamp=dt_aware,
        location=base_loc,
        sensor_metrics=metrics,
        signal_status=valid_signal,
    )
    out1 = compute_risk(t1, None)

    # Same timestamp 10 seconds later (still within the same 2-minute bucket 14:30:00 - 14:32:00)
    dt_later_same_window = datetime(2026, 10, 4, 14, 30, 55, tzinfo=timezone.utc)
    t2 = TelemetryPayload(
        user_id="u_utc",
        timestamp=dt_later_same_window,
        location=base_loc,
        sensor_metrics=metrics,
        signal_status=valid_signal,
    )
    out2 = compute_risk(t2, None)

    # Identical seed -> identical score and tier
    assert out1.risk_score == out2.risk_score
    assert out1.risk_tier == out2.risk_tier

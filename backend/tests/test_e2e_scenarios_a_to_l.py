"""
End-to-End Test Suite for Mobility Guardian Scenarios A through L.

Validates the complete context-aware predictive safety and physiological risk fusion pipeline:
- Scenario A: Normal straight-line walk (low operational risk, no alerts).
- Scenario B: Intentional circular garden walk inside permitted safe zone (suppressed, no wandering alert).
- Scenario C: Repetitive pacing with normal physiology (Window 1 holds persistence gate, Window 2 escalates to SUSPICIOUS).
- Scenario D: Repetitive pacing plus unusual personal HR deviation (Window 1 escalates to SUSPICIOUS, Window 2 to CRITICAL).
- Scenario E: Elevated HR during ordinary exercise (exertion-concordant, no wandering alert).
- Scenario F: Elevated HR without anomalous movement (stationary posture, no wandering alert).
- Scenario G: Confirmed home exit (2 outside readings past 15m buffer trigger immediate Tier 3 CRITICAL alert without ML window; deduplicated on subsequent moves).
- Scenario H: GPS uncertainty at boundary (accuracy > 25m held in debounce, no false exit).
- Scenario I: Stale, missing, or invalid HR (explicitly represented as unreliable, never treated as normal zero risk or fabricated).
- Scenario J: Duplicate and out-of-order telemetry (no false persistence increment or duplicate alerts).
- Scenario K: Recovery and safe-context reset (safe area entry resets behavioral escalation; operational risk returns to normal).
- Scenario L: WebSocket interruption and REST reconciliation (client reconnects and re-hydrates active alerts via REST).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.main import app
from app.db import (
    init_db,
    dispose_engine,
    async_session,
    ConsentRecordRow,
    SafeZoneRow,
    GeofenceStateRow,
    AlertRecordRow,
    RiskScoreHistoryRow,
    WearableReadingRow,
    CellDismissalStateRow,
    PDRStateRow,
)
from app.models import (
    AlertRecord,
    AlertStatus,
    RiskTier,
    SensorMetrics,
    SignalState,
    TelemetryPayload,
)
from app.services.inference_service import reset_inference_engine
from app.state_machines.behavioral_escalation import (
    BehavioralEscalationState,
    evaluate_behavioral_escalation,
    get_or_create_behavioral_state,
    reset_behavioral_state,
)


CANONICAL_CONSENT_ROWS = [
    {
        "user_id": "patient_A",
        "guardian_granted": True,
        "guardian_granted_at": datetime.fromisoformat("2026-10-22T08:00:00+00:00"),
        "guardian_id": "guardian_primary",
        "end_user_consent_applicable": False,
        "end_user_granted": None,
        "end_user_granted_at": None,
        "data_retention_ack": True,
    },
    {
        "user_id": "patient_B",
        "guardian_granted": True,
        "guardian_granted_at": datetime.fromisoformat("2026-10-22T08:00:00+00:00"),
        "guardian_id": "guardian_primary",
        "end_user_consent_applicable": False,
        "end_user_granted": None,
        "end_user_granted_at": None,
        "data_retention_ack": True,
    },
    {
        "user_id": "test_ingest_e6557ad7",
        "guardian_granted": True,
        "guardian_granted_at": datetime.fromisoformat("2026-10-27T18:15:06.246456+00:00"),
        "guardian_id": "caregiver_p3",
        "end_user_consent_applicable": False,
        "end_user_granted": None,
        "end_user_granted_at": None,
        "data_retention_ack": False,
    },
]

CANONICAL_SAFE_ZONE_ROWS = [
    {
        "zone_id": "zone_918588c935a6",
        "user_id": "patient_A",
        "name": "Home A",
        "description": None,
        "geometry_type": "circle",
        "center_lat": 39.9,
        "center_lng": 116.4,
        "radius_m": 200.0,
        "polygon_vertices_json": None,
        "is_enabled": True,
        "created_at": datetime.fromisoformat("2026-10-27T18:15:04.315773+00:00"),
        "updated_at": datetime.fromisoformat("2026-10-27T18:15:04.315773+00:00"),
    }
]


async def _clean_and_restore_database():
    """Wipes test rows and restores canonical baseline rows in test_app.db."""
    async with async_session() as session:
        await session.execute(delete(ConsentRecordRow))
        await session.execute(delete(SafeZoneRow))
        await session.execute(delete(GeofenceStateRow))
        await session.execute(delete(AlertRecordRow))
        await session.execute(delete(RiskScoreHistoryRow))
        await session.execute(delete(WearableReadingRow))
        await session.execute(delete(CellDismissalStateRow))
        await session.execute(delete(PDRStateRow))
        await session.commit()

        for c in CANONICAL_CONSENT_ROWS:
            session.add(
                ConsentRecordRow(
                    user_id=c["user_id"],
                    guardian_granted=c["guardian_granted"],
                    guardian_granted_at=c["guardian_granted_at"],
                    guardian_id=c["guardian_id"],
                    end_user_consent_applicable=c["end_user_consent_applicable"],
                    end_user_granted=c["end_user_granted"],
                    end_user_granted_at=c["end_user_granted_at"],
                    data_retention_ack=c["data_retention_ack"],
                )
            )
        for z in CANONICAL_SAFE_ZONE_ROWS:
            session.add(
                SafeZoneRow(
                    zone_id=z["zone_id"],
                    user_id=z["user_id"],
                    name=z["name"],
                    description=z["description"],
                    geometry_type=z["geometry_type"],
                    center_lat=z["center_lat"],
                    center_lng=z["center_lng"],
                    radius_m=z["radius_m"],
                    polygon_vertices_json=z["polygon_vertices_json"],
                    is_enabled=z["is_enabled"],
                    created_at=z["created_at"],
                    updated_at=z["updated_at"],
                )
            )
        await session.commit()


@pytest.fixture(scope="session", autouse=True)
def _session_db_lifecycle():
    """Ensure database is initialized and canonical records are restored on teardown."""
    asyncio.run(init_db())
    asyncio.run(_clean_and_restore_database())
    yield
    asyncio.run(_clean_and_restore_database())
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def _per_test_reset():
    """Reset database tables and in-memory state before and after each test."""
    asyncio.run(_clean_and_restore_database())
    reset_behavioral_state()
    reset_inference_engine()
    yield
    asyncio.run(_clean_and_restore_database())
    reset_behavioral_state()
    reset_inference_engine()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _grant_consent(client: TestClient, user_id: str) -> None:
    consent = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-22T08:00:00Z",
            "guardian_id": "guardian_e2e",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


def _create_circular_zone(client: TestClient, user_id: str, name: str, lat: float, lng: float, radius_m: float) -> str:
    resp = client.post(
        f"/api/safe-zones/{user_id}",
        json={
            "name": name,
            "geometry_type": "circle",
            "center_lat": lat,
            "center_lng": lng,
            "radius_m": radius_m,
            "is_enabled": True,
        },
    )
    assert resp.status_code == 201
    return resp.json()["zone_id"]


def _build_payload(
    user_id: str,
    timestamp: datetime,
    lat: float = 39.9800,
    lng: float = 116.3100,
    speed_mps: float = 1.2,
    horizontal_accuracy_m: float = 4.5,
    activity_type: str = "WALKING",
    battery_pct: int = 85,
    hr_bpm: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "location": {"lat": lat, "lng": lng, "altitude_m": 20.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": horizontal_accuracy_m,
            "speed_mps": speed_mps,
            "heading_deg": 90.0,
            "activity_type": activity_type,
            "battery_pct": battery_pct,
        },
        "signal_status": {"state": "VALID"},
    }
    if hr_bpm is not None:
        payload["wearable_metrics"] = {
            "heart_rate_bpm": hr_bpm,
            "timestamp": timestamp.isoformat(),
            "wear_state": "ON_WRIST",
            "sensor_status": "NORMAL",
        }
    return payload


# ==============================================================================
# SCENARIO A: Normal straight-line walk
# ==============================================================================

def test_scenario_a_normal_straight_line_walk(client: TestClient, monkeypatch):
    """
    Scenario A: Normal straight-line walk produces low operational risk score and NO alerts.
    """
    user_id = "test_scen_a_normal_walk"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.92,
        "window_id": "win_a_0",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": {
            "is_reliable": True,
            "is_corroborating_risk": False,
            "current_hr_bpm": 74,
            "z_score": 0.1,
            "pattern": "WITHIN_NORMAL_BASELINE",
        },
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, speed_mps=1.2, hr_bpm=74)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Risk score remains low and operational tier is normal
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data["risk_score"] < 65.0

    # No behavioral anomaly in top features
    top_features = [f["feature"] for f in data.get("explainability", {}).get("top_features", [])]
    assert "behavioral_trajectory_anomaly" not in top_features
    assert "physiological_arousal_corroboration" not in top_features

    # No alerts persisted in database
    alerts_resp = client.get(f"/api/alerts/user/{user_id}")
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()) == 0

    # Behavioral escalation state remains unescalated
    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is False
    assert state.consecutive_anomalous_windows == 0


# ==============================================================================
# SCENARIO B: Intentional circular walk inside permitted safe zone
# ==============================================================================

def test_scenario_b_intentional_circular_walk_inside_safe_zone(client: TestClient, monkeypatch):
    """
    Scenario B: Repetitive circular walk inside designated safe zone is suppressed from emergency escalation.
    """
    user_id = "test_scen_b_safe_garden"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Monitored person is inside their designated safe zone
    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.89,
        "window_id": "win_b_0",
        "safe_area_state": "INSIDE_SAFE_AREA",
        "heart_rate_evidence": {
            "is_reliable": True,
            "is_corroborating_risk": False,
            "current_hr_bpm": 78,
            "z_score": 0.3,
            "pattern": "WITHIN_NORMAL_BASELINE",
        },
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, speed_mps=0.9, hr_bpm=78)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Suppressed from escalation: tier remains normal transit/quiescent
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data["risk_score"] < 65.0

    # No alerts created
    alerts_resp = client.get(f"/api/alerts/user/{user_id}")
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()) == 0

    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is False
    assert state.consecutive_anomalous_windows == 0


# ==============================================================================
# SCENARIO C: Repetitive pacing with normal physiology
# ==============================================================================

def test_scenario_c_repetitive_pacing_normal_physiology(client: TestClient, monkeypatch):
    """
    Scenario C: Repetitive pacing outside safe area with normal physiology.
    Window 1: Persistence gate holds (no escalation).
    Window 2: Pure kinematic persistence triggers escalation to SUSPICIOUS (score 65.0).
    """
    user_id = "test_scen_c_pacing_normal_hr"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    normal_hr_ev = {
        "is_reliable": True,
        "is_corroborating_risk": False,
        "current_hr_bpm": 72,
        "z_score": 0.2,
        "pattern": "WITHIN_NORMAL_BASELINE",
    }

    # Window 1
    mock_ml_eval_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_c_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": normal_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_w1))

    t1 = datetime(2026, 10, 22, 10, 2, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t1, speed_mps=1.1, hr_bpm=72)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200
    data1 = resp1.json()

    # Window 1 holds: persistence gate not yet satisfied
    assert data1["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data1["risk_score"] < 65.0
    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False

    # Window 2: Distinct window completed
    mock_ml_eval_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_c_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": normal_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_w2))

    t2 = datetime(2026, 10, 22, 10, 4, 0, tzinfo=timezone.utc)
    payload2 = _build_payload(user_id=user_id, timestamp=t2, speed_mps=1.1, hr_bpm=72)
    resp2 = client.post("/api/telemetry/ingest", json=payload2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Window 2 triggers pure kinematic persistence: Tier escalates to SUSPICIOUS
    assert data2["risk_tier"] == "SUSPICIOUS"
    assert data2["risk_score"] == 65.0
    assert state.is_escalated is True
    assert state.is_hr_corroborated is False

    # Exactly 1 active AlertRecord created
    alerts_resp = client.get(f"/api/alerts/user/{user_id}")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    assert len(alerts) == 1
    assert alerts[0]["alert_tier"] == 2
    assert alerts[0]["status"] == "PENDING"
    assert alerts[0]["zone_id"] == "behavioral_escalation"


# ==============================================================================
# SCENARIO D: Repetitive pacing plus unusual personal HR deviation
# ==============================================================================

def test_scenario_d_repetitive_pacing_unusual_heart_rate_deviation(client: TestClient, monkeypatch):
    """
    Scenario D: Pacing plus resting physiological arousal (z >= 2.0 at speed < 0.8 m/s).
    Window 1: Multi-modal corroboration immediately escalates to SUSPICIOUS (score 70.0).
    Window 2: Sustained multi-modal corroboration escalates to CRITICAL (score 85.0).
    """
    user_id = "test_scen_d_pacing_hr_arousal"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    aroused_hr_ev = {
        "is_reliable": True,
        "is_corroborating_risk": True,
        "current_hr_bpm": 108,
        "z_score": 2.45,
        "pattern": "ELEVATED_UNUSUAL_RESTING",
    }

    # Window 1
    mock_ml_eval_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_d_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": aroused_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_w1))

    t1 = datetime(2026, 10, 22, 10, 2, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t1, speed_mps=0.7, hr_bpm=108)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200
    data1 = resp1.json()

    # Window 1 with physiological arousal immediately escalates to SUSPICIOUS
    assert data1["risk_tier"] == "SUSPICIOUS"
    assert data1["risk_score"] == 70.0

    top_feats1 = [f["feature"] for f in data1.get("explainability", {}).get("top_features", [])]
    assert "behavioral_trajectory_anomaly" in top_feats1
    assert "physiological_arousal_corroboration" in top_feats1

    # Window 2: Sustained multi-modal corroboration
    mock_ml_eval_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.92,
        "window_id": "win_d_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": aroused_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_w2))

    t2 = datetime(2026, 10, 22, 10, 4, 0, tzinfo=timezone.utc)
    payload2 = _build_payload(user_id=user_id, timestamp=t2, speed_mps=0.7, hr_bpm=108)
    resp2 = client.post("/api/telemetry/ingest", json=payload2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Window 2 escalates to CRITICAL
    assert data2["risk_tier"] == "CRITICAL"
    assert data2["risk_score"] == 85.0
    assert data2["polling_tier"] == 3


# ==============================================================================
# SCENARIO E: Elevated HR during ordinary exercise
# ==============================================================================

def test_scenario_e_elevated_hr_during_ordinary_exercise(client: TestClient, monkeypatch):
    """
    Scenario E: Elevated heart rate concordant with physical exertion (speed >= 1.2 m/s).
    Physiological engine does NOT corroborate risk and trajectory is NORMAL: zero wandering alert.
    """
    user_id = "test_scen_e_exercise"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    exercise_hr_ev = {
        "is_reliable": True,
        "is_corroborating_risk": False,
        "current_hr_bpm": 135,
        "z_score": 2.2,
        "pattern": "EXERTION_CONCORDANT",
    }
    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.95,
        "window_id": "win_e_0",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": exercise_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, speed_mps=2.4, hr_bpm=135)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Operational score reflects normal walking kinematics; NO wandering escalation
    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is False

    alerts_resp = client.get(f"/api/alerts/user/{user_id}")
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()) == 0


# ==============================================================================
# SCENARIO F: Elevated HR without anomalous movement (stationary)
# ==============================================================================

def test_scenario_f_elevated_hr_without_anomalous_movement_stationary(client: TestClient, monkeypatch):
    """
    Scenario F: Elevated heart rate while stationary (e.g. coffee, emotional call, post-exercise rest).
    Trajectory shows no anomalous kinematic pattern (NORMAL): zero wandering alert.
    """
    user_id = "test_scen_f_stationary_hr"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    elevated_hr_ev = {
        "is_reliable": True,
        "is_corroborating_risk": True,
        "current_hr_bpm": 115,
        "z_score": 2.6,
        "pattern": "ELEVATED_UNUSUAL_RESTING",
    }
    # Trajectory is NORMAL (stationary, no pacing or drift)
    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.95,
        "window_id": "win_f_0",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": elevated_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, speed_mps=0.05, activity_type="STATIONARY", hr_bpm=115)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Stationary state remains QUIESCENT; no wandering alert
    assert data["risk_tier"] == "QUIESCENT"
    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is False

    alerts_resp = client.get(f"/api/alerts/user/{user_id}")
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()) == 0


# ==============================================================================
# SCENARIO G: Confirmed home exit
# ==============================================================================

def test_scenario_g_confirmed_home_exit_breach_and_deduplication(client: TestClient):
    """
    Scenario G: Safe-zone departure evaluated on every fix independently of ML windows.
    Reading 0: Inside Home safe zone (center lat/lng).
    Reading 1: Outside hysteresis buffer -> EXIT_PENDING (unconfirmed).
    Reading 2: 2nd consecutive outside reading -> Confirmed BREACHED! Immediate Tier 3 CRITICAL alert.
    Reading 3: Movement across distinct H3 cells updates active alert in-place (deduplicated).
    """
    user_id = "test_scen_g_home_exit"
    _grant_consent(client, user_id)

    # Safe zone "Home" radius 50.0m centered at (39.9800, 116.3100)
    zone_id = _create_circular_zone(client, user_id, "Home", 39.9800, 116.3100, 50.0)

    # Reading 0: Inside Home
    t0 = datetime(2026, 10, 22, 9, 59, 45, tzinfo=timezone.utc)
    payload0 = _build_payload(user_id=user_id, timestamp=t0, lat=39.9800, lng=116.3100, horizontal_accuracy_m=5.0)
    resp0 = client.post("/api/telemetry/ingest", json=payload0)
    assert resp0.status_code == 200

    # Reading 1: Outside (~120m away, lat 39.9811). Exit pending (debouncing)
    t1 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t1, lat=39.9811, lng=116.3100, horizontal_accuracy_m=5.0)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200

    # No confirmed breach alert yet after 1 reading
    alerts1 = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(alerts1) == 0

    # Reading 2: Second qualifying outside reading (~150m away, lat 39.9814) -> Confirmed BREACHED!
    t2 = datetime(2026, 10, 22, 10, 0, 15, tzinfo=timezone.utc)
    payload2 = _build_payload(user_id=user_id, timestamp=t2, lat=39.9814, lng=116.3100, horizontal_accuracy_m=5.0)
    resp2 = client.post("/api/telemetry/ingest", json=payload2)
    assert resp2.status_code == 200

    # Immediate Tier 3 CRITICAL alert created
    alerts2 = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(alerts2) == 1
    alert = alerts2[0]
    assert alert["alert_tier"] == 3
    assert alert["status"] == "PENDING"
    assert alert["zone_id"] == zone_id
    alert_id = alert["alert_id"]

    # Reading 3: Person moves further outside into a different H3 cell
    t3 = datetime(2026, 10, 22, 10, 0, 30, tzinfo=timezone.utc)
    payload3 = _build_payload(user_id=user_id, timestamp=t3, lat=39.9830, lng=116.3100, horizontal_accuracy_m=5.0)
    resp3 = client.post("/api/telemetry/ingest", json=payload3)
    assert resp3.status_code == 200

    # Deduplication: Still exactly 1 alert record for the user (updated in-place)
    alerts3 = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(alerts3) == 1
    assert alerts3[0]["alert_id"] == alert_id


# ==============================================================================
# SCENARIO H: GPS uncertainty at boundary
# ==============================================================================

def test_scenario_h_gps_uncertainty_at_boundary(client: TestClient):
    """
    Scenario H: GPS horizontal accuracy worse than 25m threshold holds transitions
    and prevents false boundary exit alarms.
    """
    user_id = "test_scen_h_gps_uncertainty"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9800, 116.3100, 50.0)

    # Point appears outside (lat 39.9815), but horizontal accuracy is poor (35.0m > 25.0m limit)
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, lat=39.9815, lng=116.3100, horizontal_accuracy_m=35.0)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200

    # Zero alerts created due to GPS uncertainty filtering
    alerts = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(alerts) == 0


# ==============================================================================
# SCENARIO I: Stale, missing, or invalid HR
# ==============================================================================

def test_scenario_i_stale_missing_or_invalid_hr(client: TestClient, monkeypatch):
    """
    Scenario I: Missing or stale heart-rate evidence is explicitly represented as unreliable;
    never treated as normal zero risk or fabricated into arousal.
    Window 1 of pacing without corroborated arousal does NOT escalate.
    """
    user_id = "test_scen_i_missing_hr"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    unreliable_hr_ev = {
        "status": "MISSING_DATA",
        "is_reliable": False,
        "is_corroborating_risk": False,
        "current_hr_bpm": None,
        "z_score": None,
    }
    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_i_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": unreliable_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    payload = _build_payload(user_id=user_id, timestamp=t0, speed_mps=1.1, hr_bpm=None)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Persistence gate holds: Window 1 uncorroborated pacing does NOT escalate
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is False
    assert state.is_hr_corroborated is False


# ==============================================================================
# SCENARIO J: Duplicate and out-of-order telemetry
# ==============================================================================

def test_scenario_j_duplicate_and_out_of_order_telemetry(client: TestClient, monkeypatch):
    """
    Scenario J: Duplicate and out-of-order telemetry fixes are safely ignored
    without false persistence increments or duplicate alert triggers.
    """
    user_id = "test_scen_j_reordering"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_j_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": {"is_corroborating_risk": False},
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t_eval = datetime(2026, 10, 22, 10, 2, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t_eval, speed_mps=1.1)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200

    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1

    # Out-of-order fix: timestamp is in the past (10:01:00 < 10:02:00)
    t_past = datetime(2026, 10, 22, 10, 1, 0, tzinfo=timezone.utc)
    payload_past = _build_payload(user_id=user_id, timestamp=t_past, speed_mps=1.1)
    resp_past = client.post("/api/telemetry/ingest", json=payload_past)
    assert resp_past.status_code == 200

    # Counter did NOT increment
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False

    # Duplicate fix: identical timestamp
    resp_dup = client.post("/api/telemetry/ingest", json=payload1)
    assert resp_dup.status_code == 200
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False


# ==============================================================================
# SCENARIO K: Recovery and safe-context reset
# ==============================================================================

def test_scenario_k_recovery_and_safe_context_reset(client: TestClient, monkeypatch):
    """
    Scenario K: Entering a designated safe area resets behavioral escalation state,
    returning operational risk to normal.
    """
    user_id = "test_scen_k_recovery"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    normal_hr_ev = {"is_corroborating_risk": False}

    # Step 1: Reach escalated state (2 windows of pacing)
    w1_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_k_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": normal_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=w1_eval))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id, datetime(2026, 10, 22, 10, 2, 0, tzinfo=timezone.utc)))

    w2_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_k_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": normal_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=w2_eval))
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id, datetime(2026, 10, 22, 10, 4, 0, tzinfo=timezone.utc)))
    assert resp2.json()["risk_tier"] == "SUSPICIOUS"

    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is True

    # Step 2: Patient returns inside designated safe area
    recovery_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.90,
        "window_id": "win_k_3",
        "safe_area_state": "INSIDE_SAFE_AREA",
        "heart_rate_evidence": normal_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=recovery_eval))
    resp_rec = client.post("/api/telemetry/ingest", json=_build_payload(user_id, datetime(2026, 10, 22, 10, 6, 0, tzinfo=timezone.utc)))
    assert resp_rec.status_code == 200
    data_rec = resp_rec.json()

    # Behavioral escalation is immediately reset
    assert state.is_escalated is False
    assert state.consecutive_anomalous_windows == 0
    assert data_rec["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")


# ==============================================================================
# SCENARIO L: WebSocket interruption and REST reconciliation
# ==============================================================================

def test_scenario_l_websocket_interruption_and_rest_reconciliation(client: TestClient, monkeypatch):
    """
    Scenario L: Following WebSocket interruption, caregiver dashboard restores
    active alert state through the REST endpoint /api/alerts/user/{user_id}.
    """
    user_id = "test_scen_l_rest_reconcile"
    _grant_consent(client, user_id)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Trigger behavioral escalation alert
    aroused_hr_ev = {
        "is_reliable": True,
        "is_corroborating_risk": True,
        "current_hr_bpm": 110,
        "z_score": 2.5,
        "pattern": "ELEVATED_UNUSUAL_RESTING",
    }
    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_l_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "heart_rate_evidence": aroused_hr_ev,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    client.post("/api/telemetry/ingest", json=_build_payload(user_id, t0, speed_mps=0.6, hr_bpm=110))

    # Caregiver client simulates reconnection / page hydration by calling REST endpoints:
    # 1. Scoped user alert query
    resp_user_alerts = client.get(f"/api/alerts/user/{user_id}?status=PENDING")
    assert resp_user_alerts.status_code == 200
    user_alerts = resp_user_alerts.json()
    assert len(user_alerts) == 1
    alert = user_alerts[0]
    assert alert["user_id"] == user_id
    assert alert["status"] == "PENDING"
    assert alert["alert_tier"] == 2
    assert alert["zone_id"] == "behavioral_escalation"

    # 2. General alerts query filtered by user_id
    resp_query_alerts = client.get(f"/api/alerts?user_id={user_id}&status=PENDING")
    assert resp_query_alerts.status_code == 200
    query_alerts = resp_query_alerts.json()
    assert len(query_alerts) == 1
    assert query_alerts[0]["alert_id"] == alert["alert_id"]

    # 3. Direct alert lookup by alert_id
    alert_id = alert["alert_id"]
    resp_single_alert = client.get(f"/api/alerts/{alert_id}")
    assert resp_single_alert.status_code == 200
    single_alert = resp_single_alert.json()
    assert single_alert["alert_id"] == alert_id
    assert single_alert["status"] == "PENDING"

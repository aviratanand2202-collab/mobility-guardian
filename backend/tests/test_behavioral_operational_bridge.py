"""
Comprehensive regression tests for the Behavioral-to-Operational Risk Bridge.

Validates the 12 key safety requirements:
1. Normal walking does not trigger behavioral escalation without sufficient evidence.
2. One anomalous window does not prematurely escalate (persistence gate).
3. Sustained anomalous behavior across distinct windows triggers operational escalation when enabled.
4. Repeated evaluation of the same window cannot satisfy the persistence requirement.
5. Normal behavior following anomalous behavior resets or clears persistence appropriately.
6. Insufficient evidence and low-confidence classifications do not escalate.
7. Missing ML output, disabled inference, timeout, and inference errors preserve safe baseline behavior.
8. Degraded GPS follows the existing fallback path and resets behavioral state.
9. Safe-area behavior follows the actual configured policy (suppression inside safe area).
10. Existing high-risk operational events (e.g. vehicular speed) still produce expected alerts and are never downgraded.
11. Alert deduplication, dismissal handling, and reconnection recovery remain correct for behaviorally triggered alerts.
12. Default-off configuration ensures zero regression when flags are disabled.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, patch


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.main import app
from app.config import (
    ENABLE_BEHAVIORAL_RISK_ESCALATION,
    is_behavioral_risk_escalation_enabled,
    is_ml_inference_enabled,
)
from app.db import (
    init_db,
    dispose_engine,
    async_session,
    ConsentRecordRow,
    CellDismissalStateRow,
    PDRStateRow,
    AlertRecordRow,
    RiskScoreHistoryRow,
)
from app.models import (
    AlertStatus,
    Location,
    RiskTier,
    SensorMetrics,
    SignalState,
    SignalStatus,
    TelemetryPayload,
)
from app.services.geospatial import location_to_h3_cell
from app.services.inference_service import reset_inference_engine
from app.state_machines.behavioral_escalation import (
    BehavioralEscalationState,
    evaluate_behavioral_escalation,
    get_or_create_behavioral_state,
    reset_behavioral_state,
)


@pytest.fixture(scope="session", autouse=True)
def _db():
    """Initialize test tables once for the test session and clean up on exit."""
    asyncio.run(init_db())
    yield
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def _clean_tables_and_state():
    """Clean all tables and reset in-memory state before each test."""
    async def _clean():
        async with async_session() as session:
            await session.execute(delete(ConsentRecordRow))
            await session.execute(delete(CellDismissalStateRow))
            await session.execute(delete(PDRStateRow))
            await session.execute(delete(AlertRecordRow))
            await session.execute(delete(RiskScoreHistoryRow))
            await session.commit()
    asyncio.run(_clean())
    reset_behavioral_state()
    reset_inference_engine()
    yield
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
            "granted_at": "2026-10-19T10:00:00Z",
            "guardian_id": "guardian_bridge_test",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


def _build_payload(
    user_id: str,
    timestamp: datetime,
    lat: float = 39.9800,
    lng: float = 116.3100,
    speed_mps: float = 1.2,
    state: str = "VALID",
    net_displacement_m: float = 0.0,
    battery_pct: int = 85,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "location": {"lat": lat, "lng": lng, "altitude_m": 20.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": speed_mps,
            "heading_deg": 90.0,
            "activity_type": "WALKING",
            "battery_pct": battery_pct,
        },
        "signal_status": {
            "state": state,
            "degraded_since": timestamp.isoformat() if state == "DEGRADED_SIGNAL" else None,
        },
    }
    if state == "DEGRADED_SIGNAL":
        payload["imu_metrics"] = {
            "step_count_since_last_gps": 50,
            "net_displacement_m": net_displacement_m,
            "pdr_tier_state": "ZONE_TRANSITION",
        }
    else:
        payload["imu_metrics"] = None
    return payload


# ==============================================================================
# 1. DEFAULT-OFF & FEATURE FLAG TESTS
# ==============================================================================

def test_default_config_is_disabled():
    """Verify that ENABLE_BEHAVIORAL_RISK_ESCALATION is disabled by default."""
    assert is_behavioral_risk_escalation_enabled() is False


def test_disabled_flag_preserves_baseline_when_ml_reports_pacing(client: TestClient, monkeypatch):
    """When behavioral escalation is disabled, ML pacing classification does NOT alter operational score."""
    user_id = "u_flag_disabled"
    _grant_consent(client, user_id)

    # Enable ML inference engine, but keep behavioral escalation flag disabled
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: False)

    mock_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.95,
        "window_id": "win_pacing_0",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval))

    payload = _build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc), speed_mps=1.2)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Score remains bounded by speed (1.2 * 12 = 14.4 + pseudo_rand * 40 <= 54.4)
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data["risk_score"] < 65.0


# ==============================================================================
# 2. PERSISTENCE GATE: WINDOW 1 DOES NOT ESCALATE; WINDOW 2 ESCALATES
# ==============================================================================

def test_single_window_pacing_does_not_escalate(client: TestClient, monkeypatch):
    """Single window of PACING (window 1) increments counter to 1 but does NOT escalate to SUSPICIOUS."""
    user_id = "u_single_win"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_ml_eval_1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_pacing_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_1))

    t1 = datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t1, speed_mps=1.2)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200
    data1 = resp1.json()

    # Must remain normal transit (persistence requirement not yet met)
    assert data1["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data1["risk_score"] < 65.0

    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False


def test_sustained_two_windows_pacing_escalates_and_creates_alert(client: TestClient, monkeypatch):
    """Two consecutive distinct windows of PACING escalate operational tier to SUSPICIOUS and create AlertRecord."""
    user_id = "u_two_win_pacing"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Window 1
    mock_ml_eval_1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_pacing_w0",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_1))

    t1 = datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)
    payload1 = _build_payload(user_id=user_id, timestamp=t1, speed_mps=1.2)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200
    assert resp1.json()["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")

    # Window 2 (distinct window_id 120s later)
    mock_ml_eval_2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.92,
        "window_id": "win_pacing_w1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_2))

    t2 = datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)
    payload2 = _build_payload(user_id=user_id, timestamp=t2, speed_mps=1.2)
    resp2 = client.post("/api/telemetry/ingest", json=payload2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Must be escalated to SUSPICIOUS (Tier 2) with score clamped to >= 65.0
    assert data2["risk_tier"] == "SUSPICIOUS"
    assert data2["risk_score"] >= 65.0
    assert data2["polling_tier"] == 2

    # Explainability reflects behavioral pacing
    assert "explainability" in data2
    top_feat = data2["explainability"]["top_features"][0]
    assert top_feat["feature"] == "behavioral_trajectory_anomaly"
    assert "pacing" in top_feat["human_readable"].lower()

    # Verify PENDING alert created in SQLite database
    async def _get_alerts():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts = asyncio.run(_get_alerts())
    assert len(alerts) == 1
    assert alerts[0].status == "PENDING"
    assert alerts[0].alert_tier == 2


# ==============================================================================
# 3. REPEATED EVALUATION OF SAME WINDOW (SLIDING WINDOW PROTECTION)
# ==============================================================================

def test_repeated_evaluation_of_same_window_does_not_increment_counter(client: TestClient, monkeypatch):
    """Replaying telemetry from the same window_id must NOT increment the consecutive window counter."""
    user_id = "u_same_win"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_ml_eval_same = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_duplicate_id",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_ml_eval_same))

    t1 = datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 19, 12, 2, 2, tzinfo=timezone.utc)

    # First point
    resp1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t1, speed_mps=1.2))
    assert resp1.status_code == 200

    # Second point with identical window_id
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t2, speed_mps=1.2))
    assert resp2.status_code == 200

    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False
    assert resp2.json()["risk_tier"] != "SUSPICIOUS"


# ==============================================================================
# 4. RESET ON CHANGED BEHAVIOR (NORMAL / LAPPING)
# ==============================================================================

def test_normal_behavior_resets_persistence_counter(client: TestClient, monkeypatch):
    """A subsequent window classified as NORMAL resets the consecutive counter and clears escalation."""
    user_id = "u_reset_normal"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Point 1: Window 1 (PACING)
    mock_1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.89,
        "window_id": "win_p_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_1))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))

    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1

    # Point 2: Window 2 (NORMAL straight walk)
    mock_2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.95,
        "window_id": "win_norm_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_2))
    resp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)))

    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert state.consecutive_anomalous_windows == 0
    assert state.is_escalated is False


# ==============================================================================
# 5. INSUFFICIENT EVIDENCE & LOW CONFIDENCE
# ==============================================================================

def test_low_confidence_does_not_escalate(client: TestClient, monkeypatch):
    """Low confidence (< 0.75) does not count as verified anomaly and resets persistence."""
    user_id = "u_low_conf"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_low = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.60,  # Below 0.75 floor
        "window_id": "win_low_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_low))

    resp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))
    assert resp.status_code == 200
    data = resp.json()

    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 0


def test_accumulating_window_preserves_baseline(client: TestClient, monkeypatch):
    """While a window is accumulating (INSUFFICIENT_EVIDENCE), baseline risk is preserved without premature escalation."""
    user_id = "u_accumulating"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_accum = {
        "engine_status": "ACTIVE",
        "evidence_status": "INSUFFICIENT_EVIDENCE",
        "behavioral_indicator": None,
        "behavioral_confidence": None,
        "window_id": None,
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_accum))

    resp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 10, tzinfo=timezone.utc)))
    assert resp.status_code == 200
    assert resp.json()["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")


# ==============================================================================
# 6. MISSING ML OUTPUT / TIMEOUT / ENGINE FAILURE
# ==============================================================================

def test_missing_or_failed_ml_preserves_safe_baseline(client: TestClient, monkeypatch):
    """When ML inference fails or times out, operational pipeline safely preserves mock engine score."""
    user_id = "u_ml_fail"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_fail = {
        "engine_status": "TIMEOUT",
        "evidence_status": "ERROR",
        "error": "ML inference execution exceeded time limit.",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_fail))

    resp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert data["risk_score"] < 65.0


# ==============================================================================
# 7. DEGRADED GPS SIGNAL FALLBACK
# ==============================================================================

def test_degraded_gps_resets_behavioral_counter_and_obeys_pdr(client: TestClient, monkeypatch):
    """When GPS signal is degraded, behavioral escalation resets and PDR hysteresis is authoritative."""
    user_id = "u_degraded_reset"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    state = get_or_create_behavioral_state(user_id)
    state.consecutive_anomalous_windows = 1  # Prior valid window recorded

    # Reading 1: 30m net displacement moves PDR from INDOOR_PACING to ZONE_TRANSITION
    p1 = _build_payload(
        user_id=user_id,
        timestamp=datetime(2026, 10, 19, 12, 3, 0, tzinfo=timezone.utc),
        state="DEGRADED_SIGNAL",
        net_displacement_m=30.0,
    )
    r1 = client.post("/api/telemetry/ingest", json=p1)
    assert r1.status_code == 200
    assert r1.json()["pdr_tier"] == "ZONE_TRANSITION"

    # Behavioral GPS counter was immediately reset by degraded signal
    assert state.consecutive_anomalous_windows == 0

    # Reading 2: 60m net displacement moves PDR from ZONE_TRANSITION to UNTRACKED_DISPLACEMENT
    p2 = _build_payload(
        user_id=user_id,
        timestamp=datetime(2026, 10, 19, 12, 5, 0, tzinfo=timezone.utc),
        state="DEGRADED_SIGNAL",
        net_displacement_m=60.0,
    )
    r2 = client.post("/api/telemetry/ingest", json=p2)
    assert r2.status_code == 200
    d2 = r2.json()

    # PDR produces SUSPICIOUS (75.0) via operational engine
    assert d2["risk_tier"] == "SUSPICIOUS"
    assert d2["risk_score"] == 75.0
    assert d2["pdr_tier"] == "UNTRACKED_DISPLACEMENT"
    assert state.consecutive_anomalous_windows == 0



# ==============================================================================
# 8. SAFE AREA SUPPRESSION
# ==============================================================================

def test_pacing_inside_safe_area_suppresses_escalation(client: TestClient, monkeypatch):
    """Pacing inside a designated safe area (e.g. living room hallway) does not trigger operational alert."""
    user_id = "u_safe_area"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_inside = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.95,
        "window_id": "win_inside_1",
        "safe_area_state": "INSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_inside))

    resp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))
    assert resp.status_code == 200
    data = resp.json()

    assert data["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 0
    assert state.is_escalated is False


# ==============================================================================
# 9. EXISTING HIGH-RISK VEHICULAR SPEED PRESERVATION
# ==============================================================================

def test_high_speed_retains_critical_tier_regardless_of_behavior(client: TestClient, monkeypatch):
    """A vehicular speed (7.0 m/s) baseline CRITICAL score (>= 85) is never downgraded by behavioral evaluation."""
    user_id = "u_speed_dom"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    mock_normal = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "NORMAL",
        "behavioral_confidence": 0.99,
        "window_id": "win_speed_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_normal))

    payload = _build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc), speed_mps=7.0)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["risk_tier"] == "CRITICAL"
    assert data["risk_score"] >= 85.0


# ==============================================================================
# 10. ALERT DEDUPLICATION & DISMISSAL QUARANTINE
# ==============================================================================

def test_behavioral_alert_deduplication_and_dismissal(client: TestClient, monkeypatch):
    """Behaviorally triggered alerts deduplicate in the same H3 cell and respect caregiver dismissal."""
    user_id = "u_alert_dedup"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    state = get_or_create_behavioral_state(user_id)
    state.consecutive_anomalous_windows = 2
    state.is_escalated = True
    state.escalation_reason = "Sustained pacing detected."

    mock_pacing = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.91,
        "window_id": "win_dedup_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_pacing))

    t1 = datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 19, 12, 4, 30, tzinfo=timezone.utc)

    # Point 1 creates alert
    resp1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t1))
    assert resp1.status_code == 200

    # Point 2 in same cell updates existing alert timestamp without creating duplicate
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t2))
    assert resp2.status_code == 200

    async def _get_alerts():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts = asyncio.run(_get_alerts())
    assert len(alerts) == 1, "Must update existing alert row rather than duplicate"
    alert_id = alerts[0].alert_id

    # Caregiver dismisses alert
    resp_dismiss = client.post(f"/api/alerts/{alert_id}/dismiss")
    assert resp_dismiss.status_code == 200
    assert resp_dismiss.json()["status"] == "dismissed"

    alerts_after = asyncio.run(_get_alerts())
    assert alerts_after[0].status == "DISMISSED_SAFE"


# ==============================================================================
# 11. STALE TELEMETRY TEMPORAL GAP RESETS PERSISTENCE
# ==============================================================================

def test_stale_telemetry_gap_resets_counter(client: TestClient, monkeypatch):
    """A telemetry gap > 180 seconds resets consecutive anomalous windows."""
    user_id = "u_stale_gap"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    state = get_or_create_behavioral_state(user_id)
    state.consecutive_anomalous_windows = 1
    state.last_observation_time = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)

    # Telemetry point arrives 4 minutes later (240s > 180s)
    mock_pacing = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_gap_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_pacing))

    t_late = datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t_late))

    # Gap reset the prior count, then this window counted as window 1 (not 2)
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False


# ==============================================================================
# 12. RANDOM DRIFT ANOMALY ESCALATION
# ==============================================================================

def test_random_drift_two_windows_escalates(client: TestClient, monkeypatch):
    """Sustained RANDOM_DRIFT (erratic meandering) across 2 windows triggers operational escalation."""
    user_id = "u_random_drift"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Window 1
    mock_rd1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "RANDOM_DRIFT",
        "behavioral_confidence": 0.85,
        "window_id": "win_rd_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_rd1))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))

    # Window 2
    mock_rd2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "RANDOM_DRIFT",
        "behavioral_confidence": 0.87,
        "window_id": "win_rd_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_rd2))
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)))

    assert resp2.status_code == 200
    data = resp2.json()
    assert data["risk_tier"] == "SUSPICIOUS"
    assert data["risk_score"] >= 65.0
    assert "random drift" in data["explainability"]["top_features"][0]["human_readable"].lower()


# ==============================================================================
# 13. LIVE STREAMING PACING TRAJECTORY INTEGRATION WITH REAL ML INFERENCE ENGINE
# ==============================================================================

def test_live_simulator_pacing_trajectory_two_windows_escalates_operationally(client: TestClient, monkeypatch):
    """End-to-end integration: feed 122 streaming pacing points (two 120s windows) through the real ML engine & API.

    Verifies:
    1. During Window 1 (first 61 points, 0-120s), operational risk remains NORMAL_TRANSIT (<= 54.4) and no alert exists.
    2. During Window 2 (points 62-122, 122-244s), once the second distinct window completes,
       operational risk escalates to SUSPICIOUS (>= 65.0) and a Tier 2 PENDING alert is created in SQLite.
    """
    from backend.scripts.trajectory_simulator import MovementGenerator, lat_lng_offset

    user_id = "u_live_sim_pacing"
    _grant_consent(client, user_id)

    # Enable both ML inference and behavioral risk escalation
    monkeypatch.setattr("app.config.is_ml_inference_enabled", lambda: True)
    monkeypatch.setattr("app.services.inference_service.is_ml_inference_enabled", lambda: True)
    monkeypatch.setattr("app.config.is_behavioral_risk_escalation_enabled", lambda: True)
    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)


    gen = MovementGenerator(pattern="pacing", num_points=122, interval_seconds=2.0, speed_mps=1.2)
    pts = gen.generate()
    anchor_lat, anchor_lng = 39.9847, 116.3184
    start_ts = 1700000000.0

    last_resp_w1 = None
    last_resp_w2 = None

    for i, (xm, ym, spd, heading) in enumerate(pts):
        lat, lng = lat_lng_offset(anchor_lat, anchor_lng, xm, ym)
        ts = datetime.fromtimestamp(start_ts + i * 2.0, tz=timezone.utc)
        payload = {
            "user_id": user_id,
            "timestamp": ts.isoformat(),
            "location": {"lat": round(lat, 7), "lng": round(lng, 7), "altitude_m": 15.0},
            "sensor_metrics": {
                "horizontal_accuracy_m": 5.0,
                "speed_mps": spd,
                "heading_deg": heading,
                "activity_type": "WALKING",
                "battery_pct": 80,
            },
            "signal_status": {"state": "VALID"},
        }
        resp = client.post("/api/telemetry/ingest", json=payload)
        assert resp.status_code == 200

        if i == 60:  # End of Window 1 (120s elapsed)
            last_resp_w1 = resp.json()
        elif i == 121:  # End of Window 2 (242s elapsed)
            last_resp_w2 = resp.json()

    # Window 1 verification: persistence gate held; baseline speed heuristic governs
    assert last_resp_w1 is not None
    assert last_resp_w1["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert last_resp_w1["risk_score"] < 65.0

    # Window 2 verification: persistence gate satisfied; operational risk escalated to SUSPICIOUS
    assert last_resp_w2 is not None
    assert last_resp_w2["risk_tier"] == "SUSPICIOUS"
    assert last_resp_w2["risk_score"] >= 65.0
    assert last_resp_w2["polling_tier"] == 2

    # Check that AlertRecord was persisted in SQLite
    async def _get_alerts():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts = asyncio.run(_get_alerts())
    assert len(alerts) == 1
    assert alerts[0].status == "PENDING"
    assert alerts[0].alert_tier == 2


# ==============================================================================
# 13. MULTI-USER CONCURRENT ISOLATION
# ==============================================================================

def test_multi_user_concurrent_isolation(client: TestClient, monkeypatch):
    """Telemetry from distinct users must remain strictly isolated without cross-contaminating escalation state."""
    user_a = "u_concurrent_a"
    user_b = "u_concurrent_b"
    _grant_consent(client, user_a)
    _grant_consent(client, user_b)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Window 1 for User A
    mock_a1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_a_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_a1))
    t1 = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
    resp_a1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_a, timestamp=t1, speed_mps=1.0))
    assert resp_a1.status_code == 200

    # Window 1 for User B
    mock_b1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_b_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_b1))
    resp_b1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_b, timestamp=t1, speed_mps=1.0))
    assert resp_b1.status_code == 200

    state_a = get_or_create_behavioral_state(user_a)
    state_b = get_or_create_behavioral_state(user_b)
    assert state_a.consecutive_anomalous_windows == 1
    assert state_b.consecutive_anomalous_windows == 1
    assert not state_a.is_escalated
    assert not state_b.is_escalated

    # Window 2 for User A (User A satisfies persistence gate)
    mock_a2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.91,
        "window_id": "win_a_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_a2))
    t2 = datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)
    resp_a2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_a, timestamp=t2, speed_mps=1.0))
    assert resp_a2.status_code == 200
    assert resp_a2.json()["risk_tier"] == "SUSPICIOUS"

    # User A is escalated, but User B remains NOT escalated
    assert state_a.is_escalated is True
    assert state_a.consecutive_anomalous_windows == 2
    assert state_b.is_escalated is False
    assert state_b.consecutive_anomalous_windows == 1


# ==============================================================================
# 14. SAFE-AREA TRANSITIONS WIPE PRIOR PERSISTENCE CREDIT
# ==============================================================================

def test_safe_area_transition_wipes_prior_credit(client: TestClient, monkeypatch):
    """Entering a safe area resets persistence counter so subsequent pacing requires full re-accumulation."""
    user_id = "u_safe_wipe"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # 1. Anomalous window outside safe area
    mock_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.92,
        "window_id": "win_w1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w1))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)))
    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1

    # 2. Window inside safe area (transition)
    mock_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.95,
        "window_id": "win_w2",
        "safe_area_state": "INSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w2))
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc)))
    assert resp2.status_code == 200
    assert resp2.json()["risk_tier"] != "SUSPICIOUS"
    assert state.consecutive_anomalous_windows == 0

    # 3. Subsequent anomalous window outside safe area: must NOT trigger escalation prematurely
    mock_w3 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.91,
        "window_id": "win_w3",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w3))
    resp3 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 4, 0, tzinfo=timezone.utc)))
    assert resp3.status_code == 200
    assert resp3.json()["risk_tier"] != "SUSPICIOUS"
    assert state.consecutive_anomalous_windows == 1
    assert state.is_escalated is False


# ==============================================================================
# 15. POLLING INSTRUCTION CONSISTENCY AT LOW BATTERY (< 15%) AFTER ESCALATION
# ==============================================================================

def test_polling_instruction_consistency_low_battery_after_escalation(client: TestClient, monkeypatch):
    """Under 15% battery, behavioral escalation to SUSPICIOUS forces PULSE polling mode and battery_override_active."""
    user_id = "u_low_bat_pulse"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Window 1 (PACING, low battery 10%)
    mock_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_bat_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w1))
    payload1 = _build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc), battery_pct=10)
    resp1 = client.post("/api/telemetry/ingest", json=payload1)
    assert resp1.status_code == 200
    # Before escalation: baseline is QUIESCENT, continuous polling
    assert resp1.json()["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert resp1.json()["battery_override_active"] is False
    assert resp1.json()["polling_instruction"]["mode"] == "CONTINUOUS"

    # Window 2 (PACING, low battery 10% -> triggers escalation to SUSPICIOUS)
    mock_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.93,
        "window_id": "win_bat_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w2))
    payload2 = _build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc), battery_pct=10)
    resp2 = client.post("/api/telemetry/ingest", json=payload2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Must be escalated to SUSPICIOUS
    assert data2["risk_tier"] == "SUSPICIOUS"
    assert data2["risk_score"] >= 65.0
    assert data2["polling_tier"] == 2

    # Polling instruction must be PULSE per battery_pulse state machine (§2.6)
    assert data2["battery_override_active"] is True
    inst = data2["polling_instruction"]
    assert inst["mode"] == "PULSE"
    assert inst["burst_seconds"] == 5
    assert inst["burst_hz"] == 1
    assert inst["sleep_seconds"] == 175


# ==============================================================================
# 16. OUT-OF-ORDER TELEMETRY TIMESTAMPS
# ==============================================================================

def test_out_of_order_timestamps_preserve_stale_gap_detection(client: TestClient, monkeypatch):
    """Out-of-order telemetry timestamps must not move last_observation_time backwards or break stale gap detection."""
    user_id = "u_ooo_ts"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # 1. Packet at 12:00:00 (Window 1 evaluated)
    mock_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.88,
        "window_id": "win_ooo_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w1))
    t1 = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
    resp1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t1))
    assert resp1.status_code == 200
    state = get_or_create_behavioral_state(user_id)
    assert state.consecutive_anomalous_windows == 1
    assert state.last_observation_time == t1

    # 2. Out-of-order packet arriving with older timestamp 11:59:50 (-10s)
    t_ooo = datetime(2026, 10, 19, 11, 59, 50, tzinfo=timezone.utc)
    mock_accum = {
        "engine_status": "ACTIVE",
        "evidence_status": "INSUFFICIENT_EVIDENCE",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_accum))
    resp_ooo = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t_ooo))
    assert resp_ooo.status_code == 200

    # last_observation_time high-water mark MUST NOT have moved backwards
    assert state.last_observation_time == t1
    assert state.consecutive_anomalous_windows == 1

    # 3. In-order packet at 12:02:40 (160s after 12:00:00, within 180s gap)
    t3 = datetime(2026, 10, 19, 12, 2, 40, tzinfo=timezone.utc)
    resp3 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t3))
    assert resp3.status_code == 200
    # Did not falsely trigger stale gap reset
    assert state.consecutive_anomalous_windows == 1
    assert state.last_observation_time == t3

    # 4. Telemetry gap exceeding 180s (12:06:00 is 200s after 12:02:40)
    t4 = datetime(2026, 10, 19, 12, 6, 0, tzinfo=timezone.utc)
    resp4 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t4))
    assert resp4.status_code == 200
    # Correctly triggers stale-gap reset
    assert state.consecutive_anomalous_windows == 0
    assert state.last_observation_time == t4


# ==============================================================================
# 17. MISSING OR ERRORED ML OUTPUT AFTER EXISTING ESCALATION
# ==============================================================================

def test_missing_or_errored_ml_output_after_escalation_expires_safely(client: TestClient, monkeypatch):
    """Stale evidence due to missing or errored ML cannot silently sustain escalation indefinitely, while preserving alerts and independent operational risk."""
    user_id = "u_ml_stale_expire"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Establish escalation across two windows (12:00:00 and 12:02:00)
    mock_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_exp_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w1))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc), speed_mps=1.0))

    mock_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.92,
        "window_id": "win_exp_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w2))
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc), speed_mps=1.0))
    assert resp2.json()["risk_tier"] == "SUSPICIOUS"

    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is True

    # 1. Transient error within tolerance window (10s later): does NOT immediately drop alert
    mock_transient_error = {
        "engine_status": "TIMEOUT",
        "evidence_status": "ERROR",
        "error": "Temporary worker timeout",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_transient_error))
    resp_transient = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 10, tzinfo=timezone.utc), speed_mps=1.0))
    assert resp_transient.status_code == 200
    assert state.is_escalated is True  # Preserved against transient 1-reading drop

    # 2. Prolonged missing ML output (time elapsed > 180s from last anomalous confirmation)
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=None))
    t_expired = datetime(2026, 10, 19, 12, 5, 30, tzinfo=timezone.utc)  # 210s after 12:02:00
    resp_exp = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=t_expired, speed_mps=1.0))
    assert resp_exp.status_code == 200
    data_exp = resp_exp.json()

    # Escalation has expired; operational baseline governs (walking -> QUIESCENT / NORMAL_TRANSIT)
    assert data_exp["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert state.is_escalated is False
    assert state.consecutive_anomalous_windows == 0

    # 3. Verify SQLite alert was NOT deleted or revoked
    async def _get_alerts():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts = asyncio.run(_get_alerts())
    assert len(alerts) == 1
    assert alerts[0].status == "PENDING"
    assert alerts[0].alert_tier == 2

    # 4. Verify independently justified operational risk (e.g. vehicular speed 6.0 m/s) is NEVER downgraded
    p_high_speed = _build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 5, 35, tzinfo=timezone.utc), speed_mps=6.0)
    resp_high_speed = client.post("/api/telemetry/ingest", json=p_high_speed)
    assert resp_high_speed.status_code == 200
    data_high_speed = resp_high_speed.json()
    assert data_high_speed["risk_tier"] == "CRITICAL"
    assert data_high_speed["risk_score"] >= 85.0


# ==============================================================================
# 18. CONSECUTIVE ML ERRORS RESET ESCALATION
# ==============================================================================

def test_consecutive_ml_errors_trigger_deescalation(client: TestClient, monkeypatch):
    """Multiple consecutive ML evaluation failures (>= 3) reset behavioral escalation."""
    user_id = "u_ml_consecutive_errors"
    _grant_consent(client, user_id)

    monkeypatch.setattr("app.routers.telemetry.is_behavioral_risk_escalation_enabled", lambda: True)

    # Establish escalation across two windows
    mock_w1 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.90,
        "window_id": "win_err_1",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w1))
    client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc), speed_mps=1.0))

    mock_w2 = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "behavioral_indicator": "PACING",
        "behavioral_confidence": 0.92,
        "window_id": "win_err_2",
        "safe_area_state": "OUTSIDE_SAFE_AREA",
    }
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_w2))
    resp2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 0, tzinfo=timezone.utc), speed_mps=1.0))
    assert resp2.json()["risk_tier"] == "SUSPICIOUS"

    state = get_or_create_behavioral_state(user_id)
    assert state.is_escalated is True

    # Error reading 1: preserved
    mock_err = {"engine_status": "FAILED", "evidence_status": "ERROR", "error": "Internal inference error"}
    monkeypatch.setattr("app.routers.telemetry.evaluate_shadow_inference", AsyncMock(return_value=mock_err))
    r_err1 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 5, tzinfo=timezone.utc), speed_mps=1.0))
    assert r_err1.status_code == 200
    assert state.is_escalated is True
    assert state.consecutive_ml_errors == 1

    # Error reading 2: preserved
    r_err2 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 10, tzinfo=timezone.utc), speed_mps=1.0))
    assert r_err2.status_code == 200
    assert state.is_escalated is True
    assert state.consecutive_ml_errors == 2

    # Error reading 3: reaches threshold (3), resets escalation
    r_err3 = client.post("/api/telemetry/ingest", json=_build_payload(user_id=user_id, timestamp=datetime(2026, 10, 19, 12, 2, 15, tzinfo=timezone.utc), speed_mps=1.0))
    assert r_err3.status_code == 200
    assert r_err3.json()["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT")
    assert state.is_escalated is False
    assert state.consecutive_anomalous_windows == 0

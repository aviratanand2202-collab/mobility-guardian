"""
Phase 2 Test Suite: Safe-Zone Boundary Transition State Machine & Alert Lifecycle Integration.

Verifies:
1. Confirmed boundary exit: Triggers transition from INSIDE -> EXIT_PENDING -> BREACHED.
2. GPS Jitter suppression: Hysteresis buffer and debouncing prevent false alarms.
3. Alert persistence: Confirmed breach creates a single persistent AlertRecord (status=PENDING, alert_tier=3).
4. Alert deduplication: Moving across multiple H3 cells during an active breach updates alert location without duplicate records.
5. Confirmed re-entry: Resolves active breach with status=AUTO_RESOLVED and notifies WebSocket clients.
6. Multi-zone orchestration: Entering Company safely resolves transit breach from Home.
7. Missing zones reporting: Explicitly reports NO_ZONES_CONFIGURED without fabricating boundaries.
8. Insufficient GPS evidence: Explicitly reports INSUFFICIENT_GPS on degraded accuracy without state churn.
9. Dual-consent enforcement: Unconsented telemetry is blocked at ingestion (403).
10. Wearable independence: Normal or missing heart rate never suppresses an urgent geofence breach.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.main import app
from app.db import (
    init_db,
    dispose_engine,
    async_session,
    ConsentRecordRow,
    SafeZoneRow,
    GeofenceStateRow,
    AlertRecordRow,
    WearableReadingRow,
)
from app.models import AlertStatus, BoundaryState, GeofenceEvaluationStatus
from app.services.geospatial import location_to_h3_cell
from app.state_machines.geofence_boundary import (
    evaluate_patient_geofence_boundaries,
    GeofenceBoundaryConfig,
)


@pytest.fixture(scope="session", autouse=True)
def _db():
    asyncio.run(init_db())
    yield
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def _clean_tables():
    async def _clean():
        async with async_session() as session:
            await session.execute(delete(ConsentRecordRow))
            await session.execute(delete(SafeZoneRow))
            await session.execute(delete(GeofenceStateRow))
            await session.execute(delete(AlertRecordRow))
            await session.execute(delete(WearableReadingRow))
            await session.commit()

    asyncio.run(_clean())


def _grant_consent(client: TestClient, user_id: str):
    consent = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-22T08:00:00Z",
            "guardian_id": "guardian_phase2",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    resp = client.post("/api/consent/record", json=consent)
    assert resp.status_code == 200


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


def _telemetry_payload(user_id: str, lat: float, lng: float, accuracy_m: float = 5.0, ts: str = "2026-10-22T10:00:00Z", hr: int | None = None) -> dict:
    payload = {
        "user_id": user_id,
        "timestamp": ts,
        "location": {"lat": lat, "lng": lng, "altitude_m": 45.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": accuracy_m,
            "speed_mps": 1.2,
            "heading_deg": 90.0,
            "battery_pct": 90,
        },
        "signal_status": {"state": "VALID"},
    }
    if hr is not None:
        payload["wearable_metrics"] = {
            "heart_rate_bpm": hr,
            "timestamp": ts,
            "wear_state": "ON_WRIST",
        }
    return payload


# ==============================================================================
# Tests
# ==============================================================================

def test_missing_zones_reported_explicitly():
    """Requirement 9: If a patient has no configured safe zone, report that state explicitly without inventing boundaries."""
    client = TestClient(app)
    user_id = "patient_no_zones"
    _grant_consent(client, user_id)

    payload = _telemetry_payload(user_id, 39.9042, 116.4074)
    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200

    # Direct evaluation confirms NO_ZONES_CONFIGURED
    eval_result = asyncio.run(
        evaluate_patient_geofence_boundaries(
            user_id=user_id,
            lat=39.9042,
            lng=116.4074,
            accuracy_m=5.0,
            timestamp=datetime.now(timezone.utc),
        )
    )
    assert eval_result.status == GeofenceEvaluationStatus.NO_ZONES_CONFIGURED
    assert "No enabled safe zones configured" in eval_result.message


def test_insufficient_gps_accuracy_holds_state():
    """Requirement 9: Degraded GPS accuracy (>25m) reports INSUFFICIENT_GPS and prevents spurious transitions."""
    client = TestClient(app)
    user_id = "patient_bad_gps"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9042, 116.4074, 200.0)

    # Point far outside, but with coarse accuracy (50m > 25m threshold)
    eval_result = asyncio.run(
        evaluate_patient_geofence_boundaries(
            user_id=user_id,
            lat=39.9500,  # Far outside
            lng=116.4500,
            accuracy_m=50.0,
            timestamp=datetime.now(timezone.utc),
        )
    )
    assert eval_result.status == GeofenceEvaluationStatus.INSUFFICIENT_GPS
    assert len(eval_result.alerts_triggered) == 0


def test_gps_jitter_near_boundary_does_not_trigger_false_breach():
    """Requirement 3: Location jitter right outside boundary edge is debounced without creating false alarms."""
    client = TestClient(app)
    user_id = "patient_jitter"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    # Point 1: Comfortably inside Home center
    resp1 = client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, accuracy_m=4.0, ts="2026-10-22T10:00:00Z"))
    assert resp1.status_code == 200

    # Point 2: Jitter jump right on the boundary edge (distance ~205m, within hysteresis buffer)
    # 0.00185 deg lat ~ 205m north
    resp2 = client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.90185, 116.4000, accuracy_m=10.0, ts="2026-10-22T10:00:15Z"))
    assert resp2.status_code == 200

    # Point 3: Next point returns inside
    resp3 = client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, accuracy_m=4.0, ts="2026-10-22T10:00:30Z"))
    assert resp3.status_code == 200

    # Verify: ZERO breach alerts were created in DB
    async def _check():
        async with async_session() as session:
            alerts = (await session.execute(select(AlertRecordRow).where(AlertRecordRow.user_id == user_id))).scalars().all()
            assert len(alerts) == 0

    asyncio.run(_check())


def test_confirmed_exit_creates_persistent_breach_alert():
    """Requirement 4: When actual locations confirm exit, create one persistent breach alert with status=PENDING."""
    client = TestClient(app)
    user_id = "patient_breach"
    _grant_consent(client, user_id)
    zone_id = _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    # Reading 1: Inside Home
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:00:00Z"))

    # Reading 2: Outside Home (~350m north) -> EXIT_PENDING
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9032, 116.4000, ts="2026-10-22T10:01:00Z"))

    # Reading 3: Confirmed outside (~500m north) -> BREACHED!
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9045, 116.4000, ts="2026-10-22T10:02:00Z"))

    # Verify: Exactly 1 persistent breach alert in DB
    async def _check():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.status == AlertStatus.PENDING.value
            assert alert.zone_id == zone_id
            assert alert.alert_tier == 3

    asyncio.run(_check())


def test_repeated_points_during_breach_deduplicate_across_h3_cells():
    """Requirement 5: Movement across different H3 cells during an active breach updates the existing alert without creating duplicates."""
    client = TestClient(app)
    user_id = "patient_dedup"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    # 1. Establish breach
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:00:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9050, 116.4000, ts="2026-10-22T10:01:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9060, 116.4000, ts="2026-10-22T10:02:00Z"))

    # 2. Patient walks 2km north across multiple H3 cells
    cells_visited = []
    for i, lat_offset in enumerate([0.010, 0.015, 0.020, 0.025]):
        curr_lat = 39.9000 + lat_offset
        curr_cell = location_to_h3_cell(curr_lat, 116.4000)
        cells_visited.append(curr_cell)
        ts = f"2026-10-22T10:0{3+i}:00Z"
        resp = client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, curr_lat, 116.4000, ts=ts))
        assert resp.status_code == 200

    # Verify: There are multiple distinct H3 cells visited
    assert len(set(cells_visited)) >= 2

    # Verify: STILL EXACTLY ONE alert record exists in DB, with the latest H3 cell and timestamp
    async def _check():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].grid_cell_id == cells_visited[-1]
            assert alerts[0].status == AlertStatus.PENDING.value

    asyncio.run(_check())


def test_confirmed_reentry_auto_resolves_breach_and_notifies():
    """Requirement 6: Confirmed re-entry marks the active breach AUTO_RESOLVED and notifies WebSocket clients."""
    client = TestClient(app)
    user_id = "patient_reentry"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    # 1. Breach Home
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:00:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9050, 116.4000, ts="2026-10-22T10:01:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9060, 116.4000, ts="2026-10-22T10:02:00Z"))

    # 2. Patient returns: first point inside -> REENTRY_PENDING
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:05:00Z"))

    # 3. Patient second point inside -> Confirmed REENTERED / RESOLVED!
    with patch("app.state_machines.geofence_boundary.push_alert_update", new_callable=AsyncMock) as mock_push:
        client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:05:30Z"))
        # Verify WebSocket broadcast of AUTO_RESOLVED alert
        assert mock_push.called
        broadcast_payload = mock_push.call_args[0][1]
        assert broadcast_payload["status"] == AlertStatus.AUTO_RESOLVED.value

    # Verify: Alert in DB is now AUTO_RESOLVED
    async def _check():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].status == AlertStatus.AUTO_RESOLVED.value

    asyncio.run(_check())


def test_multiple_zones_transit_resolves_departure():
    """Requirement 7: Multiple safe zones (Home & Company) - entering Company resolves the active departure breach from Home."""
    client = TestClient(app)
    user_id = "patient_multizone"
    _grant_consent(client, user_id)

    # Home: center 39.9000, 116.4000 (radius 200m)
    # Company: center 39.9200, 116.4500 (radius 300m, ~5km away)
    home_id = _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)
    company_id = _create_circular_zone(client, user_id, "Company", 39.9200, 116.4500, 300.0)

    # 1. Start safely at Home
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T08:00:00Z"))

    # 2. Depart Home into transit -> Breach Home
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9050, 116.4000, ts="2026-10-22T08:05:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9060, 116.4000, ts="2026-10-22T08:06:00Z"))

    # Verify Home breach alert is PENDING
    async def _check_pending():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id, AlertRecordRow.status == AlertStatus.PENDING.value)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].zone_id == home_id

    asyncio.run(_check_pending())

    # 3. Arrive safely inside Company
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9200, 116.4500, ts="2026-10-22T08:30:00Z"))

    # Verify: Home breach alert was AUTO_RESOLVED upon arriving at Company safe zone
    async def _check_resolved():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].status == AlertStatus.AUTO_RESOLVED.value

    asyncio.run(_check_resolved())


def test_wearable_metrics_normal_or_missing_never_suppresses_breach():
    """Requirement 8: Normal heart rate (72 bpm) or missing heart rate must NEVER suppress an urgent geofence boundary breach."""
    client = TestClient(app)
    user_id = "patient_hr_invariance"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    # 1. Inside Home with normal heart rate (72 bpm)
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:00:00Z", hr=72))

    # 2. Exit Home with normal heart rate (72 bpm)
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9050, 116.4000, ts="2026-10-22T10:01:00Z", hr=72))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9060, 116.4000, ts="2026-10-22T10:02:00Z", hr=72))

    # Breach alert MUST trigger despite normal heart rate!
    async def _check():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].status == AlertStatus.PENDING.value
            assert alerts[0].alert_tier == 3

    asyncio.run(_check())

def test_polygonal_safe_zone_breach_and_reentry():
    """Polygonal safe zone: confirms breach when outside polygon vertices and resolution on re-entry."""
    client = TestClient(app)
    user_id = "patient_poly_breach"
    _grant_consent(client, user_id)

    # 4-point polygon around (39.90, 116.40)
    vertices = [
        [39.890, 116.390],
        [39.910, 116.390],
        [39.910, 116.410],
        [39.890, 116.410],
    ]
    resp = client.post(
        f"/api/safe-zones/{user_id}",
        json={
            "name": "Tech Park",
            "geometry_type": "polygon",
            "polygon_vertices": vertices,
            "is_enabled": True,
        },
    )
    assert resp.status_code == 201

    # Reading 1: Inside polygon
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:00:00Z"))

    # Reading 2 & 3: Way outside polygon north (39.9500)
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9500, 116.4000, ts="2026-10-22T10:01:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9600, 116.4000, ts="2026-10-22T10:02:00Z"))

    # Verify breach alert
    async def _check_breach():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id, AlertRecordRow.status == AlertStatus.PENDING.value)
            )).scalars().all()
            assert len(alerts) == 1
    asyncio.run(_check_breach())

    # Return inside polygon (2 consecutive points)
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000, ts="2026-10-22T10:05:00Z"))
    client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9010, 116.4010, ts="2026-10-22T10:05:30Z"))

    # Verify auto-resolved
    async def _check_resolved():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].status == AlertStatus.AUTO_RESOLVED.value
    asyncio.run(_check_resolved())


def test_unconsented_patient_blocked_at_ingestion():
    """Requirement 8: Consent is strictly required; unconsented telemetry is rejected with 403."""
    client = TestClient(app)
    user_id = "patient_no_consent"
    resp = client.post("/api/telemetry/ingest", json=_telemetry_payload(user_id, 39.9000, 116.4000))
    assert resp.status_code == 403
    assert "Consent not granted" in resp.json()["detail"]


def test_custom_configurable_debouncing_thresholds():
    """Requirement 3: Thresholds are fully configurable (e.g. 3 consecutive readings for exit)."""
    client = TestClient(app)
    user_id = "patient_custom_thresh"
    _grant_consent(client, user_id)
    _create_circular_zone(client, user_id, "Home", 39.9000, 116.4000, 200.0)

    custom_config = GeofenceBoundaryConfig(
        exit_consecutive_readings=3,
        reentry_consecutive_readings=3,
        hysteresis_buffer_m=10.0,
    )

    # Reading 1: Inside
    res1 = asyncio.run(evaluate_patient_geofence_boundaries(
        user_id=user_id, lat=39.9000, lng=116.4000, accuracy_m=5.0,
        timestamp=datetime.now(timezone.utc), config=custom_config
    ))
    assert res1.zone_states[0]["state"] == BoundaryState.INSIDE.value

    # Reading 2: Outside (1st) -> EXIT_PENDING
    res2 = asyncio.run(evaluate_patient_geofence_boundaries(
        user_id=user_id, lat=39.9050, lng=116.4000, accuracy_m=5.0,
        timestamp=datetime.now(timezone.utc), config=custom_config
    ))
    assert res2.zone_states[0]["state"] == BoundaryState.EXIT_PENDING.value
    assert len(res2.alerts_triggered) == 0

    # Reading 3: Outside (2nd) -> Still EXIT_PENDING (needs 3!)
    res3 = asyncio.run(evaluate_patient_geofence_boundaries(
        user_id=user_id, lat=39.9060, lng=116.4000, accuracy_m=5.0,
        timestamp=datetime.now(timezone.utc), config=custom_config
    ))
    assert res3.zone_states[0]["state"] == BoundaryState.EXIT_PENDING.value
    assert len(res3.alerts_triggered) == 0

    # Reading 4: Outside (3rd) -> BREACHED!
    res4 = asyncio.run(evaluate_patient_geofence_boundaries(
        user_id=user_id, lat=39.9070, lng=116.4000, accuracy_m=5.0,
        timestamp=datetime.now(timezone.utc), config=custom_config
    ))
    assert res4.zone_states[0]["state"] == BoundaryState.BREACHED.value
    assert len(res4.alerts_triggered) == 1

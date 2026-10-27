"""
Phase 1 Test Suite: Wearable Metrics Data Contract, Persistence, and Safe Zone CRUD Foundations.

Verifies:
1. Backward compatibility: Existing GPS-only telemetry payloads succeed without wearable_metrics.
2. Ingestion & Persistence: Valid wearable metrics are persisted into wearable_readings table.
3. Data Contract Validation: Out-of-bounds heart rates (<30, >250), malformed timestamps, and invalid wear states return 422.
4. Authorization & Consent Gating: Safe Zone CRUD operations enforce patient consent (403 when unconsented).
5. Geometry Validation: Circular and polygonal safe zones validate boundaries via ml.src.geofence abstractions.
6. Safe Zone CRUD Operations: Create, Read, Update, Soft-Disable, and Delete operations work as specified.
7. Patient Isolation: Zones are strictly scoped to user_id.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
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
    WearableReadingRow,
    SafeZoneRow,
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
            await session.execute(delete(WearableReadingRow))
            await session.execute(delete(SafeZoneRow))
            await session.commit()

    asyncio.run(_clean())


def _grant_consent(client: TestClient, user_id: str = "patient_phase1"):
    consent_payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-22T08:00:00Z",
            "guardian_id": "guardian_primary",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    resp = client.post("/api/consent/record", json=consent_payload)
    assert resp.status_code == 200, resp.text


def _sample_telemetry(user_id: str = "patient_phase1", wearable_metrics=None) -> dict:
    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-22T10:00:00Z",
        "location": {
            "lat": 39.9042,
            "lng": 116.4074,
            "altitude_m": 50.0,
        },
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": 1.2,
            "heading_deg": 180.0,
            "activity_type": "WALKING",
            "battery_pct": 85,
        },
        "signal_status": {
            "state": "VALID",
        },
    }
    if wearable_metrics is not None:
        payload["wearable_metrics"] = wearable_metrics
    return payload


# ==============================================================================
# 1. Telemetry Ingestion Contract & Persistence Tests
# ==============================================================================

def test_telemetry_gps_only_backward_compatibility():
    """Existing GPS clients transmitting payloads without wearable_metrics continue to succeed."""
    client = TestClient(app)
    _grant_consent(client, "patient_phase1")

    payload = _sample_telemetry(user_id="patient_phase1")
    assert "wearable_metrics" not in payload

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "patient_phase1"
    assert "risk_score" in data


def test_telemetry_gps_with_null_wearable_metrics():
    """Explicitly passing null for wearable_metrics is backward-compatible and accepted."""
    client = TestClient(app)
    _grant_consent(client, "patient_phase1")

    payload = _sample_telemetry(user_id="patient_phase1")
    payload["wearable_metrics"] = None

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "patient_phase1"


def test_telemetry_with_valid_wearable_metrics_persists_to_db():
    """When valid wearable_metrics are provided, reading is stored in wearable_readings table."""
    client = TestClient(app)
    _grant_consent(client, "patient_hr_01")

    wearable = {
        "heart_rate_bpm": 74,
        "timestamp": "2026-10-22T10:15:00Z",
        "wear_state": "ON_WRIST",
        "sensor_status": "ACCURACY_HIGH",
    }
    payload = _sample_telemetry(user_id="patient_hr_01", wearable_metrics=wearable)

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200

    # Query DB directly to verify persistence
    async def _check_db():
        async with async_session() as session:
            result = await session.execute(
                select(WearableReadingRow).where(WearableReadingRow.user_id == "patient_hr_01")
            )
            rows = result.scalars().all()
            assert len(rows) == 1
            row = rows[0]
            assert row.user_id == "patient_hr_01"
            assert row.heart_rate_bpm == 74
            assert row.wear_state == "ON_WRIST"
            assert row.sensor_status == "ACCURACY_HIGH"
            assert row.created_at is not None

    asyncio.run(_check_db())


@pytest.mark.parametrize("invalid_hr", [29, 0, -10, 251, 300])
def test_telemetry_rejects_out_of_bounds_heart_rate(invalid_hr):
    """Heart rate values outside physiological bounds [30, 250] must be rejected with HTTP 422."""
    client = TestClient(app)
    _grant_consent(client, "patient_hr_bounds")

    wearable = {
        "heart_rate_bpm": invalid_hr,
        "timestamp": "2026-10-22T10:15:00Z",
        "wear_state": "ON_WRIST",
    }
    payload = _sample_telemetry(user_id="patient_hr_bounds", wearable_metrics=wearable)

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 422


def test_telemetry_rejects_malformed_timestamp_in_wearable_metrics():
    """Malformed date strings in wearable_metrics must be rejected with HTTP 422."""
    client = TestClient(app)
    _grant_consent(client, "patient_hr_ts")

    wearable = {
        "heart_rate_bpm": 80,
        "timestamp": "invalid-datetime-string",
        "wear_state": "ON_WRIST",
    }
    payload = _sample_telemetry(user_id="patient_hr_ts", wearable_metrics=wearable)

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 422


def test_telemetry_rejects_invalid_wear_state():
    """Non-enum wear_state values must be rejected with HTTP 422."""
    client = TestClient(app)
    _grant_consent(client, "patient_hr_state")

    wearable = {
        "heart_rate_bpm": 80,
        "timestamp": "2026-10-22T10:15:00Z",
        "wear_state": "INVALID_STATE",
    }
    payload = _sample_telemetry(user_id="patient_hr_state", wearable_metrics=wearable)

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 422


# ==============================================================================
# 2. Safe Zone Consent Gating & Patient Authorization Tests
# ==============================================================================

def test_safe_zone_create_rejected_without_consent():
    """Creating a safe zone for an unconsented patient returns HTTP 403 Forbidden."""
    client = TestClient(app)
    zone_payload = {
        "name": "Home",
        "geometry_type": "circle",
        "center_lat": 39.9042,
        "center_lng": 116.4074,
        "radius_m": 200.0,
    }
    resp = client.post("/api/safe-zones/unconsented_patient", json=zone_payload)
    assert resp.status_code == 403
    assert "Consent not granted" in resp.json()["detail"]


def test_safe_zone_list_rejected_without_consent():
    """Listing safe zones for an unconsented patient returns HTTP 403 Forbidden."""
    client = TestClient(app)
    resp = client.get("/api/safe-zones/unconsented_patient")
    assert resp.status_code == 403


# ==============================================================================
# 3. Safe Zone Geometry Validation Tests
# ==============================================================================

def test_safe_zone_create_circle_success():
    """Valid circular zone is created and persisted with HTTP 201."""
    client = TestClient(app)
    _grant_consent(client, "patient_sz_01")

    payload = {
        "name": "Home Residence",
        "description": "Primary residence geofence",
        "geometry_type": "circle",
        "center_lat": 39.9042,
        "center_lng": 116.4074,
        "radius_m": 350.0,
        "is_enabled": True,
    }
    resp = client.post("/api/safe-zones/patient_sz_01", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["zone_id"].startswith("zone_")
    assert data["name"] == "Home Residence"
    assert data["geometry_type"] == "circle"
    assert data["center_lat"] == 39.9042
    assert data["center_lng"] == 116.4074
    assert data["radius_m"] == 350.0
    assert data["is_enabled"] is True


@pytest.mark.parametrize("invalid_circle", [
    {"name": "Home", "geometry_type": "circle", "center_lat": None, "center_lng": 116.4, "radius_m": 100},
    {"name": "Home", "geometry_type": "circle", "center_lat": 39.9, "center_lng": None, "radius_m": 100},
    {"name": "Home", "geometry_type": "circle", "center_lat": 39.9, "center_lng": 116.4, "radius_m": 0},
    {"name": "Home", "geometry_type": "circle", "center_lat": 39.9, "center_lng": 116.4, "radius_m": -50},
    {"name": "Home", "geometry_type": "circle", "center_lat": 95.0, "center_lng": 116.4, "radius_m": 100},
    {"name": "Home", "geometry_type": "circle", "center_lat": 39.9, "center_lng": 200.0, "radius_m": 100},
])
def test_safe_zone_create_circle_invalid_parameters(invalid_circle):
    """Invalid circle parameters (missing coords, negative/zero radius, out-of-bounds) return 422."""
    client = TestClient(app)
    _grant_consent(client, "patient_sz_01")

    resp = client.post("/api/safe-zones/patient_sz_01", json=invalid_circle)
    assert resp.status_code == 422


def test_safe_zone_create_polygon_success():
    """Valid polygonal zone with >= 3 vertices is created and persisted with HTTP 201."""
    client = TestClient(app)
    _grant_consent(client, "patient_sz_02")

    vertices = [
        [39.900, 116.400],
        [39.910, 116.400],
        [39.910, 116.410],
        [39.900, 116.410],
    ]
    payload = {
        "name": "Company Campus",
        "description": "Daytime workplace safe perimeter",
        "geometry_type": "polygon",
        "polygon_vertices": vertices,
        "is_enabled": True,
    }
    resp = client.post("/api/safe-zones/patient_sz_02", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["zone_id"].startswith("zone_")
    assert data["name"] == "Company Campus"
    assert data["geometry_type"] == "polygon"
    assert data["polygon_vertices"] == vertices
    assert data["center_lat"] is None
    assert data["radius_m"] is None


@pytest.mark.parametrize("invalid_poly_vertices", [
    [],
    [[39.9, 116.4]],
    [[39.9, 116.4], [39.91, 116.41]],  # only 2 points
    [[95.0, 116.4], [39.91, 116.41], [39.90, 116.42]],  # lat out of bounds
    [[39.9, 200.0], [39.91, 116.41], [39.90, 116.42]],  # lng out of bounds
])
def test_safe_zone_create_polygon_invalid_parameters(invalid_poly_vertices):
    """Polygonal zones with fewer than 3 vertices or out-of-range coordinates return 422."""
    client = TestClient(app)
    _grant_consent(client, "patient_sz_02")

    payload = {
        "name": "Invalid Polygon",
        "geometry_type": "polygon",
        "polygon_vertices": invalid_poly_vertices,
    }
    resp = client.post("/api/safe-zones/patient_sz_02", json=payload)
    assert resp.status_code == 422


# ==============================================================================
# 4. Safe Zone CRUD Lifecycle & Patient Isolation Tests
# ==============================================================================

def test_safe_zone_crud_lifecycle():
    """Complete lifecycle: create -> read -> list -> update -> disable -> delete."""
    client = TestClient(app)
    user_id = "patient_crud"
    _grant_consent(client, user_id)

    # 1. Create circular Home zone
    create_resp = client.post(
        f"/api/safe-zones/{user_id}",
        json={
            "name": "Home",
            "geometry_type": "circle",
            "center_lat": 39.9042,
            "center_lng": 116.4074,
            "radius_m": 200.0,
        },
    )
    assert create_resp.status_code == 201
    zone_id = create_resp.json()["zone_id"]

    # 2. Retrieve by zone_id
    get_resp = client.get(f"/api/safe-zones/{user_id}/{zone_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["zone_id"] == zone_id
    assert get_resp.json()["name"] == "Home"

    # 3. List zones for user
    list_resp = client.get(f"/api/safe-zones/{user_id}")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    # 4. Update zone (change name and radius)
    update_resp = client.put(
        f"/api/safe-zones/{user_id}/{zone_id}",
        json={"name": "Home Sweet Home", "radius_m": 400.0},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Home Sweet Home"
    assert update_resp.json()["radius_m"] == 400.0

    # 5. Disable zone via soft DELETE
    disable_resp = client.delete(f"/api/safe-zones/{user_id}/{zone_id}")
    assert disable_resp.status_code == 200
    assert disable_resp.json()["status"] == "disabled"

    # Verify is_enabled is False in DB and filtered when active_only=True
    get_after_disable = client.get(f"/api/safe-zones/{user_id}/{zone_id}")
    assert get_after_disable.json()["is_enabled"] is False

    active_list = client.get(f"/api/safe-zones/{user_id}?active_only=true")
    assert len(active_list.json()) == 0

    all_list = client.get(f"/api/safe-zones/{user_id}")
    assert len(all_list.json()) == 1

    # 6. Permanently delete zone
    delete_resp = client.delete(f"/api/safe-zones/{user_id}/{zone_id}?permanent=true")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    # Verify zone is gone
    get_after_del = client.get(f"/api/safe-zones/{user_id}/{zone_id}")
    assert get_after_del.status_code == 404


def test_safe_zone_patient_isolation():
    """Safe zones for patient A cannot be accessed or modified by patient B."""
    client = TestClient(app)
    _grant_consent(client, "patient_A")
    _grant_consent(client, "patient_B")

    # Patient A creates zone
    resp = client.post(
        "/api/safe-zones/patient_A",
        json={
            "name": "Home A",
            "geometry_type": "circle",
            "center_lat": 39.9,
            "center_lng": 116.4,
            "radius_m": 200.0,
        },
    )
    assert resp.status_code == 201
    zone_a_id = resp.json()["zone_id"]

    # Patient B attempts to get Patient A's zone -> 404
    resp_b_get = client.get(f"/api/safe-zones/patient_B/{zone_a_id}")
    assert resp_b_get.status_code == 404

    # Patient B attempts to update Patient A's zone -> 404
    resp_b_put = client.put(f"/api/safe-zones/patient_B/{zone_a_id}", json={"name": "Hacked"})
    assert resp_b_put.status_code == 404

    # Patient B attempts to delete Patient A's zone -> 404
    resp_b_del = client.delete(f"/api/safe-zones/patient_B/{zone_a_id}")
    assert resp_b_del.status_code == 404

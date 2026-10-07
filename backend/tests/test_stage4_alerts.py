"""
Stage 4 smoke & integration tests — verifies alert creation, deduplication,
WebSocket alert broadcasting, and the closed-loop dismissal suppression math.

Run with:  python -m pytest backend/tests/test_stage4_alerts.py -v
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
from sqlalchemy import delete, select

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
from app.services.geospatial import location_to_h3_cell


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
            "granted_at": "2026-10-06T12:00:00Z",
            "guardian_id": "guardian_stage4",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


# ---------------------------------------------------------------
# Check 1: CRITICAL telemetry creates PENDING AlertRecord with correct H3 cell
# ---------------------------------------------------------------

def test_critical_telemetry_creates_pending_alert_record(client: TestClient):
    """
    Check 1: Telemetry producing a CRITICAL risk_tier creates a PENDING
    AlertRecord in the database with the correct H3 resolution 9 grid_cell_id.
    """
    user_id = "patient_stage4_create"
    _grant_consent(client, user_id)

    lat, lng = 51.5074, -0.1278
    expected_cell = location_to_h3_cell(lat, lng)
    assert expected_cell == "89195da49b7ffff"

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:01:30Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 7.0,  # High speed drives raw_score >= 84 -> CRITICAL
            "heading_deg": 180.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }

    r = client.post("/api/telemetry/ingest", json=payload)
    assert r.status_code == 200
    risk_data = r.json()
    assert risk_data["risk_tier"] == "CRITICAL"

    # Verify AlertRecord in SQLite
    async def _get_alert():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts = asyncio.run(_get_alert())
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.status == "PENDING"
    assert alert.alert_tier == 3
    assert alert.grid_cell_id == expected_cell

    # Verify retrieval via GET /api/alerts/{alert_id}
    r_get = client.get(f"/api/alerts/{alert.alert_id}")
    assert r_get.status_code == 200
    alert_api = r_get.json()
    assert alert_api["alert_id"] == alert.alert_id
    assert alert_api["status"] == "PENDING"
    assert alert_api["grid_cell_id"] == expected_cell
    assert alert_api["alert_tier"] == 3


# ---------------------------------------------------------------
# Check 2: Deduplication — updates existing PENDING alert instead of spamming
# ---------------------------------------------------------------

def test_sustained_high_risk_deduplicates_pending_alert(client: TestClient):
    """
    Check 2: A second telemetry payload for the same user in the same grid cell,
    still high risk, does NOT create a second PENDING alert — it updates the
    existing one's timestamp.
    """
    user_id = "patient_stage4_dedup"
    _grant_consent(client, user_id)

    lat, lng = 51.5074, -0.1278
    p1 = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:00:10Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 7.0,
            "heading_deg": 180.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }
    r1 = client.post("/api/telemetry/ingest", json=p1)
    assert r1.status_code == 200
    assert r1.json()["risk_tier"] == "CRITICAL"

    async def _get_alerts():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == user_id)
            )
            return result.scalars().all()

    alerts_1 = asyncio.run(_get_alerts())
    assert len(alerts_1) == 1
    first_alert_id = alerts_1[0].alert_id
    first_timestamp = alerts_1[0].timestamp

    # Send second reading 20 seconds later in the same cell
    p2 = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:00:30Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 6.8,
            "heading_deg": 180.0,
            "battery_pct": 79,
        },
        "signal_status": {"state": "VALID"},
    }
    r2 = client.post("/api/telemetry/ingest", json=p2)
    assert r2.status_code == 200

    alerts_2 = asyncio.run(_get_alerts())
    assert len(alerts_2) == 1, "Must NOT create a duplicate alert row"
    updated_alert = alerts_2[0]
    assert updated_alert.alert_id == first_alert_id
    assert updated_alert.timestamp > first_timestamp


# ---------------------------------------------------------------
# Check 3: Closed-loop suppression after dismissal with exact math assertion
# ---------------------------------------------------------------

def test_dismissal_suppresses_subsequent_risk_score_exact_math(client: TestClient):
    """
    Check 3: Dismissing an alert via POST /api/alerts/{alert_id}/dismiss, then
    sending new telemetry for the same user+location that would otherwise score
    CRITICAL, produces a measurably lower risk_score due to the sensitivity
    multiplier — asserts the exact multiplier math (score * 0.90).
    """
    user_id = "patient_stage4_suppression"
    _grant_consent(client, user_id)

    lat, lng = 51.5074, -0.1278
    p1 = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:01:00Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 7.0,
            "heading_deg": 180.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }

    r1 = client.post("/api/telemetry/ingest", json=p1)
    assert r1.status_code == 200
    data1 = r1.json()
    initial_score = data1["risk_score"]
    assert initial_score >= 85.0  # CRITICAL baseline

    # Get the generated alert ID
    async def _get_alert_id():
        async with async_session() as session:
            result = await session.execute(
                select(AlertRecordRow).where(
                    AlertRecordRow.user_id == user_id,
                    AlertRecordRow.status == "PENDING",
                )
            )
            return result.scalar_one().alert_id

    alert_id = asyncio.run(_get_alert_id())

    # Dismiss the alert
    r_dismiss = client.post(f"/api/alerts/{alert_id}/dismiss")
    assert r_dismiss.status_code == 200
    dismiss_data = r_dismiss.json()
    assert dismiss_data["status"] == "dismissed"
    assert dismiss_data["dismissal_count_in_window"] == 1
    assert dismiss_data["cell_sensitivity_multiplier"] == 0.90

    # Verify the alert is marked DISMISSED_SAFE in SQLite
    r_alert_get = client.get(f"/api/alerts/{alert_id}")
    assert r_alert_get.status_code == 200
    assert r_alert_get.json()["status"] == "DISMISSED_SAFE"
    assert r_alert_get.json()["dismissal"]["dismissal_count_in_window"] == 1

    # Send new telemetry for the same user in the same 2-min window (same pseudo_rand seed)
    # The raw score before suppression is identical to initial_score.
    p2 = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:01:30Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 7.0,
            "heading_deg": 180.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }
    r2 = client.post("/api/telemetry/ingest", json=p2)
    assert r2.status_code == 200
    data2 = r2.json()

    # Exact multiplier math assertion: suppressed_score = round(raw_score * 0.90, 1)
    expected_score = round(initial_score * 0.90, 1)
    assert data2["risk_score"] == expected_score
    # Suppression caused de-escalation: score dropped below 85.0 to SUSPICIOUS
    assert data2["risk_tier"] == "SUSPICIOUS"


# ---------------------------------------------------------------
# Check 4: 4th dismissal forces formal safe-zone recalibration flow
# ---------------------------------------------------------------

def test_fourth_dismissal_requires_recalibration(client: TestClient):
    """
    Check 4: 4th dismissal attempt for the same grid cell within 30 days
    returns recalibration_required response.
    """
    user_id = "patient_stage4_recalib"
    lat, lng = 51.5074, -0.1278
    grid_cell_id = location_to_h3_cell(lat, lng)

    async def _create_alert(alert_id: str):
        async with async_session() as session:
            session.add(
                AlertRecordRow(
                    alert_id=alert_id,
                    user_id=user_id,
                    grid_cell_id=grid_cell_id,
                    timestamp=datetime.now(timezone.utc),
                    alert_tier=3,
                    status="PENDING",
                )
            )
            await session.commit()

    # Create and dismiss 3 alerts in the same cell
    for i in range(3):
        alert_id = f"alert_recalib_{i}"
        asyncio.run(_create_alert(alert_id))
        r_dismiss = client.post(f"/api/alerts/{alert_id}/dismiss")
        assert r_dismiss.status_code == 200
        assert r_dismiss.json()["status"] == "dismissed"
        assert r_dismiss.json()["dismissal_count_in_window"] == i + 1

    # 4th alert in the same cell
    alert_id_4 = "alert_recalib_3"
    asyncio.run(_create_alert(alert_id_4))

    # 4th dismissal must return recalibration_required
    r_dismiss_4 = client.post(f"/api/alerts/{alert_id_4}/dismiss")
    assert r_dismiss_4.status_code == 200
    res4 = r_dismiss_4.json()
    assert res4["status"] == "recalibration_required"
    assert "Routine change detected" in res4["message"]


# ---------------------------------------------------------------
# Check 5: WebSocket /ws/alerts/{user_id} receives created AlertRecord
# ---------------------------------------------------------------

def test_websocket_receives_alert_record(client: TestClient):
    """
    Check 5: A WebSocket client connected to /ws/alerts/{user_id} receives
    the AlertRecord when an alert is created.
    """
    user_id = "patient_stage4_ws"
    _grant_consent(client, user_id)

    lat, lng = 51.5074, -0.1278
    expected_cell = location_to_h3_cell(lat, lng)

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-06T12:15:00Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 7.0,
            "heading_deg": 180.0,
            "battery_pct": 80,
        },
        "signal_status": {"state": "VALID"},
    }

    with client.websocket_connect(f"/ws/alerts/{user_id}") as ws:
        r = client.post("/api/telemetry/ingest", json=payload)
        assert r.status_code == 200
        http_data = r.json()
        expected_tier = 3 if http_data["risk_tier"] == "CRITICAL" else 2

        ws_alert = ws.receive_json()
        assert ws_alert["user_id"] == user_id
        assert ws_alert["status"] == "PENDING"
        assert ws_alert["alert_tier"] == expected_tier
        assert ws_alert["grid_cell_id"] == expected_cell

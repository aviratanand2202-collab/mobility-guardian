"""
Regression tests for alert recovery & initial hydration endpoints.

Verifies:
1. Persisted pending alerts can be retrieved for a user via GET /api/alerts/user/{user_id}.
2. One user cannot retrieve another user's alerts (strict user scoping).
3. Querying /api/alerts without user_id returns 400 Bad Request.
4. Dismissed alerts retain DISMISSED_SAFE status and dismissal metadata on retrieval.
5. Status filtering and descending timestamp ordering behave correctly.

Run with: python -m pytest backend/tests/test_alert_recovery.py -v
"""
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

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
from app.services.geospatial import location_to_h3_cell


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
            "granted_at": "2026-10-14T10:00:00Z",
            "guardian_id": "guardian_recovery",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200


def test_persisted_pending_alert_retrieval(client: TestClient):
    """
    Test 1: Ingesting high-risk telemetry creates a PENDING alert, which can
    then be retrieved via GET /api/alerts/user/{user_id} and GET /api/alerts?user_id={user_id}.
    """
    user_id = "user_recov_01"
    _grant_consent(client, user_id)

    lat, lng = 51.5074, -0.1278
    expected_cell = location_to_h3_cell(lat, lng)

    payload = {
        "user_id": user_id,
        "timestamp": "2026-10-14T10:05:00Z",
        "location": {"lat": lat, "lng": lng},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.0,
            "speed_mps": 7.5,
            "heading_deg": 90.0,
            "battery_pct": 85,
        },
        "signal_status": {"state": "VALID"},
    }

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200

    # Retrieve via path endpoint
    get_res = client.get(f"/api/alerts/user/{user_id}")
    assert get_res.status_code == 200
    alerts = get_res.json()
    assert len(alerts) == 1
    assert alerts[0]["user_id"] == user_id
    assert alerts[0]["status"] == "PENDING"
    assert alerts[0]["alert_tier"] == 3
    assert alerts[0]["grid_cell_id"] == expected_cell
    assert alerts[0]["dismissal"] is None

    # Retrieve via query parameter endpoint
    query_res = client.get(f"/api/alerts?user_id={user_id}")
    assert query_res.status_code == 200
    query_alerts = query_res.json()
    assert len(query_alerts) == 1
    assert query_alerts[0]["alert_id"] == alerts[0]["alert_id"]


def test_user_scoping_prevents_cross_user_leakage(client: TestClient):
    """
    Test 2: Each user can only retrieve their own alerts; querying without
    user_id returns HTTP 400 Bad Request to prevent un-scoped leaks.
    """
    user_a = "user_recov_alice"
    user_b = "user_recov_bob"
    _grant_consent(client, user_a)
    _grant_consent(client, user_b)

    # Ingest for Alice
    client.post("/api/telemetry/ingest", json={
        "user_id": user_a,
        "timestamp": "2026-10-14T10:10:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 0.0, "battery_pct": 90},
        "signal_status": {"state": "VALID"},
    })

    # Ingest for Bob
    client.post("/api/telemetry/ingest", json={
        "user_id": user_b,
        "timestamp": "2026-10-14T10:11:00Z",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 0.0, "battery_pct": 90},
        "signal_status": {"state": "VALID"},
    })

    # Alice gets only her alert
    res_a = client.get(f"/api/alerts/user/{user_a}")
    assert res_a.status_code == 200
    alerts_a = res_a.json()
    assert len(alerts_a) == 1
    assert alerts_a[0]["user_id"] == user_a

    # Bob gets only his alert
    res_b = client.get(f"/api/alerts/user/{user_b}")
    assert res_b.status_code == 200
    alerts_b = res_b.json()
    assert len(alerts_b) == 1
    assert alerts_b[0]["user_id"] == user_b

    # Unrelated user gets empty list
    res_c = client.get("/api/alerts/user/user_unknown_charlie")
    assert res_c.status_code == 200
    assert res_c.json() == []

    # Un-scoped GET /api/alerts without user_id rejected
    res_unscoped = client.get("/api/alerts")
    assert res_unscoped.status_code == 400
    assert "user_id query parameter is required" in res_unscoped.json()["detail"]


def test_dismissed_alert_retains_dismissed_status_and_metadata(client: TestClient):
    """
    Test 3: When an alert is dismissed, subsequent retrieval via the user
    endpoint returns DISMISSED_SAFE along with complete dismissal metadata.
    """
    user_id = "user_recov_dismiss"
    _grant_consent(client, user_id)

    # Ingest telemetry
    client.post("/api/telemetry/ingest", json={
        "user_id": user_id,
        "timestamp": "2026-10-14T10:20:00Z",
        "location": {"lat": 51.5074, "lng": -0.1278},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 8.0, "heading_deg": 0.0, "battery_pct": 90},
        "signal_status": {"state": "VALID"},
    })

    # Get generated alert
    alerts = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(alerts) == 1
    alert_id = alerts[0]["alert_id"]
    assert alerts[0]["status"] == "PENDING"

    # Caregiver dismisses the alert
    dismiss_res = client.post(f"/api/alerts/{alert_id}/dismiss")
    assert dismiss_res.status_code == 200
    assert dismiss_res.json()["status"] == "dismissed"

    # Recover alerts for user after dismissal (e.g. browser refresh simulation)
    recovered = client.get(f"/api/alerts/user/{user_id}").json()
    assert len(recovered) == 1
    rec = recovered[0]
    assert rec["alert_id"] == alert_id
    assert rec["status"] == "DISMISSED_SAFE"
    assert rec["dismissal"] is not None
    assert rec["dismissal"]["dismissal_count_in_window"] == 1
    assert rec["dismissal"]["dismissed_at"] is not None
    assert rec["dismissal"]["decay_clock_start"] is not None


def test_status_filtering_and_descending_ordering(client: TestClient):
    """
    Test 4: User alert retrieval respects status filtering and orders
    alerts descending by timestamp (newest first).
    """
    user_id = "user_recov_order"
    now = datetime.now(timezone.utc)

    # Directly insert 3 alert rows with distinct timestamps and statuses
    async def _seed():
        async with async_session() as session:
            # Oldest alert: DISMISSED_SAFE
            session.add(AlertRecordRow(
                alert_id="alert_order_1",
                user_id=user_id,
                grid_cell_id="cell_1",
                timestamp=now - timedelta(minutes=10),
                alert_tier=2,
                status="DISMISSED_SAFE",
                dismissed_at=now - timedelta(minutes=9),
                dismissal_decay_clock_start=now - timedelta(minutes=9),
                dismissal_count_in_window=1,
            ))
            # Middle alert: PENDING
            session.add(AlertRecordRow(
                alert_id="alert_order_2",
                user_id=user_id,
                grid_cell_id="cell_2",
                timestamp=now - timedelta(minutes=5),
                alert_tier=2,
                status="PENDING",
            ))
            # Newest alert: PENDING
            session.add(AlertRecordRow(
                alert_id="alert_order_3",
                user_id=user_id,
                grid_cell_id="cell_3",
                timestamp=now,
                alert_tier=3,
                status="PENDING",
            ))
            await session.commit()

    asyncio.run(_seed())

    # 1. Unfiltered: returns all 3, newest first
    all_res = client.get(f"/api/alerts/user/{user_id}")
    assert all_res.status_code == 200
    all_alerts = all_res.json()
    assert len(all_alerts) == 3
    assert all_alerts[0]["alert_id"] == "alert_order_3"  # newest
    assert all_alerts[1]["alert_id"] == "alert_order_2"
    assert all_alerts[2]["alert_id"] == "alert_order_1"  # oldest

    # 2. Filter status=PENDING: returns only the 2 pending alerts
    pending_res = client.get(f"/api/alerts/user/{user_id}?status=PENDING")
    assert pending_res.status_code == 200
    pending_alerts = pending_res.json()
    assert len(pending_alerts) == 2
    assert pending_alerts[0]["alert_id"] == "alert_order_3"
    assert pending_alerts[1]["alert_id"] == "alert_order_2"

    # 3. Filter status=DISMISSED_SAFE: returns only the 1 dismissed alert
    dismissed_res = client.get(f"/api/alerts/user/{user_id}?status=DISMISSED_SAFE")
    assert dismissed_res.status_code == 200
    dismissed_alerts = dismissed_res.json()
    assert len(dismissed_alerts) == 1
    assert dismissed_alerts[0]["alert_id"] == "alert_order_1"

    # 4. Limit parameter: caps at limit
    limit_res = client.get(f"/api/alerts/user/{user_id}?limit=2")
    assert limit_res.status_code == 200
    assert len(limit_res.json()) == 2

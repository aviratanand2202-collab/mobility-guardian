"""
Unit and Integration tests for Wearable Biometrics API endpoints (/api/wearable).
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import app
from app.db import init_db, dispose_engine, async_session, WearableReadingRow, ConsentRecordRow


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    asyncio.run(init_db())
    yield
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def clean_wearables():
    async def _clean():
        async with async_session() as session:
            await session.execute(delete(WearableReadingRow))
            await session.execute(delete(ConsentRecordRow))
            await session.commit()
    asyncio.run(_clean())


def _grant_consent(client: TestClient, user_id: str):
    payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-23T10:00:00Z",
            "guardian_id": "guardian_hr_test",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    resp = client.post("/api/consent/record", json=payload)
    assert resp.status_code == 200


def test_wearable_latest_no_data():
    client = TestClient(app)
    resp = client.get("/api/wearable/user_empty/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "NO_DATA"
    assert data["user_id"] == "user_empty"
    assert data["reading"] is None


def test_telemetry_with_wearable_and_latest_endpoint():
    client = TestClient(app)
    user_id = "patient_hr_live"
    _grant_consent(client, user_id)

    # Ingest telemetry with wearable metrics
    now_iso = datetime.now(timezone.utc).isoformat()
    telemetry_payload = {
        "user_id": user_id,
        "timestamp": now_iso,
        "location": {"lat": 39.9042, "lng": 116.4074, "altitude_m": 12.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 4.5,
            "speed_mps": 0.5,
            "heading_deg": 90.0,
            "activity_type": "WALKING",
            "battery_pct": 92,
        },
        "signal_status": {"state": "VALID"},
        "wearable_metrics": {
            "heart_rate_bpm": 78,
            "timestamp": now_iso,
            "wear_state": "ON_WRIST",
            "sensor_status": "ACCURACY_HIGH",
        },
    }

    ingest_resp = client.post("/api/telemetry/ingest", json=telemetry_payload)
    assert ingest_resp.status_code == 200
    risk_output = ingest_resp.json()
    assert risk_output["wearable_metrics"] is not None
    assert risk_output["wearable_metrics"]["heart_rate_bpm"] == 78

    # Query latest wearable reading
    latest_resp = client.get(f"/api/wearable/{user_id}/latest")
    assert latest_resp.status_code == 200
    latest_data = latest_resp.json()
    assert latest_data["status"] == "SUCCESS"
    assert latest_data["user_id"] == user_id
    assert latest_data["reading"]["heart_rate_bpm"] == 78
    assert latest_data["reading"]["wear_state"] == "ON_WRIST"
    assert latest_data["is_stale"] is False
    assert latest_data["age_seconds"] < 10.0

    # Query wearable history
    hist_resp = client.get(f"/api/wearable/{user_id}/history?limit=10")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert len(hist_data) == 1
    assert hist_data[0]["heart_rate_bpm"] == 78
    assert hist_data[0]["wear_state"] == "ON_WRIST"

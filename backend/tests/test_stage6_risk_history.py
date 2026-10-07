"""
Stage 6 integration tests: Risk history persistence, trend line querying,
and raw payload inspection endpoints.

Run with: pytest backend/tests/test_stage6_risk_history.py -v
"""
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
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
    RiskScoreHistoryRow,
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
            await session.execute(delete(RiskScoreHistoryRow))
            await session.commit()
    asyncio.run(_clean())
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _grant_consent(client: TestClient, user_id: str):
    """Helper to register consent for a user."""
    payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_test",
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }
    resp = client.post("/api/consent/record", json=payload)
    assert resp.status_code == 200


def _make_telemetry(
    user_id: str,
    timestamp: datetime,
    lat: float = 13.0827,
    lng: float = 80.2707,
    speed_mps: float = 1.2,
    battery_pct: int = 80,
) -> dict:
    return {
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "location": {"lat": lat, "lng": lng, "altitude_m": 10.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": speed_mps,
            "heading_deg": 45.0,
            "activity_type": "WALKING",
            "battery_pct": battery_pct,
        },
        "signal_status": {"state": "VALID", "degraded_since": None},
        "imu_metrics": None,
    }


def test_risk_history_persistence_and_order(client: TestClient):
    """
    Test 1: Check that computing risk via /api/telemetry/ingest persists
    history rows, returned in ascending timestamp order with required fields.
    """
    user_id = "user_hist_01"
    _grant_consent(client, user_id)

    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    timestamps = [base_time + timedelta(seconds=i * 10) for i in range(5)]

    posted_scores = []
    for ts in timestamps:
        payload = _make_telemetry(user_id, ts)
        resp = client.post("/api/telemetry/ingest", json=payload)
        assert resp.status_code == 200
        posted_scores.append(resp.json()["risk_score"])

    # Query risk history
    resp = client.get(f"/api/risk-history/{user_id}")
    assert resp.status_code == 200
    entries = resp.json()

    assert len(entries) == 5

    # Assert strictly ascending timestamp order
    returned_ts = [
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        for e in entries
    ]
    for i in range(len(returned_ts) - 1):
        assert returned_ts[i] <= returned_ts[i + 1]

    # Assert required fields are present and raw_output_json is excluded
    for idx, entry in enumerate(entries):
        assert "id" in entry
        assert "timestamp" in entry
        assert "risk_tier" in entry
        assert "risk_score" in entry
        assert "polling_tier" in entry
        assert "grid_cell_id" in entry
        assert "raw_output_json" not in entry
        assert entry["risk_score"] == posted_scores[idx]


def test_risk_history_time_window_filtering(client: TestClient):
    """
    Test 2: Check that start and end query params accurately filter entries
    to a narrow temporal window.
    """
    user_id = "user_hist_02"
    _grant_consent(client, user_id)

    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=40)
    t1 = now - timedelta(minutes=30)
    t2 = now - timedelta(minutes=20)
    t3 = now - timedelta(minutes=10)

    for ts in [t0, t1, t2, t3]:
        resp = client.post(
            "/api/telemetry/ingest", json=_make_telemetry(user_id, ts)
        )
        assert resp.status_code == 200

    # Query with window covering only [t1, t2]
    start_filter = (t1 - timedelta(seconds=1)).isoformat()
    end_filter = (t2 + timedelta(seconds=1)).isoformat()

    resp = client.get(
        f"/api/risk-history/{user_id}?start={start_filter}&end={end_filter}"
    )
    assert resp.status_code == 200
    filtered = resp.json()

    assert len(filtered) == 2
    # Verify timestamps match t1 and t2
    ts_list = [
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        for e in filtered
    ]
    assert abs((ts_list[0] - t1).total_seconds()) < 1.0
    assert abs((ts_list[1] - t2).total_seconds()) < 1.0


def test_risk_history_raw_endpoint(client: TestClient):
    """
    Test 3: Check that /api/risk-history/{user_id}/{history_id}/raw returns
    the full original RiskScoreOutput JSON, including kinematic_features
    and explainability, and returns 404 for unknown records.
    """
    user_id = "user_hist_03"
    _grant_consent(client, user_id)

    ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    resp = client.post(
        "/api/telemetry/ingest", json=_make_telemetry(user_id, ts)
    )
    assert resp.status_code == 200
    ingest_json = resp.json()

    # Get the history id
    hist_resp = client.get(f"/api/risk-history/{user_id}")
    assert hist_resp.status_code == 200
    entries = hist_resp.json()
    assert len(entries) == 1
    history_id = entries[0]["id"]

    # Request the raw payload
    raw_resp = client.get(f"/api/risk-history/{user_id}/{history_id}/raw")
    assert raw_resp.status_code == 200
    raw_json = raw_resp.json()

    # Verify deep fields from MockRiskScoreOutput
    assert raw_json["user_id"] == user_id
    assert raw_json["risk_tier"] == ingest_json["risk_tier"]
    assert raw_json["risk_score"] == ingest_json["risk_score"]
    assert "kinematic_features" in raw_json
    assert "tortuosity_index" in raw_json["kinematic_features"]
    assert "entropy_value" in raw_json["kinematic_features"]
    assert "explainability" in raw_json
    assert "top_features" in raw_json["explainability"]
    assert "trigger_state" in raw_json
    assert "polling_instruction" in raw_json

    # Test 404 on missing record ID
    missing_resp = client.get(f"/api/risk-history/{user_id}/99999/raw")
    assert missing_resp.status_code == 404

    # Test 404 on mismatched user ID
    mismatch_resp = client.get(
        f"/api/risk-history/wrong_user/{history_id}/raw"
    )
    assert mismatch_resp.status_code == 404


def test_risk_history_limit_capping(client: TestClient):
    """
    Test 4: Verify that the limit query parameter is respected and capped at 2000.
    """
    user_id = "user_hist_04"
    _grant_consent(client, user_id)

    base = datetime.now(timezone.utc) - timedelta(minutes=15)
    for i in range(10):
        client.post(
            "/api/telemetry/ingest",
            json=_make_telemetry(user_id, base + timedelta(seconds=i * 5)),
        )

    # Test small limit
    resp_small = client.get(f"/api/risk-history/{user_id}?limit=3")
    assert resp_small.status_code == 200
    assert len(resp_small.json()) == 3

    # Test excessive limit (capped at 2000 without error)
    resp_large = client.get(f"/api/risk-history/{user_id}?limit=5000")
    assert resp_large.status_code == 200
    # We only have 10 records, but the request was capped and succeeded
    assert len(resp_large.json()) == 10


def test_risk_history_list_includes_location(client: TestClient):
    """
    Test 5: Verify that GET /api/risk-history/{user_id} includes the location
    field extracted from the stored raw_output_json when present, and returns
    None gracefully when a historical row predates the location field addition.
    """
    user_id = "user_hist_loc_01"
    _grant_consent(client, user_id)

    ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    telemetry = _make_telemetry(user_id, ts, lat=37.7749, lng=-122.4194)
    resp = client.post("/api/telemetry/ingest", json=telemetry)
    assert resp.status_code == 200

    hist_resp = client.get(f"/api/risk-history/{user_id}")
    assert hist_resp.status_code == 200
    entries = hist_resp.json()
    assert len(entries) == 1
    assert "location" in entries[0]
    loc = entries[0]["location"]
    assert loc is not None
    assert loc["lat"] == 37.7749
    assert loc["lng"] == -122.4194
    assert loc["altitude_m"] == 10.0

    # Also simulate a legacy pre-Stage-8 historical row with no location key
    import json
    from app.db import async_session, RiskScoreHistoryRow

    legacy_raw = json.dumps({
        "user_id": user_id,
        "timestamp": ts.isoformat(),
        "risk_tier": "QUIESCENT",
        "risk_score": 10.0,
        "polling_tier": 0,
        # note: NO location key at all
    })

    async def _insert_legacy():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id=user_id,
                    timestamp=ts + timedelta(seconds=1),
                    risk_tier="QUIESCENT",
                    risk_score=10.0,
                    polling_tier=0,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=legacy_raw,
                )
            )
            await session.commit()

    asyncio.run(_insert_legacy())

    hist_resp2 = client.get(f"/api/risk-history/{user_id}")
    assert hist_resp2.status_code == 200
    entries2 = hist_resp2.json()
    assert len(entries2) == 2
    # First entry has location
    assert entries2[0]["location"] is not None
    # Legacy entry gracefully has null location without raising any error
    assert entries2[1]["location"] is None


"""
Stage 0 smoke test — verifies all 5 acceptance criteria.

Run with:  python -m pytest backend/tests/test_stage0.py -v
"""
import asyncio
from datetime import datetime, timezone, timedelta
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


@pytest.fixture(scope="session", autouse=True)
def _db():
    """Initialize tables once for the test session and clean up on exit."""
    asyncio.run(init_db())
    yield
    asyncio.run(dispose_engine())


@pytest.fixture(autouse=True)
def _clean_tables():
    """Clean all tables before each test to ensure complete test isolation."""
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


# ---------------------------------------------------------------
# Criterion 1: Consent persists (tested at API level; criterion 1
#   says "survives a server restart" — in-process we verify the
#   record round-trips through the DB, which is the persistence
#   guarantee. Full kill-and-restart is a manual verification.)
# ---------------------------------------------------------------

CONSENT_BODY = {
    "user_id": "patient_042",
    "guardian_consent": {
        "granted": True,
        "granted_at": "2026-09-17T08:00:00Z",
        "guardian_id": "guardian_001",
    },
    "end_user_consent_applicable": False,
    "end_user_consent": None,
    "data_retention_ack": True,
}


def test_consent_record_and_retrieve(client):
    """Criterion 1: POST consent → GET it back."""
    r = client.post("/api/consent/record", json=CONSENT_BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "recorded"

    r2 = client.get("/api/consent/patient_042")
    assert r2.status_code == 200
    got = r2.json()
    assert got["user_id"] == "patient_042"
    assert got["guardian_consent"]["granted"] is True
    assert got["guardian_consent"]["guardian_id"] == "guardian_001"
    assert got["data_retention_ack"] is True


# ---------------------------------------------------------------
# Criterion 2: 403 without consent
# ---------------------------------------------------------------

TELEMETRY_BODY = {
    "user_id": "unconsented_user",
    "timestamp": "2026-09-17T08:30:00Z",
    "location": {"lat": 51.5, "lng": -0.1},
    "sensor_metrics": {
        "horizontal_accuracy_m": 5.0,
        "speed_mps": 1.2,
        "heading_deg": 90.0,
        "battery_pct": 80,
    },
    "signal_status": {"state": "VALID"},
}


def test_telemetry_rejected_without_consent(client):
    """Criterion 2: ingest without consent → 403."""
    r = client.post("/api/telemetry/ingest", json=TELEMETRY_BODY)
    assert r.status_code == 403
    assert "Consent not granted" in r.json()["detail"]


# ---------------------------------------------------------------
# Criterion 3: 200 with consent, PDR still works on DEGRADED_SIGNAL
# ---------------------------------------------------------------

def test_telemetry_accepted_with_consent_pdr_works(client):
    """Criterion 3: consent granted → ingest succeeds, PDR triggers."""
    # Grant consent first
    consent = {
        "user_id": "patient_pdr",
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-09-17T08:00:00Z",
            "guardian_id": "guardian_002",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    r = client.post("/api/consent/record", json=consent)
    assert r.status_code == 200

    # Send DEGRADED_SIGNAL telemetry with displacement > 27m
    payload = {
        "user_id": "patient_pdr",
        "timestamp": "2026-09-17T08:35:00Z",
        "location": {"lat": 51.5, "lng": -0.1},
        "sensor_metrics": {
            "horizontal_accuracy_m": 25.0,
            "speed_mps": 0.5,
            "heading_deg": 180.0,
            "battery_pct": 60,
        },
        "imu_metrics": {
            "step_count_since_last_gps": 40,
            "net_displacement_m": 30.0,
            "pdr_tier_state": "ZONE_TRANSITION",
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL",
            "degraded_since": "2026-09-17T08:30:00Z",
        },
    }
    r = client.post("/api/telemetry/ingest", json=payload)
    assert r.status_code == 200
    assert r.json()["pdr_tier"] == "ZONE_TRANSITION"


# ---------------------------------------------------------------
# Criterion 4: Dismissal decay-clock (4th dismissal → allowed=False)
# ---------------------------------------------------------------

def test_dismissal_4th_returns_not_allowed():
    """Criterion 4: 4 dismissals in 30 days → 4th returns allowed=False."""
    async def _test():
        from app.state_machines.dismissal_quarantine import (
            get_or_create_cell_state,
            save_cell_state,
        )

        cell_id = "test_cell_criterion4"
        base = datetime(2026, 9, 1, tzinfo=timezone.utc)

        for i in range(3):
            state = await get_or_create_cell_state(cell_id)
            allowed, count = state.register_dismissal(base + timedelta(hours=i))
            await save_cell_state(state)
            assert allowed is True, f"Dismissal {i+1} should be allowed"

        # 4th dismissal
        state = await get_or_create_cell_state(cell_id)
        allowed, count = state.register_dismissal(base + timedelta(hours=3))
        await save_cell_state(state)
        assert allowed is False, "4th dismissal must be disallowed"
        assert count == 3  # still 3 in window (4th was rejected)

        # Verify decay_clock_start is still pinned to the FIRST dismissal
        state2 = await get_or_create_cell_state(cell_id)
        assert state2.decay_clock_start == base
        assert isinstance(state2.decay_clock_start, datetime)
        assert len(state2.dismissal_timestamps) == 3
        assert all(isinstance(ts, datetime) for ts in state2.dismissal_timestamps)

    asyncio.run(_test())


# ---------------------------------------------------------------
# Criterion 5: PDR hysteresis transition sequence
# ---------------------------------------------------------------

def test_pdr_hysteresis_exact_sequence():
    """
    Criterion 5: displacement sequence [10, 30, 25, 60, 49, 40] →
    tiers [INDOOR_PACING, ZONE_TRANSITION, ZONE_TRANSITION,
           UNTRACKED_DISPLACEMENT, UNTRACKED_DISPLACEMENT, ZONE_TRANSITION]
    """
    async def _test():
        from app.state_machines.pdr_hysteresis import (
            PDRTier,
            get_or_create_state,
            save_state,
        )

        user_id = "pdr_test_criterion5"
        displacements = [10, 30, 25, 60, 49, 40]
        expected = [
            PDRTier.INDOOR_PACING,
            PDRTier.ZONE_TRANSITION,
            PDRTier.ZONE_TRANSITION,
            PDRTier.UNTRACKED_DISPLACEMENT,
            PDRTier.UNTRACKED_DISPLACEMENT,
            PDRTier.ZONE_TRANSITION,
        ]

        for i, (d, exp) in enumerate(zip(displacements, expected)):
            state = await get_or_create_state(user_id)
            result = state.update(d)
            await save_state(user_id, state)
            assert result == exp, (
                f"Step {i}: displacement={d}, expected {exp.value}, got {result.value}"
            )

    asyncio.run(_test())

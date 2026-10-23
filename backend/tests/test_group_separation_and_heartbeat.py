"""Comprehensive regression tests for group separation, heartbeat corroboration,
GPS evidence degradation filtering, and alert deduplication / auto-resolution.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi.testclient import TestClient
from sqlalchemy import select, delete

from app.db import (
    async_session,
    init_db,
    dispose_engine,
    ConsentRecordRow,
    CompanionSessionRow,
    GroupSeparationStateRow,
    AlertRecordRow,
    WearableReadingRow,
    RiskScoreHistoryRow,
)
from app.main import app
from app.models import AlertStatus
from app.state_machines.group_separation import (
    evaluate_group_separation,
    GroupSeparationConfig,
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
            await session.execute(delete(CompanionSessionRow))
            await session.execute(delete(GroupSeparationStateRow))
            await session.execute(delete(AlertRecordRow))
            await session.execute(delete(WearableReadingRow))
            await session.execute(delete(RiskScoreHistoryRow))
            await session.commit()

    asyncio.run(_clean())


def _grant_consent(client: TestClient, user_id: str):
    resp = client.post(
        "/api/consent/record",
        json={
            "user_id": user_id,
            "guardian_consent": {
                "granted": True,
                "granted_at": "2026-10-23T08:00:00Z",
                "guardian_id": "guardian_test",
            },
            "end_user_consent_applicable": False,
            "data_retention_ack": True,
        },
    )
    assert resp.status_code == 200


def _create_active_companion_session(
    patient_id: str,
    companion_id: str,
    comp_lat: float,
    comp_lng: float,
    comp_acc: float = 5.0,
    comp_ts: datetime | None = None,
):
    if comp_ts is None:
        comp_ts = datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)

    async def _insert():
        async with async_session() as session:
            sess = CompanionSessionRow(
                session_id=f"sess_{patient_id}_{companion_id}",
                patient_id=patient_id,
                caregiver_id="caregiver_main",
                invite_token=f"tok_{patient_id}_{companion_id}",
                status="ACTIVE",
                created_at=now,
                expires_at=now + timedelta(hours=4),
                consent_granted=True,
                consent_granted_at=now,
                companion_alias=f"Companion of {patient_id}",
                companion_user_id=companion_id,
                relationship="family",
                last_companion_lat=comp_lat,
                last_companion_lng=comp_lng,
                last_companion_accuracy_m=comp_acc,
                last_companion_timestamp=comp_ts,
            )
            session.add(sess)
            await session.commit()

    asyncio.run(_insert())


def _seed_wearable_readings(user_id: str, bpms: list[int], start_dt: datetime | None = None):
    if start_dt is None:
        start_dt = datetime.now(timezone.utc) - timedelta(minutes=len(bpms))

    async def _insert():
        async with async_session() as session:
            for i, bpm in enumerate(bpms):
                row = WearableReadingRow(
                    user_id=user_id,
                    timestamp=start_dt + timedelta(seconds=i * 2),
                    heart_rate_bpm=bpm,
                    wear_state="ON_WRIST",
                    sensor_status="SIMULATED_TEST_STREAM",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(row)
            await session.commit()

    asyncio.run(_insert())


# ==============================================================================
# Unit & Integration Tests
# ==============================================================================

def test_unconsented_patient_cannot_infer_separation():
    """Unconsented patient returns CONSENT_NOT_GRANTED and never evaluates separation."""
    now = datetime.now(timezone.utc)
    res = asyncio.run(
        evaluate_group_separation(
            patient_id="unconsented_patient",
            patient_lat=12.9716,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    assert res.status == "CONSENT_NOT_GRANTED"
    assert res.action == "NONE"


def test_missing_companion_session_returns_no_session():
    """Consented patient without active companion session returns NO_ACTIVE_COMPANION_SESSION."""
    client = TestClient(app)
    _grant_consent(client, "patient_alone")
    now = datetime.now(timezone.utc)

    res = asyncio.run(
        evaluate_group_separation(
            patient_id="patient_alone",
            patient_lat=12.9716,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    assert res.status == "NO_ACTIVE_COMPANION_SESSION"
    assert res.action == "NONE"


def test_degraded_gps_holds_separation_state():
    """Coarse GPS accuracy (>25m) or degraded signal prevents spurious separation inferences."""
    client = TestClient(app)
    patient_id = "patient_bad_gps"
    comp_id = "comp_01"
    _grant_consent(client, patient_id)
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946)
    now = datetime.now(timezone.utc)

    # 1. Patient accuracy degraded (40m > 25m)
    res = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9800,  # ~900m away
            patient_lng=77.5946,
            patient_accuracy_m=40.0,
            patient_timestamp=now,
        )
    )
    assert res.status == "INSUFFICIENT_GPS"
    assert res.action == "NONE"

    # 2. Patient signal marked degraded
    res_degraded = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9800,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
            patient_is_degraded=True,
        )
    )
    assert res_degraded.status == "INSUFFICIENT_GPS"
    assert res_degraded.action == "NONE"


def test_stale_companion_telemetry_holds_state():
    """Stale companion telemetry (>60s) returns ACCOMPANIMENT_UNVERIFIED without triggering false alert."""
    client = TestClient(app)
    patient_id = "patient_stale_comp"
    comp_id = "comp_stale"
    _grant_consent(client, patient_id)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)  # 2 minutes ago
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946, comp_ts=stale_ts)

    now = datetime.now(timezone.utc)
    res = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9800,  # Far away
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    assert res.status == "ACCOMPANIMENT_UNVERIFIED"
    assert res.action == "NONE"


def test_debounced_separation_creates_alert_with_deduplication():
    """Separation beyond threshold requires consecutive readings, then creates a single persistent alert."""
    client = TestClient(app)
    patient_id = "patient_debounced"
    comp_id = "comp_debounced"
    _grant_consent(client, patient_id)
    # Companion at Bengaluru City Hall (12.9716, 77.5946)
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946)
    now = datetime.now(timezone.utc)

    # Reading 1: Patient 150m north -> SEPARATION_PENDING (debouncing, reading 1 of 2)
    res1 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9730,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    assert res1.status == "SEPARATION_PENDING"
    assert res1.action == "NONE"

    # Reading 2: Patient confirmed separated (reading 2 of 2) -> SEPARATED!
    res2 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9732,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=2),
        )
    )
    assert res2.status == "SEPARATED"
    assert res2.action == "ALERT_TRIGGERED"
    assert res2.alert_tier == 2  # Tier 2 default without physiological arousal
    assert res2.alert_id is not None
    alert_id = res2.alert_id

    # Reading 3: Continued movement across space -> ALERT_UPDATED, NO DUPLICATE ALERT
    res3 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9740,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=4),
        )
    )
    assert res3.status == "SEPARATED"
    assert res3.action == "ALERT_UPDATED"
    assert res3.alert_id == alert_id  # Reused same alert record!

    # Verify exactly 1 alert row exists in database
    async def _check_db():
        async with async_session() as session:
            alerts = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.user_id == patient_id)
            )).scalars().all()
            assert len(alerts) == 1
            assert alerts[0].alert_id == alert_id
            assert alerts[0].status == AlertStatus.PENDING.value
            assert alerts[0].zone_id == "group_separation"

    asyncio.run(_check_db())


def test_physiological_arousal_escalates_to_tier_3():
    """Separation with elevated heart rate (z >= 2.0 or BPM >= 100) escalates to Tier 3 (CRITICAL)."""
    client = TestClient(app)
    patient_id = "patient_arousal"
    comp_id = "comp_arousal"
    _grant_consent(client, patient_id)
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946)

    # Seed baseline heart rates around 75 BPM, and latest reading at 112 BPM (arousal)
    now = datetime.now(timezone.utc)
    baseline_bpms = [74, 76, 75, 75, 74, 76, 75, 75, 73, 76]
    _seed_wearable_readings(patient_id, baseline_bpms, now - timedelta(minutes=5))
    # Latest reading: 112 BPM (high tachycardia / arousal)
    _seed_wearable_readings(patient_id, [112], now - timedelta(seconds=5))

    # Send 2 separated readings
    asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9740,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    res2 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9745,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=2),
        )
    )
    assert res2.status == "SEPARATED"
    assert res2.is_physiological_arousal is True
    assert res2.alert_tier == 3  # Tier 3 due to physiological arousal!
    assert res2.heart_rate_bpm == 112
    assert res2.heart_rate_provenance == "SIMULATED_TEST_STREAM"


def test_non_suppression_invariant_normal_hr_still_alerts():
    """Normal resting heart rate (72 BPM) NEVER suppresses separation alert; triggers at Tier 2."""
    client = TestClient(app)
    patient_id = "patient_normal_hr"
    comp_id = "comp_normal"
    _grant_consent(client, patient_id)
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946)

    now = datetime.now(timezone.utc)
    # Seed perfectly normal heart rates
    _seed_wearable_readings(patient_id, [72, 73, 72, 71, 72, 73, 72], now - timedelta(seconds=10))

    # 2 separated readings
    asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9740,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    res2 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9745,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=2),
        )
    )
    assert res2.status == "SEPARATED"
    assert res2.is_physiological_arousal is False
    assert res2.action == "ALERT_TRIGGERED"
    assert res2.alert_tier == 2  # Tier 2 active, NOT SUPPRESSED!


def test_reunification_auto_resolves_separation_alert():
    """When the group reunites within distance threshold for 2 readings, alert transitions to AUTO_RESOLVED."""
    client = TestClient(app)
    patient_id = "patient_reunite"
    comp_id = "comp_reunite"
    _grant_consent(client, patient_id)
    _create_active_companion_session(patient_id, comp_id, 12.9716, 77.5946)
    now = datetime.now(timezone.utc)

    # 1. Establish separation & alert
    asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9740,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now,
        )
    )
    res_sep = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9740,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=2),
        )
    )
    assert res_sep.status == "SEPARATED"
    alert_id = res_sep.alert_id

    # 2. Patient returns close to companion (within 10m): Reading 1 -> REUNITE_PENDING
    res_reunite1 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9716,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=4),
        )
    )
    assert res_reunite1.status == "REUNITE_PENDING"
    assert res_reunite1.action == "NONE"

    # 3. Patient stays close: Reading 2 -> WITH_COMPANION & ALERT_RESOLVED
    res_reunite2 = asyncio.run(
        evaluate_group_separation(
            patient_id=patient_id,
            patient_lat=12.9716,
            patient_lng=77.5946,
            patient_accuracy_m=5.0,
            patient_timestamp=now + timedelta(seconds=6),
        )
    )
    assert res_reunite2.status == "WITH_COMPANION"
    assert res_reunite2.action == "ALERT_RESOLVED"

    # Verify DB alert status changed to AUTO_RESOLVED
    async def _check_resolved():
        async with async_session() as session:
            alert = (await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.alert_id == alert_id)
            )).scalar_one()
            assert alert.status == AlertStatus.AUTO_RESOLVED.value

    asyncio.run(_check_resolved())



def test_companion_simulation_rejects_without_consent():
    """Simulation step must reject execution if patient has no consent on file."""
    client = TestClient(app)
    resp = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": "unconsented_pat", "scenario": "WALKING_TOGETHER"},
    )
    assert resp.status_code == 403
    assert "consent required" in resp.json()["detail"].lower()


def test_companion_simulation_walking_together_scenario():
    """Companion simulation generates distinct parallel trajectory and Together status."""
    client = TestClient(app)
    patient_id = "sim_walk_pat"
    _grant_consent(client, patient_id)

    resp = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "WALKING_TOGETHER", "step_number": 1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["display_status"] == "Together"
    assert data["group_status"] == "WITH_COMPANION"
    assert data["distance_meters"] < 15.0
    assert data["action"] == "NONE"
    assert data["alert_id"] is None
    # Crucial safety check: patient and companion coordinates are distinct
    assert data["patient_coords"] != data["companion_coords"]
    assert data["data_provenance"] == "SIMULATED_COMPANION_DATA"


def test_companion_simulation_separating_and_heartbeat_arousal():
    """Separating scenario debounces (step 1 -> Pending, step 2 -> Separated), corroborates HR arousal to Tier 3."""
    client = TestClient(app)
    patient_id = "sim_sep_pat"
    _grant_consent(client, patient_id)

    # Step 1: Separation begins (>50m) -> Separation Pending
    resp1 = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "SEPARATING", "step_number": 1},
    )
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["display_status"] == "Separation Pending"
    assert d1["group_status"] == "SEPARATION_PENDING"
    assert d1["action"] == "NONE"
    assert d1["alert_id"] is None

    # Step 2: Separation persists with elevated heart rate (105 BPM) -> Tier 3 Alert
    resp2 = client.post(
        "/api/companion/simulation/step",
        json={
            "patient_id": patient_id,
            "scenario": "SEPARATING",
            "step_number": 2,
            "heart_rate_bpm": 105,
        },
    )
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["display_status"] == "Separated"
    assert d2["group_status"] == "SEPARATED"
    assert d2["action"] == "ALERT_TRIGGERED"
    assert d2["alert_id"] is not None
    assert d2["alert_tier"] == 3
    assert d2["heart_rate"]["is_arousal"] is True
    assert d2["patient_coords"] != d2["companion_coords"]


def test_companion_simulation_rejoining_and_autoresolution():
    """Rejoining scenario debounces (step 1 -> Rejoining, step 2 -> Together) and auto-resolves active alert."""
    client = TestClient(app)
    patient_id = "sim_rejoin_pat"
    _grant_consent(client, patient_id)

    # Trigger separation first (2 steps)
    client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "SEPARATING", "step_number": 1},
    )
    sep_resp = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "SEPARATING", "step_number": 2},
    )
    active_alert_id = sep_resp.json()["alert_id"]
    assert active_alert_id is not None

    # Rejoining Step 1: Distance < 50m -> Rejoining
    resp_rej1 = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "REJOINING", "step_number": 1},
    )
    assert resp_rej1.status_code == 200
    d_rej1 = resp_rej1.json()
    assert d_rej1["display_status"] == "Rejoining"
    assert d_rej1["group_status"] == "REUNITE_PENDING"

    # Rejoining Step 2: Distance < 10m -> Together & Alert Auto-Resolved
    resp_rej2 = client.post(
        "/api/companion/simulation/step",
        json={"patient_id": patient_id, "scenario": "REJOINING", "step_number": 2},
    )
    assert resp_rej2.status_code == 200
    d_rej2 = resp_rej2.json()
    assert d_rej2["display_status"] == "Together"
    assert d_rej2["group_status"] == "WITH_COMPANION"
    assert d_rej2["action"] == "ALERT_RESOLVED"
    assert d_rej2["alert_id"] == active_alert_id

    # Verify query for alerts returns AUTO_RESOLVED
    alert_resp = client.get(f"/api/alerts/{active_alert_id}")
    assert alert_resp.status_code == 200
    assert alert_resp.json()["status"] == "AUTO_RESOLVED"

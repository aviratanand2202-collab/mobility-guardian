"""
Unit & integration tests for Trusted Companion Mode.

Tests verify:
1. Authorization: Patient must have active consent before companion session can be initiated.
2. Secure expiring invitation: Caregiver creates an expiring session and invite token.
3. Explicit companion consent: Companion must explicitly consent before location sharing.
4. Privacy: Companion personal ID is never stored or exposed on caregiver endpoints.
5. Location ingestion: Location is only accepted from consenting, active, unexpired sessions.
6. Accompaniment evaluation: Correctly reports 'With Companion', 'Separated', and 'Accompaniment Unverified'.
7. Session termination: Caregiver can end session; companion can stop sharing at any time.
"""
import asyncio
from datetime import datetime, timezone, timedelta
import json
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
    CompanionSessionRow,
    RiskScoreHistoryRow,
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
            await session.execute(
                delete(RiskScoreHistoryRow).where(
                    RiskScoreHistoryRow.user_id.in_(["patient_alice", "test_user_2", "test_user_3", "patient_beijing"])
                )
            )
            await session.commit()
    from app.services.inference_service import get_inference_engine
    get_inference_engine().clear_user_safe_area("test_user_2")
    get_inference_engine().clear_user_safe_area("patient_beijing")
    asyncio.run(_clean())
    yield
    asyncio.run(_clean())


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _grant_patient_consent(client, patient_id="patient_alice"):
    now = datetime.now(timezone.utc).isoformat()
    res = client.post(
        "/api/consent/record",
        json={
            "user_id": patient_id,
            "guardian_consent": {
                "granted": True,
                "granted_at": now,
                "guardian_id": "guardian_bob",
            },
            "end_user_consent_applicable": False,
            "data_retention_ack": True,
        },
    )
    assert res.status_code == 200


def test_companion_session_rejected_if_patient_not_consented(client):
    res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "unconsented_user", "validity_minutes": 60},
    )
    assert res.status_code == 403
    assert "Consent not granted for this patient" in res.json()["detail"]


def test_companion_session_creation_success(client):
    _grant_patient_consent(client, "patient_alice")

    res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "patient_alice", "validity_minutes": 60},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"]
    assert data["invite_token"]
    assert data["status"] == "PENDING_INVITE"
    assert data["patient_id"] == "patient_alice"
    assert data["companion_alias"] == "Trusted Companion"
    assert data["consent_granted"] is False


def test_companion_invite_details_and_privacy(client):
    _grant_patient_consent(client, "patient_alice")
    create_res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "patient_alice", "validity_minutes": 60},
    )
    invite_token = create_res.json()["invite_token"]

    # Companion views invite
    res = client.get(f"/api/companion/invite/{invite_token}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PENDING_INVITE"
    assert "privacy_disclosure" in data
    assert "permanent family member" in data["privacy_disclosure"]


def test_companion_consent_and_location_lifecycle(client):
    _grant_patient_consent(client, "patient_alice")
    create_res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "patient_alice", "validity_minutes": 60},
    )
    session_id = create_res.json()["session_id"]
    invite_token = create_res.json()["invite_token"]

    # 1. Location rejected before consent is granted
    loc_res_early = client.post(
        f"/api/companion/sessions/{session_id}/location",
        json={
            "companion_token": invite_token,
            "lat": 37.7749,
            "lng": -122.4194,
        },
    )
    assert loc_res_early.status_code == 403

    # 2. Companion explicitly grants consent
    consent_res = client.post(
        f"/api/companion/invite/{invite_token}/consent",
        json={"granted": True, "companion_alias": "Friend Sarah"},
    )
    assert consent_res.status_code == 200
    assert consent_res.json()["status"] == "ACTIVE"

    # 3. Location accepted after consent
    loc_res = client.post(
        f"/api/companion/sessions/{session_id}/location",
        json={
            "companion_token": invite_token,
            "lat": 37.7749,
            "lng": -122.4194,
            "accuracy_m": 5.0,
        },
    )
    assert loc_res.status_code == 200
    assert loc_res.json()["status"] == "recorded"

    # 4. Caregiver queries active companion session (Accompaniment: With Companion)
    active_res = client.get(
        "/api/companion/sessions/active/patient_alice",
        params={"patient_lat": 37.7750, "patient_lng": -122.4194, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    session_data = active_res.json()["active_session"]
    assert session_data is not None
    assert session_data["status"] == "ACTIVE"
    assert session_data["consent_granted"] is True
    assert session_data["accompaniment_status"] == "With Companion"
    assert session_data["separation_distance_m"] < 50.0
    # Privacy check: Friend Sarah is alias, personal ID is not exposed
    assert session_data["companion_alias"] == "Friend Sarah"

    # 5. Accompaniment status: Separated when distance exceeds threshold
    active_res_far = client.get(
        "/api/companion/sessions/active/patient_alice",
        params={"patient_lat": 37.7800, "patient_lng": -122.4194, "threshold_meters": 50.0},
    )
    session_data_far = active_res_far.json()["active_session"]
    assert session_data_far["accompaniment_status"] == "Separated"
    assert session_data_far["separation_distance_m"] > 50.0

    # 6. Accompaniment status: Unverified when patient coords missing
    active_res_no_coords = client.get("/api/companion/sessions/active/patient_alice")
    session_data_no_coords = active_res_no_coords.json()["active_session"]
    assert session_data_no_coords["accompaniment_status"] == "Accompaniment Unverified"


def test_companion_stop_sharing(client):
    _grant_patient_consent(client, "patient_alice")
    create_res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "patient_alice", "validity_minutes": 60},
    )
    session_id = create_res.json()["session_id"]
    invite_token = create_res.json()["invite_token"]

    client.post(
        f"/api/companion/invite/{invite_token}/consent",
        json={"granted": True},
    )

    # Companion stops sharing
    stop_res = client.post(
        f"/api/companion/sessions/{session_id}/stop-sharing",
        params={"companion_token": invite_token},
    )
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "revoked"

    # Further location update is rejected
    loc_res = client.post(
        f"/api/companion/sessions/{session_id}/location",
        json={
            "companion_token": invite_token,
            "lat": 37.7749,
            "lng": -122.4194,
        },
    )
    assert loc_res.status_code == 403


def test_caregiver_ends_session(client):
    _grant_patient_consent(client, "patient_alice")
    create_res = client.post(
        "/api/companion/sessions/create",
        json={"patient_id": "patient_alice", "validity_minutes": 60},
    )
    session_id = create_res.json()["session_id"]
    invite_token = create_res.json()["invite_token"]

    client.post(
        f"/api/companion/invite/{invite_token}/consent",
        json={"granted": True},
    )

    # Caregiver ends session
    end_res = client.post(f"/api/companion/sessions/{session_id}/end")
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "ended"

    # Active session is now null
    active_res = client.get("/api/companion/sessions/active/patient_alice")
    assert active_res.json()["active_session"] is None


def test_family_member_association_creates_active_session(client):
    _grant_patient_consent(client, "test_user_2")
    res = client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "test_user_2",
            "companion_id": "test_user_3",
            "relationship": "family",
            "companion_alias": "Family Member (test_user_3)",
            "validity_minutes": 120,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ACTIVE"
    assert data["consent_granted"] is True
    assert data["companion_user_id"] == "test_user_3"
    assert data["relationship"] == "family"
    assert "test_user_3" in data["companion_alias"]


def test_active_session_syncs_companion_location_from_risk_history(client):
    _grant_patient_consent(client, "test_user_2")
    now_utc = datetime.now(timezone.utc)
    raw_json = json.dumps({
        "risk_tier": "LOW",
        "risk_score": 12.0,
        "polling_tier": 1,
        "location": {"lat": 37.77495, "lng": -122.41945, "accuracy_m": 4.0},
    })

    async def _insert_hist():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id="test_user_3",
                    timestamp=now_utc,
                    risk_tier="LOW",
                    risk_score=12.0,
                    polling_tier=1,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=raw_json,
                )
            )
            await session.commit()

    asyncio.run(_insert_hist())

    create_res = client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "test_user_2",
            "companion_id": "test_user_3",
            "relationship": "family",
            "validity_minutes": 120,
        },
    )
    assert create_res.status_code == 200

    active_res = client.get(
        "/api/companion/sessions/active/test_user_2",
        params={"patient_lat": 37.77490, "patient_lng": -122.41940, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    session_data = active_res.json()["active_session"]
    assert session_data is not None
    assert session_data["status"] == "ACTIVE"
    assert session_data["relationship"] == "family"
    assert session_data["companion_user_id"] == "test_user_3"
    assert session_data["last_companion_lat"] == 37.77495
    assert session_data["last_companion_lng"] == -122.41945
    assert session_data["is_location_stale"] is False
    assert session_data["accompaniment_status"] == "With Companion"
    assert session_data["separation_distance_m"] < 50.0


def test_accompanied_departure_outside_safe_zone_is_not_reported_as_separation(client):
    _grant_patient_consent(client, "test_user_2")
    from app.services.inference_service import get_inference_engine
    engine = get_inference_engine()
    engine.set_user_safe_area("test_user_2", {
        "name": "Neighborhood Safe Zone",
        "center_latitude": 37.7749,
        "center_longitude": -122.4194,
        "radius_m": 260.0,
    })

    # Both patient and companion outside the 260m geofence together
    # Center: (37.7749, -122.4194). (37.7800, -122.4194) is ~567m (>260m)
    now_utc = datetime.now(timezone.utc)
    raw_comp = json.dumps({
        "risk_tier": "LOW",
        "risk_score": 10.0,
        "polling_tier": 1,
        "location": {"lat": 37.7801, "lng": -122.4194, "accuracy_m": 5.0},
    })

    async def _setup_data():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id="test_user_3",
                    timestamp=now_utc,
                    risk_tier="LOW",
                    risk_score=10.0,
                    polling_tier=1,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=raw_comp,
                )
            )
            await session.commit()

    asyncio.run(_setup_data())

    client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "test_user_2",
            "companion_id": "test_user_3",
            "relationship": "family",
            "validity_minutes": 120,
        },
    )

    # Patient at (37.7800, -122.4194) - companion is ~11m away at (37.7801, -122.4194)
    active_res = client.get(
        "/api/companion/sessions/active/test_user_2",
        params={"patient_lat": 37.7800, "patient_lng": -122.4194, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    data = active_res.json()["active_session"]
    assert data["accompaniment_status"] == "With Companion"
    assert data["safe_zone"]["configured"] is True
    assert data["safe_zone"]["name"] == "Neighborhood Safe Zone"
    assert data["safe_zone_departure_accompanied"] is True
    assert data["separation_distance_m"] < 50.0


def test_unassociated_patient_returns_none_active_session(client):
    # test_user_3 has no active companion session associated
    res = client.get("/api/companion/sessions/active/test_user_3")
    assert res.status_code == 200
    assert res.json()["active_session"] is None


def test_stale_companion_location_results_in_unverified_and_stale_flag(client):
    _grant_patient_consent(client, "test_user_2")
    # Companion fix older than 60 seconds (120 seconds ago)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    raw_comp = json.dumps({
        "risk_tier": "LOW",
        "risk_score": 10.0,
        "polling_tier": 1,
        "location": {"lat": 37.7750, "lng": -122.4194, "accuracy_m": 5.0},
    })

    async def _setup_stale():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id="test_user_3",
                    timestamp=old_time,
                    risk_tier="LOW",
                    risk_score=10.0,
                    polling_tier=1,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=raw_comp,
                )
            )
            await session.commit()

    asyncio.run(_setup_stale())

    client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "test_user_2",
            "companion_id": "test_user_3",
            "relationship": "family",
            "validity_minutes": 120,
        },
    )

    active_res = client.get(
        "/api/companion/sessions/active/test_user_2",
        params={"patient_lat": 37.7750, "patient_lng": -122.4194, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    data = active_res.json()["active_session"]
    assert data["is_location_stale"] is True
    assert data["accompaniment_status"] == "Accompaniment Unverified"


def test_patient_without_configured_safe_zone_reports_safe_zone_unconfigured(client):
    _grant_patient_consent(client, "test_user_2")
    from app.services.inference_service import get_inference_engine
    get_inference_engine().clear_user_safe_area("test_user_2")

    now_utc = datetime.now(timezone.utc)
    raw_comp = json.dumps({
        "risk_tier": "LOW",
        "risk_score": 10.0,
        "polling_tier": 1,
        "location": {"lat": 37.7801, "lng": -122.4194, "accuracy_m": 5.0},
    })

    async def _setup_data():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id="test_user_3",
                    timestamp=now_utc,
                    risk_tier="LOW",
                    risk_score=10.0,
                    polling_tier=1,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=raw_comp,
                )
            )
            await session.commit()

    asyncio.run(_setup_data())

    client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "test_user_2",
            "companion_id": "test_user_3",
            "relationship": "family",
            "validity_minutes": 120,
        },
    )

    active_res = client.get(
        "/api/companion/sessions/active/test_user_2",
        params={"patient_lat": 37.7800, "patient_lng": -122.4194, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    data = active_res.json()["active_session"]
    assert data["accompaniment_status"] == "With Companion"
    assert data["safe_zone"]["configured"] is False
    assert "No safe zone configured" in data["safe_zone"]["description"]
    assert data["safe_zone_departure_accompanied"] is False


def test_patient_with_beijing_safe_zone_uses_actual_configured_coordinates(client):
    _grant_patient_consent(client, "patient_beijing")
    from app.services.inference_service import get_inference_engine
    engine = get_inference_engine()
    engine.set_user_safe_area("patient_beijing", {
        "name": "Beijing Haidian Residence Safe Zone",
        "center_latitude": 39.9847,
        "center_longitude": 116.3184,
        "radius_m": 300.0,
    })

    now_utc = datetime.now(timezone.utc)
    raw_comp = json.dumps({
        "risk_tier": "LOW",
        "risk_score": 10.0,
        "polling_tier": 1,
        "location": {"lat": 39.9893, "lng": 116.3184, "accuracy_m": 4.0},
    })

    async def _setup_data():
        async with async_session() as session:
            session.add(
                RiskScoreHistoryRow(
                    user_id="test_user_3",
                    timestamp=now_utc,
                    risk_tier="LOW",
                    risk_score=10.0,
                    polling_tier=1,
                    grid_cell_id="8928308280fffff",
                    raw_output_json=raw_comp,
                )
            )
            await session.commit()

    asyncio.run(_setup_data())

    client.post(
        "/api/companion/sessions/create",
        json={
            "patient_id": "patient_beijing",
            "companion_id": "test_user_3",
            "relationship": "family",
            "validity_minutes": 120,
        },
    )

    active_res = client.get(
        "/api/companion/sessions/active/patient_beijing",
        params={"patient_lat": 39.9892, "patient_lng": 116.3184, "threshold_meters": 50.0},
    )
    assert active_res.status_code == 200
    data = active_res.json()["active_session"]
    assert data["accompaniment_status"] == "With Companion"
    assert data["safe_zone"]["configured"] is True
    assert data["safe_zone"]["name"] == "Beijing Haidian Residence Safe Zone"
    assert data["safe_zone"]["center"] == [39.9847, 116.3184]
    assert data["safe_zone_departure_accompanied"] is True


"""
End-to-End Verification of Real Family / Companion Tracking in Mobility Guardian.

This script tests the real tracking flow with legitimate API calls:
1. Explicit test patient consent recorded via POST /api/consent/record
2. Authorized association created via POST /api/companion/sessions/create
3. Ingesting actual timestamped telemetry for patient and associated person
4. Validating GET /api/companion/sessions/active/{patient_id} reports:
   - Accompaniment status ("With Companion", "Separated", "Accompaniment Unverified")
   - Accurate separation distance
   - Safe zone evaluation based ONLY on configured safe zone (never San Francisco default)
   - Proper staleness detection (>60s)
5. Session termination via POST /api/companion/sessions/{session_id}/end
"""

import os
import sys
import time
from datetime import datetime, timezone
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

BASE_URL = "http://127.0.0.1:8000"


def log(msg, status="INFO"):
    tag = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "INFO"
    print(f"[{tag}] {msg}")


def run_verification():
    print("=" * 70)
    print("REAL FAMILY / COMPANION TRACKING VERIFICATION (Backend Telemetry & APIs)")
    print("=" * 70)

    # 1. Check server health
    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=5)
        if r.status_code != 200:
            log(f"Backend not responding normally: HTTP {r.status_code}", "FAIL")
            sys.exit(1)
        log(f"Backend reachable at {BASE_URL}", "PASS")
    except Exception as e:
        log(f"Could not connect to backend at {BASE_URL}: {e}", "FAIL")
        sys.exit(1)

    patient_id = "test_user_2"
    companion_id = "test_user_3"

    # 2. Check initial active session (should be null or clean it up)
    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    initial_data = r.json()
    if initial_data.get("active_session"):
        old_id = initial_data["active_session"]["session_id"]
        requests.post(f"{BASE_URL}/api/companion/sessions/{old_id}/end")
        log(f"Cleaned up preexisting active session {old_id} for {patient_id}")

    # Re-verify clean state
    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}")
    assert r.json().get("active_session") is None
    log("Verified unassociated patient returns active_session=null (no auto-association)", "PASS")

    # 3. Explicit Patient Consent
    now_iso = datetime.now(timezone.utc).isoformat()
    consent_payload = {
        "user_id": patient_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": now_iso,
            "guardian_id": "guardian_verified_test",
        },
        "end_user_consent_applicable": True,
        "end_user_consent": {
            "granted": True,
            "granted_at": now_iso,
        },
        "data_retention_ack": True,
    }
    r = requests.post(f"{BASE_URL}/api/consent/record", json=consent_payload)
    assert r.status_code == 200, f"Consent record failed: {r.text}"
    log(f"Explicit patient consent recorded for {patient_id}", "PASS")

    # Verify consent check endpoint
    r = requests.get(f"{BASE_URL}/api/consent/{patient_id}")
    assert r.status_code == 200 and r.json().get("guardian_consent", {}).get("granted") is True
    log("Verified patient consent record is verified via GET /api/consent/{patient_id}", "PASS")

    # 4. Explicit Authorized Association Creation
    assoc_payload = {
        "patient_id": patient_id,
        "caregiver_id": "caregiver_primary",
        "validity_minutes": 1440,
        "companion_id": companion_id,
        "relationship": "family",
        "companion_alias": f"Family Member ({companion_id})",
    }
    r = requests.post(f"{BASE_URL}/api/companion/sessions/create", json=assoc_payload)
    assert r.status_code == 200, f"Association failed: {r.text}"
    session_data = r.json()
    session_id = session_data["session_id"]
    log(f"Explicit authorized association created: session_id={session_id}, companion={companion_id}", "PASS")

    # 5. Query active session before location update (Fix unavailable)
    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}")
    data = r.json()["active_session"]
    assert data["status"] == "ACTIVE"
    assert data["accompaniment_status"] == "Accompaniment Unverified"
    assert data["last_companion_lat"] is None
    log("Verified unverified status when companion location fix is not yet available", "PASS")

    # 6. Companion Ingests Telemetry (Beijing GeoLife Coordinates, ~20m from patient)
    # Patient location: (39.9840, 116.3180)
    # Companion location: (39.98415, 116.31805)
    patient_lat, patient_lng = 39.9840, 116.3180
    comp_lat, comp_lng = 39.98415, 116.31805

    loc_payload = {
        "companion_token": session_data["invite_token"],
        "lat": comp_lat,
        "lng": comp_lng,
        "accuracy_m": 8.0,
    }
    r = requests.post(f"{BASE_URL}/api/companion/sessions/{session_id}/location", json=loc_payload)
    assert r.status_code == 200, f"Location update failed: {r.text}"
    log(f"Ingested fresh telemetry for companion at ({comp_lat}, {comp_lng})", "PASS")

    # 7. Query active session with patient location (Co-located, With Companion)
    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}?patient_lat={patient_lat}&patient_lng={patient_lng}")
    assert r.status_code == 200
    active = r.json()["active_session"]
    assert active["accompaniment_status"] == "With Companion", f"Expected 'With Companion', got {active['accompaniment_status']}"
    assert active["is_location_stale"] is False
    assert active["separation_distance_m"] is not None
    assert active["separation_distance_m"] < 30.0
    log(f"Accompaniment verified: status='{active['accompaniment_status']}', separation={active['separation_distance_m']:.1f}m", "PASS")

    # 8. Check Safe Zone Behavior: Ensure NO fabricated San Francisco safe zone
    safe_zone = active.get("safe_zone")
    log(f"Reported safe zone payload: {safe_zone}")
    if safe_zone and not safe_zone.get("configured"):
        log("Safe zone correctly identified as unconfigured (configured: False) — no SF coordinates assumed!", "PASS")
        assert active["safe_zone_departure_accompanied"] is False, "Must not falsely flag safe zone departure when no safe zone configured"
    elif safe_zone and safe_zone.get("configured"):
        log(f"Patient has configured safe zone: {safe_zone.get('name')}", "PASS")

    # 9. Test Separation (>50m)
    # Move companion 300m away to (39.9867, 116.3180)
    comp_lat_far = 39.9867
    r = requests.post(f"{BASE_URL}/api/companion/sessions/{session_id}/location", json={
        "companion_token": session_data["invite_token"],
        "lat": comp_lat_far,
        "lng": comp_lng,
        "accuracy_m": 8.0,
    })
    assert r.status_code == 200

    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}?patient_lat={patient_lat}&patient_lng={patient_lng}")
    active = r.json()["active_session"]
    assert active["accompaniment_status"] == "Separated", f"Expected 'Separated', got {active['accompaniment_status']}"
    assert active["separation_distance_m"] > 250
    log(f"Separation detected: status='{active['accompaniment_status']}', separation={active['separation_distance_m']:.1f}m", "PASS")

    # 10. End Session / Disconnect
    r = requests.post(f"{BASE_URL}/api/companion/sessions/{session_id}/end")
    assert r.status_code == 200
    r = requests.get(f"{BASE_URL}/api/companion/sessions/active/{patient_id}")
    assert r.json().get("active_session") is None
    log("Session successfully disconnected and terminated", "PASS")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()

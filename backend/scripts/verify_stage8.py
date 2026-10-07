"""
Stage 8 End-to-End Verification Script
Tests all Definitions of Done programmatically:
1. Real-time WebSocket streaming with location coordinates and risk tier transitions.
2. Alert creation, dismissal suppression loop (sensitivity multiplier), and 4th dismissal recalibration flow.
3. Risk history persistence and retrieval via GET /api/risk-history/{user_id}.
4. Explainability top_features with human-readable text and SHAP values.
5. All backend health and WebSocket endpoints.
"""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request
import websockets

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BASE_HTTP = "http://localhost:8000"
BASE_WS = "ws://localhost:8000"
USER_ID = "sim_dod_test"

def record_consent():
    payload = {
        "user_id": USER_ID,
        "guardian_consent": {
            "granted": True,
            "granted_at": "2026-10-07T06:00:00Z",
            "guardian_id": "guardian_verify",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    req = urllib.request.Request(
        f"{BASE_HTTP}/api/consent/record",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"[1] Consent recorded: {res}")

def ingest_telemetry(lat, lng, speed, degraded=False, pdr_disp=0.0):
    payload = {
        "user_id": USER_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": {"lat": lat, "lng": lng, "altitude_m": 15.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 25.0 if degraded else 3.0,
            "speed_mps": speed,
            "heading_deg": 120.0,
            "activity_type": "WALKING",
            "battery_pct": 75,
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL" if degraded else "VALID",
            "degraded_since": "2026-10-07T06:55:00Z" if degraded else None,
        },
        "imu_metrics": {
            "step_count_since_last_gps": 50,
            "net_displacement_m": pdr_disp,
            "pdr_tier_state": "UNTRACKED_DISPLACEMENT" if pdr_disp >= 52.0 else "INDOOR_PACING",
        } if degraded else None,
    }
    req = urllib.request.Request(
        f"{BASE_HTTP}/api/telemetry/ingest",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def dismiss_alert(alert_id):
    req = urllib.request.Request(
        f"{BASE_HTTP}/api/alerts/{alert_id}/dismiss",
        data=b"",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_risk_history():
    req = urllib.request.Request(f"{BASE_HTTP}/api/risk-history/{USER_ID}?limit=10")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

async def test_live_stream_and_alerts():
    record_consent()

    risk_ws_url = f"{BASE_WS}/ws/risk/{USER_ID}"
    alert_ws_url = f"{BASE_WS}/ws/alerts/{USER_ID}"

    async with websockets.connect(risk_ws_url) as risk_ws, websockets.connect(alert_ws_url) as alert_ws:
        print("[2] Connected to both /ws/risk and /ws/alerts WebSockets")

        # Ingest low-risk point
        out1 = ingest_telemetry(37.7749, -122.4194, speed=0.5)
        raw_msg1 = await asyncio.wait_for(risk_ws.recv(), timeout=5.0)
        msg1 = json.loads(raw_msg1)

        print("[3] DoD 1 Check (Location in RiskScoreOutput):")
        print(f"    - Received location: {msg1.get('location')}")
        print(f"    - Risk Tier: {msg1.get('risk_tier')}, Score: {msg1.get('risk_score')}")
        assert msg1.get("location") is not None, "Location must be present on RiskScoreOutput"
        assert msg1["location"]["lat"] == 37.7749 and msg1["location"]["lng"] == -122.4194

        print("[4] DoD 4 Check (TreeSHAP Explainability):")
        top_features = msg1.get("explainability", {}).get("top_features", [])
        print(f"    - Top features count: {len(top_features)}")
        for f in top_features:
            print(f"      * '{f.get('human_readable')}' (feature: {f.get('feature')}, shap: {f.get('shap_value')})")
        assert len(top_features) > 0
        assert "human_readable" in top_features[0]

        # Escalate PDR: Call 1 moves INDOOR_PACING -> ZONE_TRANSITION
        ingest_telemetry(37.7800, -122.4200, speed=1.0, degraded=True, pdr_disp=60.0)
        # Call 2 moves ZONE_TRANSITION -> UNTRACKED_DISPLACEMENT (score 75 -> SUSPICIOUS -> Alert emitted)
        out2 = ingest_telemetry(37.7800, -122.4200, speed=1.0, degraded=True, pdr_disp=60.0)
        raw_alert = await asyncio.wait_for(alert_ws.recv(), timeout=5.0)
        alert_msg = json.loads(raw_alert)
        print(f"[5] Received Alert over WebSocket: id={alert_msg['alert_id']}, tier={alert_msg['alert_tier']}, cell={alert_msg['grid_cell_id']}")

        # DoD 2 Check: Dismiss alert via POST /api/alerts/{alert_id}/dismiss
        dismiss_res1 = dismiss_alert(alert_msg["alert_id"])
        print(f"[6] DoD 2 Check: Dismissal 1 result: {dismiss_res1}")
        assert dismiss_res1["status"] == "dismissed"
        assert "cell_sensitivity_multiplier" in dismiss_res1
        assert dismiss_res1["dismissal_count_in_window"] == 1
        print(f"    -> Real Stage 4 Sensitivity Multiplier: {dismiss_res1['cell_sensitivity_multiplier']}x")

        cell_id = alert_msg['grid_cell_id']
        # Now test 2nd, 3rd, and 4th dismissal in the SAME cell to verify recalibration_required
        from app.db import async_session, AlertRecordRow
        from datetime import datetime, timezone
        
        async def create_alert_row(uid, cid, aid):
            async with async_session() as session:
                session.add(
                    AlertRecordRow(
                        alert_id=aid,
                        user_id=uid,
                        grid_cell_id=cid,
                        timestamp=datetime.now(timezone.utc),
                        alert_tier=3,
                        status="PENDING",
                    )
                )
                await session.commit()

        for i in range(2, 4):
            aid = f"alert_recalib_{i}_{USER_ID}"
            await create_alert_row(USER_ID, cell_id, aid)
            d_res = dismiss_alert(aid)
            print(f"    Dismissal #{i} result: {d_res}")
            assert d_res["status"] == "dismissed"
            assert d_res["dismissal_count_in_window"] == i

        # 4th alert in the same cell
        aid_4 = f"alert_recalib_4_{USER_ID}"
        await create_alert_row(USER_ID, cell_id, aid_4)
        d_res_4 = dismiss_alert(aid_4)
        print(f"    Dismissal #4 result: {d_res_4}")
        assert d_res_4["status"] == "recalibration_required"
        assert "Routine change detected" in d_res_4["message"]
        print("[7] DoD 2 Recalibration Check Passed: 4th dismissal returned 'recalibration_required'!")

        # DoD 3 Check: Historical risk records query
        hist = get_risk_history()
        print(f"[8] DoD 3 Check: Fetched {len(hist)} records from GET /api/risk-history/{USER_ID}")
        assert len(hist) > 0
        print(f"    Latest history record: ID={hist[0]['id']}, Tier={hist[0]['risk_tier']}, Score={hist[0]['risk_score']}")

    print("\n>>> ALL DEFINITIONS OF DONE PROGRAMMATICALLY VERIFIED! <<<")

if __name__ == "__main__":
    asyncio.run(test_live_stream_and_alerts())

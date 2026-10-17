"""Verify live WebSocket broadcasting on /ws/risk/{user_id}."""
import asyncio
from datetime import datetime, timezone
import json
import urllib.request
import websockets

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
TEST_USER = "ws_test_user"


async def main():
    print(f"Connecting to WebSocket: {WS_URL}/ws/risk/{TEST_USER}...")

    # 1. Register consent for test user
    consent_url = f"{BACKEND_URL}/api/consent/record"
    consent_payload = {
        "user_id": TEST_USER,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_ws_test",
        },
        "end_user_consent_applicable": False,
        "data_retention_ack": True,
    }
    req = urllib.request.Request(
        consent_url,
        data=json.dumps(consent_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(f"Consent recorded: HTTP {resp.status}")

    # 2. Connect to WebSocket
    async with websockets.connect(f"{WS_URL}/ws/risk/{TEST_USER}") as ws:
        print("WebSocket connected successfully!")

        # 3. Post a telemetry reading
        ingest_url = f"{BACKEND_URL}/api/telemetry/ingest"
        telemetry = {
            "user_id": TEST_USER,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": {"lat": 39.9847, "lng": 116.3184, "altitude_m": 150.0},
            "sensor_metrics": {
                "horizontal_accuracy_m": 5.0,
                "speed_mps": 1.2,
                "heading_deg": 90.0,
                "activity_type": "WALKING",
                "battery_pct": 85,
            },
            "signal_status": {"state": "VALID", "degraded_since": None},
            "imu_metrics": None,
        }

        ingest_req = urllib.request.Request(
            ingest_url,
            data=json.dumps(telemetry).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ingest_req) as resp:
            print(f"Telemetry ingested: HTTP {resp.status}")

        # 4. Receive message over WebSocket
        msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
        data = json.loads(msg)
        print("WebSocket message received:")
        print(f"  User ID:    {data.get('user_id')}")
        print(f"  Risk Tier:  {data.get('risk_tier')}")
        print(f"  Risk Score: {data.get('risk_score')}")
        print(f"  Location:   {data.get('location')}")
        assert data.get("user_id") == TEST_USER
        assert data.get("risk_tier") in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
        print("\n[VERIFIED] WebSocket broadcasting functions correctly!")


if __name__ == "__main__":
    asyncio.run(main())

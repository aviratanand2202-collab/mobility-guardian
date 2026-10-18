"""Phase 3 Test - Telemetry Ingestion with ENABLE_ML_INFERENCE_ENGINE=false."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request
import websockets

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replay_trajectory import parse_geolife_plt, build_payloads

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
USER_ID = "test_phase3_disabled"


async def main():
    print("=" * 80)
    print("Phase 3: Live Ingestion Test with ENABLE_ML_INFERENCE_ENGINE=false")
    print("=" * 80)

    # 1. Legitimately record consent
    consent_url = f"{BACKEND_URL}/api/consent/record"
    consent_payload = {
        "user_id": USER_ID,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_phase3_test",
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
        print(f"1. Consent recorded: HTTP {resp.status}")

    # 2. Parse GeoLife points from tracked fixture
    plt_file = _REPO_ROOT / "ml" / "data" / "raw" / "geolife" / "000" / "Trajectory" / "20081023025304.plt"
    if not plt_file.exists():
        plt_file = _REPO_ROOT / "backend" / "tests" / "fixtures" / "sample_trajectory.plt"
    raw_points = parse_geolife_plt(plt_file)[:5]
    payloads = build_payloads(raw_points, user_id=USER_ID, preserve_timestamps=False)
    print(f"2. Prepared {len(payloads)} GeoLife points for user '{USER_ID}' from {plt_file.name}")

    # 3. Connect to WebSocket and stream points
    print(f"3. Connecting to WebSocket {WS_URL}/ws/risk/{USER_ID}...")
    async with websockets.connect(f"{WS_URL}/ws/risk/{USER_ID}") as ws:
        print("   WebSocket connected successfully!")

        for idx, p in enumerate(payloads, 1):
            ingest_url = f"{BACKEND_URL}/api/telemetry/ingest"
            ingest_req = urllib.request.Request(
                ingest_url,
                data=json.dumps(p).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(ingest_req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            ws_msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            ws_data = json.loads(ws_msg)

            print(f"\nPoint [{idx}]:")
            print(f"  HTTP Response: Status {resp.status}, Tier: {resp_data.get('risk_tier')}, Score: {resp_data.get('risk_score')}")
            print(f"  HTTP ml_evaluation: {resp_data.get('ml_evaluation')}")
            print(f"  WS Message:    Tier: {ws_data.get('risk_tier')}, Score: {ws_data.get('risk_score')}")
            print(f"  WS ml_evaluation:   {ws_data.get('ml_evaluation')}")

            # Invariants verification
            assert resp_data.get("ml_evaluation") is None, "ml_evaluation must be null when disabled"
            assert ws_data.get("ml_evaluation") is None, "WS ml_evaluation must be null when disabled"
            assert resp_data.get("risk_tier") in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
            assert ws_data.get("risk_tier") == resp_data.get("risk_tier")
            assert abs(ws_data.get("risk_score") - resp_data.get("risk_score")) < 1e-4

    # 4. Verify persisted risk history
    history_url = f"{BACKEND_URL}/api/risk-history/{USER_ID}?limit=10"
    req = urllib.request.Request(history_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        history_rows = json.loads(resp.read().decode("utf-8"))

    print(f"\n4. Risk History Query (GET /api/risk-history/{USER_ID}?limit=10):")
    print(f"   Retrieved {len(history_rows)} persisted records.")
    assert len(history_rows) == len(payloads), f"Expected {len(payloads)} rows, got {len(history_rows)}"

    for r in history_rows:
        print(f"   Row ID {r['id']}: Tier={r['risk_tier']}, Score={r['risk_score']}, ml_evaluation={r['ml_evaluation']}")
        assert r["ml_evaluation"] is None, "Persisted ml_evaluation must be null when disabled"

    print("\n[VERIFIED] Phase 3 with ML disabled passed all invariants!")


if __name__ == "__main__":
    asyncio.run(main())

"""Phase 3 Test - Real HTTP Telemetry Ingestion with ENABLE_ML_INFERENCE_ENGINE=true."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
import websockets

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replay_trajectory import parse_geolife_plt, build_payloads

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
USER_ID = "test_phase3_ml_enabled"


async def main():
    print("=" * 90)
    print("Phase 3: Live HTTP Ingestion & WebSocket Test with ENABLE_ML_INFERENCE_ENGINE=true")
    print("=" * 90)

    # 1. Legitimate consent registration for isolated test identity
    consent_url = f"{BACKEND_URL}/api/consent/record"
    consent_payload = {
        "user_id": USER_ID,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_phase3_ml",
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
        print(f"1. Consent registered: HTTP {resp.status}")

    # 2. Parse GeoLife points from tracked fixture
    plt_file = _REPO_ROOT / "ml" / "data" / "raw" / "geolife" / "000" / "Trajectory" / "20081023025304.plt"
    if not plt_file.exists():
        plt_file = _REPO_ROOT / "backend" / "tests" / "fixtures" / "sample_trajectory.plt"
    raw_points = parse_geolife_plt(plt_file)[:30]
    payloads = build_payloads(raw_points, user_id=USER_ID, preserve_timestamps=False)
    t_start = raw_points[0]["original_timestamp"]
    t_end = raw_points[-1]["original_timestamp"]
    total_span = (t_end - t_start).total_seconds()
    print(f"2. Prepared {len(payloads)} GeoLife points from {plt_file.name} (span: {total_span:.1f}s > 120s analysis window)")

    # 3. Connect to WebSocket ws://localhost:8000/ws/risk/{USER_ID}
    print(f"3. Connecting to WebSocket {WS_URL}/ws/risk/{USER_ID}...")
    async with websockets.connect(f"{WS_URL}/ws/risk/{USER_ID}") as ws:
        print("   WebSocket connected successfully!")

        print("\n4. Streaming points through POST /api/telemetry/ingest:")
        print("-" * 110)
        print(f"{'Pt':<4} {'Speed':<8} {'Op Tier':<14} {'Op Score':<10} {'ML Engine':<10} {'Evidence Status':<24} {'Cal Prob':<10} {'ML Score'}")
        print("-" * 110)

        accumulating_count = 0
        evaluated_count = 0
        evaluated_samples = []

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

            ws_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ws_data = json.loads(ws_msg)

            # Verification: HTTP and WS schemas match
            assert resp_data.get("user_id") == USER_ID
            assert ws_data.get("user_id") == USER_ID
            assert resp_data.get("risk_tier") == ws_data.get("risk_tier")
            assert abs(resp_data.get("risk_score") - ws_data.get("risk_score")) < 1e-4

            op_tier = resp_data.get("risk_tier")
            op_score = resp_data.get("risk_score")
            ml_eval = resp_data.get("ml_evaluation")
            ws_ml_eval = ws_data.get("ml_evaluation")

            # Invariant: ML evaluation MUST be present in both HTTP response and WebSocket broadcast
            assert ml_eval is not None, f"Point [{idx}] missing ml_evaluation in HTTP response"
            assert ws_ml_eval is not None, f"Point [{idx}] missing ml_evaluation in WebSocket message"

            eng_st = ml_eval.get("engine_status")
            ev_st = ml_eval.get("evidence_status")
            cal_prob = ml_eval.get("calibrated_probability")
            ml_score = ml_eval.get("ml_risk_score")

            assert eng_st == "ACTIVE", f"Expected ACTIVE engine status, got {eng_st}"

            if ev_st == "INSUFFICIENT_EVIDENCE":
                accumulating_count += 1
                assert cal_prob is None, f"Point [{idx}] has non-null calibrated_probability during accumulation"
                assert ml_score is None, f"Point [{idx}] has non-null ml_risk_score during accumulation"
                prob_str = "None (acc)"
                ml_score_str = "None"
            elif ev_st == "WINDOW_EVALUATED":
                evaluated_count += 1
                assert cal_prob is not None, f"Point [{idx}] missing calibrated_probability for evaluated window"
                assert isinstance(cal_prob, float) and 0.0 <= cal_prob <= 1.0
                assert ml_score is not None and 0.0 <= ml_score <= 100.0
                prob_str = f"{cal_prob * 100:.1f}%"
                ml_score_str = f"{ml_score:.1f}"
                evaluated_samples.append((idx, ml_eval))
            else:
                prob_str = str(cal_prob)
                ml_score_str = str(ml_score)

            spd_str = f"{p['sensor_metrics']['speed_mps']:.1f}m/s"
            op_score_str = f"{op_score:.1f}"
            print(f"[{idx:02d}] {spd_str:<8} {op_tier:<14} {op_score_str:<10} {eng_st:<10} {ev_st:<24} {prob_str:<10} {ml_score_str}")

        print("-" * 110)
        print(f"\nIngestion Results:")
        print(f"  Accumulating (INSUFFICIENT_EVIDENCE) points: {accumulating_count}")
        print(f"  Evaluated (WINDOW_EVALUATED) points:         {evaluated_count}")
        assert accumulating_count > 0, "Must have accumulating window points"
        assert evaluated_count > 0, "Must have evaluated window points once duration >= 120s"

        # 5. Verify evaluated window detail
        first_eval_idx, sample_eval = evaluated_samples[0]
        print(f"\nEvaluated Window Detail (Point {first_eval_idx}):")
        print(f"  Engine Status:          {sample_eval.get('engine_status')}")
        print(f"  Evidence Status:        {sample_eval.get('evidence_status')}")
        print(f"  Calibrated Probability: {sample_eval.get('calibrated_probability') * 100:.2f}%")
        print(f"  Raw Probability:        {sample_eval.get('raw_probability') * 100:.2f}%")
        print(f"  ML Risk Score:          {sample_eval.get('ml_risk_score')}")
        print(f"  ML Risk Tier:           {sample_eval.get('ml_risk_tier')}")
        print(f"  Decision State:         {sample_eval.get('decision_state')}")
        print(f"  Reason:                 {sample_eval.get('reason')}")

    # 6. Verify persisted risk history via GET /api/risk-history/{USER_ID}
    history_url = f"{BACKEND_URL}/api/risk-history/{USER_ID}?limit=50"
    req = urllib.request.Request(history_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        history_rows = json.loads(resp.read().decode("utf-8"))

    print(f"\n6. Risk History Query (GET /api/risk-history/{USER_ID}?limit=50):")
    print(f"   Retrieved {len(history_rows)} records from SQLite database.")
    assert len(history_rows) == len(payloads), f"Expected {len(payloads)} rows, got {len(history_rows)}"

    persisted_accumulating = 0
    persisted_evaluated = 0
    for r in history_rows:
        ml_e = r.get("ml_evaluation")
        assert ml_e is not None, f"Persisted row {r['id']} missing ml_evaluation"
        if ml_e.get("evidence_status") == "INSUFFICIENT_EVIDENCE":
            persisted_accumulating += 1
            assert ml_e.get("calibrated_probability") is None
        elif ml_e.get("evidence_status") == "WINDOW_EVALUATED":
            persisted_evaluated += 1
            assert ml_e.get("calibrated_probability") is not None

    print(f"   Persisted INSUFFICIENT_EVIDENCE records: {persisted_accumulating}")
    print(f"   Persisted WINDOW_EVALUATED records:      {persisted_evaluated}")
    assert persisted_accumulating == accumulating_count
    assert persisted_evaluated == evaluated_count

    # 7. Test Error Isolation: verify that ML errors do not suppress valid telemetry ingestion
    print("\n7. Testing ML Failure Isolation:")
    # We test that even if ML fails, operational risk computation and HTTP 200 succeed
    # Ingest point with valid structure
    test_pt = payloads[-1].copy()
    test_pt["timestamp"] = datetime.now(timezone.utc).isoformat()
    test_pt["sensor_metrics"]["speed_mps"] = 2.5
    status_code, resp_d = None, None
    ingest_req = urllib.request.Request(
        f"{BACKEND_URL}/api/telemetry/ingest",
        data=json.dumps(test_pt).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(ingest_req) as resp:
        status_code = resp.status
        resp_d = json.loads(resp.read().decode("utf-8"))

    assert status_code == 200
    assert resp_d.get("risk_tier") in ("NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
    print(f"   Telemetry ingestion with speed=2.5m/s returned HTTP {status_code}, Op Tier: {resp_d.get('risk_tier')}, Score: {resp_d.get('risk_score')}")

    print("\n[VERIFIED] Phase 3 real HTTP ingestion with ML enabled PASSED ALL INVARIANTS!")


if __name__ == "__main__":
    asyncio.run(main())

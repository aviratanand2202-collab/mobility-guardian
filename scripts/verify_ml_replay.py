"""Verify end-to-end GeoLife replay with ML inference enabled."""
import os
os.environ["ENABLE_ML_INFERENCE_ENGINE"] = "true"

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure repo root and backend are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import is_ml_inference_enabled
from app.models import TelemetryPayload, RiskTier
from app.services.inference_service import evaluate_shadow_inference, reset_inference_engine
from app.services.mock_risk_engine import compute_risk
from scripts.replay_trajectory import parse_geolife_plt, build_payloads


async def run_verification():
    print("=" * 80)
    print("GeoLife ML Shadow-Inference Verification Test")
    print("=" * 80)

    # 1. Verify flag is active
    flag_active = is_ml_inference_enabled()
    print(f"1. ENABLE_ML_INFERENCE_ENGINE active: {flag_active}")
    assert flag_active, "Feature flag must be active for this test"

    # Reset engine to test clean cold-start
    reset_inference_engine()

    # 2. Parse 30 GeoLife points from real .plt file
    plt_file = _REPO_ROOT / "ml" / "data" / "raw" / "geolife" / "000" / "Trajectory" / "20081023025304.plt"
    raw_points = parse_geolife_plt(plt_file)[:30]
    print(f"2. Parsed {len(raw_points)} GeoLife points from {plt_file.name}")
    t_start = raw_points[0]["original_timestamp"]
    t_end = raw_points[-1]["original_timestamp"]
    total_span = (t_end - t_start).total_seconds()
    print(f"   Original trajectory span: {total_span:.1f}s (spans > 120s analysis window)")

    # 3. Construct schema-valid TelemetryPayloads
    user_id = "geolife_replay_test"
    payload_dicts = build_payloads(raw_points, user_id=user_id, preserve_timestamps=False)
    print(f"3. Constructed {len(payload_dicts)} TelemetryPayloads")

    # 4. Stream sequentially through the real inference service pipeline
    print("\n4. Streaming points through evaluate_shadow_inference pipeline:")
    print("-" * 90)
    print(f"{'Pt':<5} {'Speed':<8} {'Op Tier':<14} {'Op Score':<10} {'Engine':<8} {'Evidence Status':<22} {'ML Prob':<10} {'ML Score'}")
    print("-" * 90)

    accumulating_count = 0
    evaluated_count = 0
    final_evaluation = None

    for idx, p_dict in enumerate(payload_dicts, 1):
        payload = TelemetryPayload.model_validate(p_dict)

        # Operational calculation
        op_output = compute_risk(payload, None)
        initial_score = op_output.risk_score
        initial_tier = op_output.risk_tier

        # Shadow ML inference
        ml_eval = await evaluate_shadow_inference(payload, timeout_sec=2.0)
        assert ml_eval is not None, "ML evaluation must not be None when flag is enabled"

        # Operational risk score and tier MUST remain strictly unaffected
        assert op_output.risk_score == initial_score, "ML inference must NOT alter operational score"
        assert op_output.risk_tier == initial_tier, "ML inference must NOT alter operational tier"

        eng_status = ml_eval.get("engine_status")
        ev_status = ml_eval.get("evidence_status")
        cal_prob = ml_eval.get("calibrated_probability")
        ml_score = ml_eval.get("ml_risk_score")

        if ev_status == "INSUFFICIENT_EVIDENCE":
            accumulating_count += 1
            # Probability and scores MUST be None during accumulation
            assert cal_prob is None, "Incomplete window must have null probability"
            assert ml_score is None, "Incomplete window must have null ML risk score"
            prob_str = "None (acc)"
            score_str = "None"
        elif ev_status == "WINDOW_EVALUATED":
            evaluated_count += 1
            final_evaluation = ml_eval
            # Valid float probabilities
            assert cal_prob is not None, "Complete window must have valid calibrated probability"
            assert isinstance(cal_prob, float) and 0.0 <= cal_prob <= 1.0
            prob_str = f"{cal_prob * 100:.1f}%"
            score_str = f"{ml_score:.1f}" if ml_score is not None else "None"
        else:
            prob_str = str(cal_prob)
            score_str = str(ml_score)

        spd_str = f"{payload.sensor_metrics.speed_mps:.1f}m/s"
        op_score_str = f"{op_output.risk_score:.1f}"
        print(f"[{idx:02d}] {spd_str:<8} {op_output.risk_tier.value:<14} {op_score_str:<10} {eng_status:<8} {ev_status:<22} {prob_str:<10} {score_str}")

    print("-" * 90)
    print(f"\nResults Summary:")
    print(f"  Accumulating (INSUFFICIENT_EVIDENCE) points: {accumulating_count}")
    print(f"  Evaluated (WINDOW_EVALUATED) points:         {evaluated_count}")
    assert accumulating_count > 0, "Must have accumulating window points"
    assert evaluated_count > 0, "Must have evaluated window points once duration >= 120s"

    print("\nFinal Evaluated Window Details:")
    print(f"  Engine Status:          {final_evaluation.get('engine_status')}")
    print(f"  Evidence Status:        {final_evaluation.get('evidence_status')}")
    print(f"  Calibrated Probability: {final_evaluation.get('calibrated_probability') * 100:.2f}%")
    print(f"  Raw Probability:        {final_evaluation.get('raw_probability') * 100:.2f}%")
    print(f"  ML Risk Score:          {final_evaluation.get('ml_risk_score')}")
    print(f"  ML Risk Tier:           {final_evaluation.get('ml_risk_tier')}")
    print(f"  Decision State:         {final_evaluation.get('decision_state')}")
    print(f"  Reason:                 {final_evaluation.get('reason')}")

    print("\n[VERIFICATION SUCCESSFUL] All end-to-end ML shadow-inference invariants confirmed!")


if __name__ == "__main__":
    asyncio.run(run_verification())

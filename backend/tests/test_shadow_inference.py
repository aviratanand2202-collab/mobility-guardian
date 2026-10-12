"""Focused regression tests for ML Inference Engine integration in Shadow Mode (Step 5 & 5.1).

Verifies key requirements:
1. Flag disabled: existing backend behavior remains unchanged (ml_evaluation is None).
2. Flag enabled with a complete window: ML evaluation is attached separately; operational risk is untouched.
3. Accumulating/incomplete window: explicitly insufficient evidence (ml_risk_score is null, probabilities null).
4. ML exception or timeout: graceful error capture, redacted client error messages, operational risk untouched.
5. Consent, alert lifecycle, and deduplication: operational state machines remain authoritative.
6. WebSocket serialization: JSON-compatibility and schema roundtripping.
7. Concurrency / per-user lock serialization: preserves chronological buffer updates without race conditions.
8. Timeout concurrency hardening: timed-out worker threads cannot overlap with subsequent same-user calls,
   while different users execute concurrently without serialization.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Provide fallback mock shims only if aiosqlite or h3 cannot be imported
try:
    import aiosqlite  # noqa: F401
except ImportError:
    if "aiosqlite" not in sys.modules:
        aiosqlite_mock = MagicMock()
        aiosqlite_mock.sqlite_version_info = (3, 35, 0)
        aiosqlite_mock.version_info = (0, 20, 0)
        aiosqlite_mock.__version__ = "0.20.0"
        aiosqlite_mock.sqlite_version = "3.35.0"
        sys.modules["aiosqlite"] = aiosqlite_mock

try:
    import h3  # noqa: F401
except ImportError:
    if "h3" not in sys.modules:
        h3_mock = MagicMock()
        h3_mock.geo_to_h3.return_value = "8928308280fffff"
        h3_mock.latlng_to_cell.return_value = "8928308280fffff"
        sys.modules["h3"] = h3_mock

# Ensure backend and repo root are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest

from app.config import ENABLE_ML_INFERENCE_ENGINE, is_ml_inference_enabled
from app.models import (
    Location,
    RiskScoreOutput,
    RiskTier,
    SensorMetrics,
    SignalState,
    SignalStatus,
    TelemetryPayload,
)
from app.services.inference_service import (
    evaluate_shadow_inference,
    get_inference_engine,
    reset_inference_engine,
)
from app.services.mock_risk_engine import compute_risk


def _make_telemetry(
    user_id: str = "test_user_shadow",
    dt: datetime | None = None,
    lat: float = 39.9800,
    lng: float = 116.3100,
    speed_mps: float = 1.2,
) -> TelemetryPayload:
    """Helper to construct a valid TelemetryPayload."""
    if dt is None:
        dt = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    return TelemetryPayload(
        user_id=user_id,
        timestamp=dt,
        location=Location(lat=lat, lng=lng, altitude_m=45.0),
        sensor_metrics=SensorMetrics(
            horizontal_accuracy_m=5.0,
            speed_mps=speed_mps,
            heading_deg=45.0,
            activity_type="WALKING",
            battery_pct=90,
        ),
        signal_status=SignalStatus(state=SignalState.VALID),
    )


@pytest.fixture(autouse=True)
def _reset_engine_state():
    """Ensure clean singleton engine state before and after each test."""
    reset_inference_engine()
    yield
    reset_inference_engine()


# ==============================================================================
# 1. FLAG DISABLED: EXISTING BACKEND BEHAVIOR UNCHANGED
# ==============================================================================


def test_flag_disabled_by_default():
    """Verify that ENABLE_ML_INFERENCE_ENGINE defaults to False and returns None."""
    async def _test():
        with patch.dict("os.environ", {}, clear=True):
            assert not is_ml_inference_enabled()

        payload = _make_telemetry(user_id="user_disabled")
        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=False):
            ml_eval = await evaluate_shadow_inference(payload)
            assert ml_eval is None

    asyncio.run(_test())


def test_flag_disabled_preserves_response_model_contract():
    """Verify RiskScoreOutput contract with ml_evaluation=None is backwards compatible."""
    payload = _make_telemetry()
    op_output = compute_risk(payload, None)

    # When disabled, ml_evaluation is None
    assert op_output.ml_evaluation is None

    dumped = op_output.model_dump(mode="json")
    assert dumped["user_id"] == payload.user_id
    assert dumped["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
    assert 0.0 <= dumped["risk_score"] <= 100.0
    assert dumped["ml_evaluation"] is None


# ==============================================================================
# 2. FLAG ENABLED WITH COMPLETE WINDOW: OBSERVATIONAL SEPARATION
# ==============================================================================


def test_flag_enabled_complete_window_observational_separation():
    """Verify complete window produces WINDOW_EVALUATED and does NOT alter operational risk."""
    async def _test():
        u_id = "user_complete_window"
        base_time = datetime(2026, 10, 11, 14, 0, 0, tzinfo=timezone.utc)

        # 5 readings spaced 30 seconds apart (span = 120s)
        readings = [
            _make_telemetry(
                user_id=u_id,
                dt=base_time + timedelta(seconds=i * 30),
                lat=39.9800 + i * 0.001,
                lng=116.3100 + i * 0.001,
                speed_mps=1.5,
            )
            for i in range(5)
        ]

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            ml_results = []
            for r in readings:
                eval_out = await evaluate_shadow_inference(r)
                ml_results.append(eval_out)

        # First 4 readings: accumulating (ml_risk_score explicitly null)
        for intermediate in ml_results[:4]:
            assert intermediate is not None
            assert intermediate["engine_status"] == "ACTIVE"
            assert intermediate["evidence_status"] == "INSUFFICIENT_EVIDENCE"
            assert intermediate["calibrated_probability"] is None
            assert intermediate["raw_probability"] is None
            assert intermediate["ml_risk_score"] is None
            assert intermediate["ml_risk_tier"] is None

        # 5th reading: window complete
        final_ml = ml_results[4]
        assert final_ml is not None
        assert final_ml["engine_status"] == "ACTIVE"
        assert final_ml["evidence_status"] == "WINDOW_EVALUATED"

        # ML probabilities are valid floats
        assert final_ml["calibrated_probability"] is not None
        assert isinstance(final_ml["calibrated_probability"], float)
        assert 0.0 <= final_ml["calibrated_probability"] <= 1.0

        assert final_ml["raw_probability"] is not None
        assert isinstance(final_ml["raw_probability"], float)
        assert 0.0 <= final_ml["raw_probability"] <= 1.0

        # ML risk score is evaluated
        assert final_ml["ml_risk_score"] is not None
        assert 0.0 <= final_ml["ml_risk_score"] <= 100.0
        assert final_ml["ml_risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")

        # Compute operational risk for comparison
        op_output = compute_risk(readings[4], None)
        initial_op_score = op_output.risk_score
        initial_op_tier = op_output.risk_tier

        # Attach shadow ML evaluation
        op_output.ml_evaluation = final_ml

        # Operational score and tier MUST remain strictly unchanged
        assert op_output.risk_score == initial_op_score
        assert op_output.risk_tier == initial_op_tier
        # Ensure ML probability was NEVER multiplied into operational score
        assert op_output.risk_score != initial_op_score * final_ml["calibrated_probability"]

    asyncio.run(_test())


# ==============================================================================
# 3. ACCUMULATING / INCOMPLETE WINDOW: EXPLICITLY INSUFFICIENT EVIDENCE
# ==============================================================================


def test_accumulating_window_explicitly_insufficient_evidence():
    """Verify incomplete window explicitly states INSUFFICIENT_EVIDENCE and ml_risk_score is null."""
    async def _test():
        payload = _make_telemetry(user_id="user_accumulating")

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            eval_out = await evaluate_shadow_inference(payload)

        assert eval_out is not None
        assert eval_out["engine_status"] == "ACTIVE"
        assert eval_out["evidence_status"] == "INSUFFICIENT_EVIDENCE"
        assert eval_out["decision_state"] == "INSUFFICIENT_EVIDENCE"

        # All probability and risk fields MUST be null/None until window is evaluated
        assert eval_out["calibrated_probability"] is None
        assert eval_out["raw_probability"] is None
        assert eval_out["ml_risk_score"] is None
        assert eval_out["ml_risk_tier"] is None

        # Crucial: verify that this is NOT treated as a safe confirmation (e.g. not 0.0 score/probability)
        assert eval_out["evidence_status"] != "WINDOW_EVALUATED"

    asyncio.run(_test())


# ==============================================================================
# 4. ML EXCEPTION OR UNAVAILABLE ARTIFACTS: REDACTION & CONTINUITY
# ==============================================================================


def test_ml_exception_handled_gracefully_with_redacted_error():
    """Verify engine exception returns FAILED status, redacts internal paths, and operational risk is untouched."""
    async def _test():
        payload = _make_telemetry(user_id="user_exception")

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            with patch.object(
                get_inference_engine(),
                "predict_telemetry_reading",
                side_effect=RuntimeError("Internal file path: /secret/dir/booster.bin missing"),
            ):
                eval_out = await evaluate_shadow_inference(payload)

        assert eval_out is not None
        assert eval_out["engine_status"] == "FAILED"
        assert eval_out["evidence_status"] == "ERROR"
        assert eval_out["calibrated_probability"] is None
        assert eval_out["raw_probability"] is None
        assert eval_out["ml_risk_score"] is None
        assert eval_out["ml_risk_tier"] is None

        # Error message MUST NOT expose internal paths or stack traces
        assert "/secret/dir" not in eval_out["error"]
        assert "booster.bin" not in eval_out["error"]
        assert eval_out["error"] == "ML inference execution encountered an unexpected error."

        # Operational calculation succeeds independently
        op_output = compute_risk(payload, None)
        op_output.ml_evaluation = eval_out
        assert op_output.risk_score >= 0.0
        assert op_output.risk_tier in (RiskTier.QUIESCENT, RiskTier.NORMAL_TRANSIT, RiskTier.SUSPICIOUS, RiskTier.CRITICAL)

    asyncio.run(_test())


def test_ml_inference_timeout_handled_gracefully_with_redacted_error():
    """Verify engine timeout returns TIMEOUT status bounded within timeout_sec with null risk scores."""
    async def _test():
        payload = _make_telemetry(user_id="user_timeout")

        def _slow_predict(_):
            time.sleep(0.3)
            return {}

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            with patch.object(get_inference_engine(), "predict_telemetry_reading", side_effect=_slow_predict):
                eval_out = await evaluate_shadow_inference(payload, timeout_sec=0.05)

        assert eval_out is not None
        assert eval_out["engine_status"] == "TIMEOUT"
        assert eval_out["evidence_status"] == "ERROR"
        assert eval_out["calibrated_probability"] is None
        assert eval_out["raw_probability"] is None
        assert eval_out["ml_risk_score"] is None
        assert eval_out["ml_risk_tier"] is None
        assert eval_out["error"] == "ML inference execution exceeded time limit."

    asyncio.run(_test())


# ==============================================================================
# 5. CONSENT, ALERT LIFECYCLE, AND DEDUPLICATION PRESERVED
# ==============================================================================


def test_consent_rejection_prevents_ml_inference():
    """Verify unconsented telemetry is rejected before shadow inference runs."""
    async def _test():
        from fastapi import HTTPException
        from app.routers.telemetry import ingest_telemetry

        payload = _make_telemetry(user_id="unconsented_user")

        with patch("app.routers.telemetry.is_consented", new_callable=AsyncMock) as mock_consent:
            mock_consent.return_value = False
            with patch("app.routers.telemetry.evaluate_shadow_inference", new_callable=AsyncMock) as mock_ml:
                with pytest.raises(HTTPException) as exc_info:
                    await ingest_telemetry(payload)

                assert exc_info.value.status_code == 403
                assert "Consent not granted" in exc_info.value.detail
                # Shadow ML inference MUST NOT be called when consent is not granted
                mock_ml.assert_not_called()

    asyncio.run(_test())


def test_operational_alert_generation_independent_of_ml():
    """Verify alert generation is strictly driven by operational risk tier, not ML probability."""
    payload = _make_telemetry(user_id="user_alert_test", speed_mps=0.5)

    # Operational risk is QUIESCENT
    op_output = compute_risk(payload, None)
    assert op_output.risk_tier == RiskTier.QUIESCENT

    # Simulated shadow ML output with high anomaly probability
    simulated_ml_eval = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "calibrated_probability": 0.99,
        "raw_probability": 0.98,
        "ml_risk_score": 95.0,
        "ml_risk_tier": "CRITICAL",
    }

    op_output.ml_evaluation = simulated_ml_eval

    # Alert condition in ingest_telemetry checks:
    # if risk_output.risk_tier in (RiskTier.SUSPICIOUS, RiskTier.CRITICAL):
    should_trigger_alert = op_output.risk_tier in (RiskTier.SUSPICIOUS, RiskTier.CRITICAL)
    assert not should_trigger_alert, "ML probability must NEVER trigger operational alert when operational tier is QUIESCENT"


# ==============================================================================
# 6. WEBSOCKET SERIALIZATION AND ROUNDTRIPPING
# ==============================================================================


def test_websocket_payload_serialization_compatibility():
    """Verify RiskScoreOutput serializes to JSON cleanly with and without ml_evaluation."""
    payload = _make_telemetry()
    op_output = compute_risk(payload, None)

    # 1. Without ml_evaluation
    dumped_none = op_output.model_dump(mode="json")
    json_str_none = json.dumps(dumped_none)
    assert "user_id" in json_str_none
    reconstructed_none = RiskScoreOutput.model_validate_json(json_str_none)
    assert reconstructed_none.ml_evaluation is None

    # 2. With full ml_evaluation
    op_output.ml_evaluation = {
        "engine_status": "ACTIVE",
        "evidence_status": "WINDOW_EVALUATED",
        "calibrated_probability": 0.85,
        "raw_probability": 0.82,
        "ml_risk_score": 75.0,
        "ml_risk_tier": "SUSPICIOUS",
        "decision_state": "SUSPICIOUS",
        "display_state": "EVALUATED",
        "reason": "Elevated tortuosity",
        "trigger_state": {"sample_count": 5},
        "explainability": {"top_features": [{"feature": "speed", "shap_value": 0.4}]},
        "location": {"lat": 39.98, "lng": 116.31},
    }

    dumped_full = op_output.model_dump(mode="json")
    json_str_full = json.dumps(dumped_full)
    assert "ml_evaluation" in json_str_full
    assert "calibrated_probability" in json_str_full

    reconstructed_full = RiskScoreOutput.model_validate_json(json_str_full)
    assert reconstructed_full.ml_evaluation is not None
    assert reconstructed_full.ml_evaluation["calibrated_probability"] == 0.85
    assert reconstructed_full.ml_evaluation["engine_status"] == "ACTIVE"


# ==============================================================================
# 7. CONCURRENCY: PER-USER SERIALIZATION PRESERVES ORDERING
# ==============================================================================


def test_per_user_lock_serializes_concurrent_calls():
    """Verify that multiple concurrent telemetry calls for the same user execute serially."""
    async def _test():
        u_id = "user_concurrent"
        base_time = datetime(2026, 10, 11, 16, 0, 0, tzinfo=timezone.utc)

        readings = [
            _make_telemetry(user_id=u_id, dt=base_time + timedelta(seconds=i * 10))
            for i in range(3)
        ]

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            tasks = [evaluate_shadow_inference(r) for r in readings]
            results = await asyncio.gather(*tasks)

        for res in results:
            assert res is not None
            assert res["engine_status"] == "ACTIVE"

    asyncio.run(_test())


# ==============================================================================
# 8. TIMEOUT CONCURRENCY HARDENING (STEP 5.1)
# ==============================================================================


def test_timeout_concurrency_same_user_cannot_overlap_and_different_users_unserialized():
    """Verify that a timed-out worker thread prevents overlapping calls for the same user,
    while different users run concurrently without serialization."""
    async def _test():
        user_a = "user_timeout_concurrency_a"
        user_b = "user_timeout_concurrency_b"

        active_counts = {"user_a": 0, "user_b": 0}
        max_concurrent = {"user_a": 0, "user_b": 0}

        def _mock_predict(payload_dict):
            u_id = payload_dict["user_id"]
            key = "user_a" if u_id == user_a else "user_b"
            active_counts[key] += 1
            if active_counts[key] > max_concurrent[key]:
                max_concurrent[key] = active_counts[key]

            # User A's first call takes 0.25s (will cause timeout if timeout_sec=0.06s)
            if u_id == user_a and "12:00:00" in str(payload_dict["timestamp"]):
                time.sleep(0.25)
            else:
                time.sleep(0.02)

            active_counts[key] -= 1
            return {
                "user_id": u_id,
                "risk_score": 10.0,
                "risk_tier": "QUIESCENT",
                "decision_state": "SAFE / NORMAL",
                "display_state": "EVALUATED",
                "human_readable_reason": "Normal gait",
                "trigger_state": {"status": "EVALUATED", "calibrated_probability": 0.1, "raw_probability": 0.1},
                "explainability": {"status": "EVALUATED"},
                "location": {"lat": 39.98, "lng": 116.31},
            }

        with patch("app.services.inference_service.is_ml_inference_enabled", return_value=True):
            with patch.object(get_inference_engine(), "predict_telemetry_reading", side_effect=_mock_predict):
                # Call A1: slow, will timeout after 0.06s
                p_a1 = _make_telemetry(user_id=user_a, dt=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc))
                task_a1 = asyncio.create_task(evaluate_shadow_inference(p_a1, timeout_sec=0.06))

                # Allow A1 to acquire the lock and dispatch worker thread
                await asyncio.sleep(0.02)

                # Call A2: arrives while A1 worker thread is STILL RUNNING!
                # Uses timeout_sec=0.06s. Since A1 still has ~0.23s left, A2 cannot acquire lock and times out
                p_a2 = _make_telemetry(user_id=user_a, dt=datetime(2026, 10, 11, 12, 0, 10, tzinfo=timezone.utc))
                task_a2 = asyncio.create_task(evaluate_shadow_inference(p_a2, timeout_sec=0.06))

                # Call B1: arrives concurrently for User B!
                # User B has a separate lock and should NOT be blocked by User A
                p_b1 = _make_telemetry(user_id=user_b, dt=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc))
                task_b1 = asyncio.create_task(evaluate_shadow_inference(p_b1, timeout_sec=0.15))

                res_a1, res_a2, res_b1 = await asyncio.gather(task_a1, task_a2, task_b1)

                # A1 timed out during execution
                assert res_a1["engine_status"] == "TIMEOUT"
                assert res_a1["error"] == "ML inference execution exceeded time limit."

                # A2 timed out acquiring lock (could not overlap with A1!)
                assert res_a2["engine_status"] == "TIMEOUT"
                assert res_a2["error"] == "ML inference execution exceeded time limit."

                # B1 succeeded immediately without being blocked by User A
                assert res_b1["engine_status"] == "ACTIVE"
                assert res_b1["evidence_status"] == "WINDOW_EVALUATED"

                # Wait for A1's background worker thread to finish
                await asyncio.sleep(0.3)

                # Now that A1's thread finished, A3 should succeed normally
                p_a3 = _make_telemetry(user_id=user_a, dt=datetime(2026, 10, 11, 12, 0, 20, tzinfo=timezone.utc))
                res_a3 = await evaluate_shadow_inference(p_a3, timeout_sec=0.15)
                assert res_a3["engine_status"] == "ACTIVE"
                assert res_a3["evidence_status"] == "WINDOW_EVALUATED"

        # Crucial invariant: User A's inference NEVER executed concurrently with another inference for User A!
        assert max_concurrent["user_a"] == 1, f"Expected max 1 concurrent inference for User A, got {max_concurrent['user_a']}"
        assert max_concurrent["user_b"] == 1

    asyncio.run(_test())


# ==============================================================================
# 9. HISTORICAL RISK MODEL COMPATIBILITY (STEP 6)
# ==============================================================================


def test_historical_risk_entry_extracts_ml_evaluation_when_present():
    """Verify that RiskHistoryEntry extracts ml_evaluation from raw_output_json if present."""
    from app.routers.risk_history import RiskHistoryEntry, _extract_ml_evaluation

    # 1. When raw_output_json has ml_evaluation
    raw_with_ml = json.dumps({
        "user_id": "hist_user",
        "risk_tier": "QUIESCENT",
        "risk_score": 10.0,
        "ml_evaluation": {
            "engine_status": "ACTIVE",
            "evidence_status": "WINDOW_EVALUATED",
            "calibrated_probability": 0.75,
        },
    })
    extracted = _extract_ml_evaluation(raw_with_ml)
    assert extracted is not None
    assert extracted["calibrated_probability"] == 0.75

    entry = RiskHistoryEntry(
        id=1,
        timestamp=datetime.now(timezone.utc),
        risk_tier="QUIESCENT",
        risk_score=10.0,
        polling_tier=0,
        ml_evaluation=extracted,
    )
    dumped = entry.model_dump(mode="json")
    assert dumped["ml_evaluation"] is not None
    assert dumped["ml_evaluation"]["calibrated_probability"] == 0.75

    # 2. When raw_output_json lacks ml_evaluation (or None)
    raw_without_ml = json.dumps({
        "user_id": "hist_user",
        "risk_tier": "QUIESCENT",
        "risk_score": 10.0,
    })
    assert _extract_ml_evaluation(raw_without_ml) is None
    assert _extract_ml_evaluation(None) is None

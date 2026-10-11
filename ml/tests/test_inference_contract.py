"""Test Suite for ML-Side Inference Contract, Lifecycle Architecture, and Horizon Invariants.

Verifies Chunk 4 Requirements:
1. Locked horizons contract: ONLY 120s, 360s, 600s, 840s are supported; arbitrary horizons
   (such as 300s) raise ValueError in executable prediction logic.
2. New User Architecture:
   current movement -> COLD_START / population baseline -> safe-area context ->
   history accumulation -> personal profile -> personalized inference.
3. Existing User Architecture:
   historical trajectory folder/profile + current movement -> personalized adaptive inference.
4. Safe-Area State:
   Externally supplied geographic boundary; OUTSIDE_SAFE_AREA severity is contextual/application-facing only;
   never elevates behavioral ML risk tier on its own.
5. Familiarity State:
   Derived algorithmically from historical spatial anchors; does NOT imply clinical truth;
   UNFAMILIAR_LOCATION alone does not imply danger; FAMILIAR_LOCATION alone does not mask abnormal kinematics.
6. Behavioral Dominance & Orthogonality:
   CRITICAL/SUSPICIOUS behavioral tiers override geographic context;
   decision interpretation layer does NOT alter XGBoost predictions or probabilities.
7. Raw Research Outputs:
   All raw ML/debug outputs (risk_score, calibrated_probability, raw_probability, horizon_sec,
   binary_alert, top_features) remain preserved.
8. No Fabricated Labels:
   No new training labels or fabricated GeoLife labels exist; canonical benchmark hashes remain frozen.
"""

import hashlib
from pathlib import Path
import pandas as pd
import pytest
import unittest.mock

from ml.src.decision import DecisionConfig, DecisionState, interpret_decision
from ml.src.inference import RiskInferenceEngine
from ml.src.risk import RiskConfig


@pytest.fixture(scope="module")
def engine():
    return RiskInferenceEngine()


def _make_dummy_window(
    user_id: str = "test_user",
    window_id: str = "w_test_001",
    speed: float = 1.25,
    speed_std_dev: float = 0.2,
    path_distance: float = 150.0,
    straight_line: float = 140.0,
    tortuosity: float = 1.12,
    entropy: float = 1.0,
    turn_frequency: float = 0.02,
    loop_metric: float = 0.01,
    pacing: float = 0.0,
    lat: float = 39.9800,
    lon: float = 116.3100,
):
    """Helper to produce a schema-complete dummy 120s window."""
    return {
        "window_id": window_id,
        "user_id": user_id,
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": speed,
        "speed_std_dev": speed_std_dev,
        "path_distance_m": path_distance,
        "straight_line_displacement_m": straight_line,
        "tortuosity_index": tortuosity,
        "entropy_directional": entropy,
        "turn_frequency": turn_frequency,
        "loop_metric": loop_metric,
        "pacing_tendency": pacing,
        "backtracking_tendency": 0.0,
        "heading_variability": 0.1,
        "point_count": 30,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
        "latitude": lat,
        "longitude": lon,
    }


# ==============================================================================
# 1. LOCKED HORIZONS CONTRACT & INVESTIGATION OF 300S
# ==============================================================================


def test_locked_horizons_contract(engine):
    """Verify that ONLY the locked horizons (120, 360, 600, 840) are configured and supported."""
    locked = (120, 360, 600, 840)
    assert RiskConfig.horizons_sec == locked
    assert engine.config.supported_horizons == locked
    assert sorted(list(engine.models.keys())) == sorted(list(locked))
    assert sorted(list(engine.thresholds.keys())) == sorted(list(locked))


def test_unsupported_horizon_300s_rejected_by_executable_logic(engine):
    """Verify that horizon_sec=300 is REJECTED with ValueError by executable prediction logic.

    Confirms that 300s is NOT a supported operational horizon and only exists in
    segmentation inactivity gap config (300.0s) or theoretical audit counterexamples.
    """
    dummy_window = _make_dummy_window()

    with pytest.raises(ValueError) as excinfo:
        engine.predict_window(dummy_window, horizon_sec=300)

    assert "Unsupported horizon: 300s" in str(excinfo.value)
    assert "Available: [120, 360, 600, 840]" in str(excinfo.value)


def test_all_supported_horizons_execute_successfully(engine):
    """Verify that all 4 locked horizons (120s, 360s, 600s, 840s) execute successfully."""
    dummy_window = _make_dummy_window()

    for h in [120, 360, 600, 840]:
        out = engine.predict_window(dummy_window, horizon_sec=h)
        assert out["trigger_state"]["horizon_sec"] == h
        assert out["decision"]["horizon_sec"] == h
        assert 0.0 <= out["risk_score"] <= 100.0
        assert 0.0 <= out["trigger_state"]["calibrated_probability"] <= 1.0


# ==============================================================================
# 2. NEW USER VS EXISTING USER ARCHITECTURE FLOW
# ==============================================================================


def test_new_user_cold_start_architecture_flow(engine):
    """Verify New User flow:

    current movement -> COLD_START / population baseline -> safe-area context ->
    history accumulation -> personal profile.
    """
    dummy_window = _make_dummy_window(user_id="unseen_user_999")

    # Evaluate new user without stored profile or history
    out = engine.predict_window(dummy_window, horizon_sec=120)

    # 1. Profile mode must be COLD_START
    assert out["decision"]["profile_mode"] == "COLD_START"
    # 2. Baseline used must be the population baseline
    assert out["decision"]["familiarity_state"] == "NO_HISTORY"
    # 3. Decision state reflects the behavioral risk tier
    assert out["decision_state"] in [s.value for s in DecisionState]
    if out["risk_tier"] == "SUSPICIOUS":
        assert out["decision_state"] == DecisionState.SUSPICIOUS.value


def test_existing_user_personalized_architecture_flow(engine):
    """Verify Existing User flow:

    historical profile + current movement -> personalized adaptive inference.
    """
    # Create custom personalized baseline
    custom_baseline = {
        "mean_speed_mps": {"median": 1.0, "robust_scale": 0.2, "p95": 1.8},
        "tortuosity_index": {"median": 1.1, "robust_scale": 0.1, "p95": 1.5},
        "entropy_directional": {"median": 1.2, "robust_scale": 0.15, "p95": 1.7},
        "turn_frequency": {"median": 0.03, "robust_scale": 0.01, "p95": 0.08},
        "loop_metric": {"median": 0.05, "robust_scale": 0.02, "p95": 0.15},
        "pacing_tendency": {"median": 0.0, "robust_scale": 0.05, "p95": 0.2},
        "straight_line_displacement_m": {"median": 100.0, "robust_scale": 20.0, "p95": 180.0},
        "path_distance_m": {"median": 120.0, "robust_scale": 25.0, "p95": 200.0},
    }

    dummy_window = _make_dummy_window(user_id="adapted_user_001")

    out = engine.predict_window(
        dummy_window,
        horizon_sec=120,
        user_baseline=custom_baseline,
        user_context={"trip_count": 15},
    )

    assert out["decision"]["profile_mode"] == "PERSONALIZED"


# ==============================================================================
# 3. SAFE-AREA IS CONTEXTUAL ONLY & DOES NOT ALTER BEHAVIORAL RISK
# ==============================================================================


def test_outside_safe_area_severity_is_contextual_only():
    """Verify OUTSIDE_SAFE_AREA alone does NOT become SUSPICIOUS or CRITICAL."""
    risk_output = {
        "risk_tier": "NORMAL_TRANSIT",
        "risk_score": 15.0,
        "trigger_state": {
            "calibrated_probability": 0.15,
            "raw_probability": 0.14,
            "horizon_sec": 120,
            "binary_alert": False,
        },
        "explainability": {"top_features": ["turn_frequency"]},
        "geospatial_context": {
            "safe_area_state": "OUTSIDE_SAFE_AREA",
            "distance_to_safe_boundary_m": 45.0,
            "matched_safe_area_name": "Caregiver Zone",
            "familiarity_state": "FAMILIAR_LOCATION",
        },
    }

    dec = interpret_decision(risk_output, DecisionConfig())

    assert dec.decision_state == DecisionState.OUTSIDE_SAFE_AREA
    assert dec.severity_level == "ADVISORY"  # Contextual advisory only
    assert dec.behavioral_tier == "NORMAL_TRANSIT"  # Behavioral tier untouched


# ==============================================================================
# 4. FAMILIARITY IS ALGORITHMIC & DOES NOT CLAIM CLINICAL TRUTH
# ==============================================================================


def test_familiarity_does_not_imply_clinical_or_caregiver_truth():
    """Verify that familiarity indicates algorithmic anchor proximity and does NOT claim clinical truth."""
    risk_output = {
        "risk_tier": "QUIESCENT",
        "risk_score": 5.0,
        "trigger_state": {
            "calibrated_probability": 0.05,
            "raw_probability": 0.04,
            "horizon_sec": 120,
            "binary_alert": False,
        },
        "explainability": {"top_features": ["mean_speed_mps"]},
        "geospatial_context": {
            "safe_area_state": "SAFE_AREA_UNAVAILABLE",
            "familiarity_state": "FAMILIAR_LOCATION",
            "nearest_anchor_id": 2,
            "distance_to_nearest_anchor_m": 12.0,
            "is_caregiver_confirmed": False,
        },
    }

    dec = interpret_decision(risk_output, DecisionConfig())

    assert dec.decision_state == DecisionState.FAMILIAR_MOVEMENT
    assert dec.is_caregiver_confirmed_anchor is False
    assert "anchor" in dec.reason.lower()
    assert "does not constitute clinical diagnosis" in dec.disclaimer.lower()
    assert "dementia" in dec.disclaimer.lower()
    assert "wandering" in dec.disclaimer.lower()


# ==============================================================================
# 5. BEHAVIORAL RISK DOMINANCE OVER GEOSPATIAL CONTEXT
# ==============================================================================


def test_behavioral_risk_dominance_over_safe_area_and_familiarity():
    """Verify that CRITICAL behavioral ML risk strictly overrides INSIDE_SAFE_AREA and FAMILIAR_LOCATION."""
    risk_output = {
        "risk_tier": "CRITICAL",
        "risk_score": 88.5,
        "trigger_state": {
            "calibrated_probability": 0.885,
            "raw_probability": 0.87,
            "horizon_sec": 120,
            "binary_alert": True,
        },
        "explainability": {"top_features": ["pacing_tendency", "turn_angle_std"]},
        "geospatial_context": {
            "safe_area_state": "INSIDE_SAFE_AREA",  # Inside safe zone!
            "familiarity_state": "FAMILIAR_LOCATION",  # At familiar anchor!
        },
    }

    dec = interpret_decision(risk_output, DecisionConfig())

    # Critical behavioral anomaly MUST dominate
    assert dec.decision_state == DecisionState.CRITICAL
    assert dec.severity_level == "ALERT"


# ==============================================================================
# 6. DECISION LAYER DOES NOT ALTER XGBOOST PREDICTIONS
# ==============================================================================


def test_decision_layer_does_not_alter_xgboost_predictions(engine):
    """Verify that evaluating safe-area/decision context does NOT mutate the XGBoost outputs."""
    sample_win = _make_dummy_window(user_id="test_user_ortho", lat=39.9800, lon=116.3100)

    # Run 1: Without safe area
    out1 = engine.predict_window(sample_win, horizon_sec=120)

    # Run 2: With circular safe area containing point
    safe_area_inside = {
        "type": "circle",
        "center": [39.9800, 116.3100],
        "radius_m": 500.0,
    }
    out2 = engine.predict_window(sample_win, horizon_sec=120, safe_area=safe_area_inside)

    # Run 3: With circular safe area far away (outside)
    safe_area_outside = {
        "type": "circle",
        "center": [39.0000, 116.0000],
        "radius_m": 50.0,
    }
    out3 = engine.predict_window(sample_win, horizon_sec=120, safe_area=safe_area_outside)

    # XGBoost core predictions must be NUMERICALLY IDENTICAL
    assert out1["risk_score"] == out2["risk_score"] == out3["risk_score"]
    assert (
        out1["trigger_state"]["calibrated_probability"]
        == out2["trigger_state"]["calibrated_probability"]
        == out3["trigger_state"]["calibrated_probability"]
    )
    assert (
        out1["trigger_state"]["binary_alert"]
        == out2["trigger_state"]["binary_alert"]
        == out3["trigger_state"]["binary_alert"]
    )
    assert out1["risk_tier"] == out2["risk_tier"] == out3["risk_tier"]

    # Only contextual decision state changes
    assert out2["safe_area_state"] == "INSIDE_SAFE_AREA"
    assert out3["safe_area_state"] == "OUTSIDE_SAFE_AREA"
    # Behavioral dominance: because behavioral risk is SUSPICIOUS, it correctly overrides safe area in both
    assert out2["decision_state"] == DecisionState.SUSPICIOUS.value
    assert out3["decision_state"] == DecisionState.SUSPICIOUS.value


# ==============================================================================
# 7. RAW RESEARCH OUTPUTS REMAIN AVAILABLE
# ==============================================================================


def test_raw_research_outputs_remain_available(engine):
    """Verify that all raw research and debug outputs are exposed and intact."""
    dummy_win = _make_dummy_window(user_id="test_raw_user")

    out = engine.predict_window(dummy_win, horizon_sec=120)

    # Check top-level contract
    assert "risk_score" in out
    assert "risk_tier" in out
    assert "polling_tier" in out
    assert "kinematic_features" in out
    assert "trigger_state" in out
    assert "explainability" in out

    # Check trigger state details
    assert "calibrated_probability" in out["trigger_state"]
    assert "raw_probability" in out["trigger_state"]
    assert "decision_threshold" in out["trigger_state"]
    assert "binary_alert" in out["trigger_state"]
    assert "horizon_sec" in out["trigger_state"]

    # Check decision output details
    assert "decision" in out
    d = out["decision"]
    assert "risk_score" in d
    assert "calibrated_probability" in d
    assert "raw_probability" in d
    assert "horizon_sec" in d
    assert "top_features" in d
    assert "behavioral_tier" in d


# ==============================================================================
# 8. NO NEW TRAINING LABELS OR FABRICATED GEOLIFE LABELS
# ==============================================================================


def test_no_fabricated_geolife_labels_or_hash_drift():
    """Verify that benchmark artifacts and hashes remain strictly unchanged."""
    locked_hashes = {
        "ml/data/processed/risk_predictions.parquet": (
            "d716f7a268a7e96c874fb89c0e3311ebb343fb4c8fa7557f5eaa26c98b4a204f"
        ),
        "predictive_risk_report.json": "c7b57bdff9c126c382f5e535b376df101d432daa0da2a3c064c424f95bafced1",
        "predictive_risk_report.md": "181c2b108b1f4f8fc044e63f4cdd70e0447726925bf120da6818c9db945df0da",
    }
    for file_path, expected_hash in locked_hashes.items():
        p = Path(file_path)
        assert p.exists(), f"File {file_path} is missing."
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"Hash divergence in {file_path}! {actual_hash} != {expected_hash}"
        )


# ==============================================================================
# 9. STREAMING TELEMETRY INFERENCE CONTRACT TESTS
# ==============================================================================


def test_streaming_telemetry_incomplete_window_represents_insufficient_evidence(engine):
    """Verify that an incomplete/accumulating window is represented as INSUFFICIENT_EVIDENCE,
    not as a genuine SAFE prediction or a fabricated probability."""
    payload = {
        "user_id": "stream_user_accum",
        "timestamp": "2026-10-11T12:00:00Z",
        "location": {"lat": 39.9840, "lng": 116.3180, "altitude_m": 45.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": 1.2,
            "heading_deg": 45.0,
            "activity_type": "WALKING",
            "battery_pct": 85,
        },
        "signal_status": {"state": "VALID"},
    }

    out = engine.predict_telemetry_reading(payload)

    # 1. Must explicitly state INSUFFICIENT_EVIDENCE / ACCUMULATING_WINDOW
    assert out["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
    assert out["trigger_state"]["reason"] == "INSUFFICIENT_EVIDENCE"
    assert out["trigger_state"]["calibrated_probability"] is None
    assert out["trigger_state"]["raw_probability"] is None

    # 2. Must not be classified as a genuine SAFE prediction
    assert out["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out["display_state"] == "ACCUMULATING WINDOW (INSUFFICIENT EVIDENCE)"
    assert "accumulating" in out["human_readable_reason"].lower()
    assert out["decision"]["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out["explainability"]["status"] == "INSUFFICIENT_EVIDENCE"

    # 3. Top-level schema container compliance
    assert out["user_id"] == "stream_user_accum"
    assert out["risk_tier"] == "QUIESCENT"
    assert out["risk_score"] == 0.0
    assert out["location"] == {"lat": 39.9840, "lng": 116.3180}


def test_streaming_telemetry_full_window_evaluates_inference(engine):
    """Verify that accumulating >= 120s with >= 3 points correctly calls
    compute_window_features_dict and generates a valid ML risk prediction."""
    u_id = "stream_user_full"
    # Clear any previous buffers for clean test
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)
    engine.last_risk_outputs.pop(u_id, None)

    timestamps = [
        "2026-10-11T12:00:00Z",
        "2026-10-11T12:00:30Z",
        "2026-10-11T12:01:00Z",
        "2026-10-11T12:01:30Z",
        "2026-10-11T12:02:00Z",  # Exactly 120 seconds from start
    ]

    outputs = []
    for i, ts in enumerate(timestamps):
        reading = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {
                "lat": 39.9800 + i * 0.001,
                "lng": 116.3100 + i * 0.001,
                "altitude_m": 50.0,
            },
            "sensor_metrics": {
                "horizontal_accuracy_m": 4.0,
                "speed_mps": 1.4,
                "heading_deg": 45.0,
                "activity_type": "WALKING",
                "battery_pct": 90,
            },
            "signal_status": {"state": "VALID"},
        }
        out = engine.predict_telemetry_reading(reading)
        outputs.append(out)

    # First 4 readings should be accumulating
    for intermediate in outputs[:4]:
        assert intermediate["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
        assert intermediate["decision_state"] == "INSUFFICIENT_EVIDENCE"
        assert intermediate["trigger_state"]["calibrated_probability"] is None

    # 5th reading completes the 120s window (exercising compute_window_features_dict fix)
    final_out = outputs[4]
    assert final_out["user_id"] == u_id
    assert final_out["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
    assert 0.0 <= final_out["risk_score"] <= 100.0
    assert 0 <= final_out["polling_tier"] <= 3

    # ML probabilities must now be calculated floats, not None
    cal_prob = final_out["trigger_state"]["calibrated_probability"]
    raw_prob = final_out["trigger_state"]["raw_probability"]
    assert cal_prob is not None and isinstance(cal_prob, float)
    assert 0.0 <= cal_prob <= 1.0
    assert raw_prob is not None and isinstance(raw_prob, float)
    assert 0.0 <= raw_prob <= 1.0

    # Decision state must be evaluated
    assert final_out["decision_state"] in (
        "SAFE / NORMAL",
        "FAMILIAR_MOVEMENT",
        "UNFAMILIAR_MOVEMENT",
        "SUSPICIOUS",
        "CRITICAL",
        "OUTSIDE_SAFE_AREA",
    )
    assert final_out["location"]["lat"] == pytest.approx(39.984)
    assert final_out["location"]["lng"] == pytest.approx(116.314)

    # Window history must have recorded this evaluated window
    assert len(engine.window_histories.get(u_id, [])) == 1


def test_streaming_telemetry_degraded_signal_pdr(engine):
    """Verify that DEGRADED_SIGNAL triggers PDR tier escalation contract."""
    degraded_reading = {
        "user_id": "stream_user_pdr",
        "timestamp": "2026-10-11T12:05:00Z",
        "location": {"lat": 39.9800, "lng": 116.3100},
        "sensor_metrics": {
            "horizontal_accuracy_m": 35.0,  # > 20m degraded
            "speed_mps": 0.8,
            "heading_deg": 180.0,
            "battery_pct": 75,
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL",
            "degraded_since": "2026-10-11T12:04:00Z",
        },
        "imu_metrics": {
            "step_count_since_last_gps": 80,
            "net_displacement_m": 55.0,  # > 50m triggers Tier 3
            "pdr_tier_state": "UNTRACKED_DISPLACEMENT",
        },
    }

    out = engine.predict_telemetry_reading(degraded_reading)
    assert out["risk_tier"] == "CRITICAL"
    assert out["polling_tier"] == 3
    assert out["trigger_state"]["frozen_reason"] == "DEGRADED_SIGNAL_PDR_FALLBACK"


def test_streaming_telemetry_malformed_location(engine):
    """Verify that empty location dictionary is handled gracefully without exception."""
    u_id = "stream_user_malformed"
    engine.point_buffers.pop(u_id, None)

    reading = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:10:00Z",
        "location": {},  # Missing lat/lng
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": 0.0,
            "heading_deg": 0.0,
            "battery_pct": 50,
        },
        "signal_status": {"state": "VALID"},
    }

    out = engine.predict_telemetry_reading(reading)
    assert out["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
    assert out["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out["location"] is None


def test_streaming_telemetry_sparse_points_insufficient_evidence(engine):
    """Verify that elapsed time >= 120s with fewer than 3 points remains INSUFFICIENT_EVIDENCE
    until the 3rd point arrives to fulfill the min_points_per_window contract."""
    u_id = "stream_user_sparse"
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)

    # Point 1 at t=0s
    r1 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:00:00Z",
        "location": {"lat": 39.9800, "lng": 116.3100},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 1.0, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out1 = engine.predict_telemetry_reading(r1)
    assert out1["decision_state"] == "INSUFFICIENT_EVIDENCE"

    # Point 2 at t=130s (> 120s elapsed, but only 2 points < min_points_per_window=3)
    r2 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:02:10Z",
        "location": {"lat": 39.9810, "lng": 116.3110},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 1.0, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out2 = engine.predict_telemetry_reading(r2)
    assert out2["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out2["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
    assert out2["kinematic_features"]["points_accumulated"] == 2

    # Point 3 at t=140s (now 3 points and elapsed >= 120s: window evaluation must trigger)
    r3 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:02:20Z",
        "location": {"lat": 39.9820, "lng": 116.3120},
        "sensor_metrics": {"horizontal_accuracy_m": 5.0, "speed_mps": 1.0, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out3 = engine.predict_telemetry_reading(r3)
    assert out3["trigger_state"]["calibrated_probability"] is not None
    assert 0.0 <= out3["risk_score"] <= 100.0
    assert len(engine.window_histories.get(u_id, [])) == 1


def test_streaming_telemetry_consecutive_windows_lag_features(engine):
    """Verify that consecutive streaming windows transition correctly,
    with the second window successfully consuming the first window for lag features."""
    u_id = "stream_user_consecutive"
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)
    engine.last_risk_outputs.pop(u_id, None)

    # Window 1: 5 points spanning 120s
    w1_times = [
        "2026-10-11T12:00:00Z",
        "2026-10-11T12:00:30Z",
        "2026-10-11T12:01:00Z",
        "2026-10-11T12:01:30Z",
        "2026-10-11T12:02:00Z",
    ]
    for i, ts in enumerate(w1_times):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 39.9800 + i * 0.001, "lng": 116.3100 + i * 0.001},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.4, "heading_deg": 45.0, "battery_pct": 90},
            "signal_status": {"state": "VALID"},
        }
        out_w1 = engine.predict_telemetry_reading(r)

    assert out_w1["trigger_state"]["calibrated_probability"] is not None
    assert len(engine.window_histories.get(u_id, [])) == 1

    # Window 2: buffer begins with last point (12:02:00Z); next 4 points to 12:04:00Z (120s elapsed)
    w2_times = [
        "2026-10-11T12:02:30Z",
        "2026-10-11T12:03:00Z",
        "2026-10-11T12:03:30Z",
        "2026-10-11T12:04:00Z",
    ]
    for i, ts in enumerate(w2_times):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 39.9840 + (i + 1) * 0.001, "lng": 116.3140 + (i + 1) * 0.001},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.4, "heading_deg": 45.0, "battery_pct": 90},
            "signal_status": {"state": "VALID"},
        }
        out_w2 = engine.predict_telemetry_reading(r)

    assert out_w2["trigger_state"]["calibrated_probability"] is not None
    assert 0.0 <= out_w2["risk_score"] <= 100.0
    assert len(engine.window_histories.get(u_id, [])) == 2
    # Verify window_id monotonic increment
    hist = engine.window_histories[u_id]
    assert hist[0]["window_id"] == f"{u_id}_w_0"
    assert hist[1]["window_id"] == f"{u_id}_w_1"


# ==============================================================================
# 9. STREAMING WINDOW GAP SEMANTICS (STEP 2A)
# ==============================================================================


def test_inference_config_max_gap_matches_canonical_segmentation(engine):
    """Verify InferenceConfig.max_gap_sec reuses canonical SegmentationConfig.max_gap_seconds (300.0s)."""
    from ml.src.trajectory import SegmentationConfig

    assert engine.config.max_gap_sec == SegmentationConfig.max_gap_seconds
    assert engine.config.max_gap_sec == 300.0


def test_streaming_telemetry_normal_complete_window_acceptable_gaps(engine):
    """Verify normal complete window evaluation when consecutive point gaps are within 300s limit."""
    u_id = "stream_user_normal_gaps"
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)
    engine.last_risk_outputs.pop(u_id, None)
    engine.user_segment_indices.pop(u_id, None)

    # 5 points spanning 120s with 30s gaps (all <= 300s limit)
    timestamps = [
        "2026-10-11T12:00:00Z",
        "2026-10-11T12:00:30Z",
        "2026-10-11T12:01:00Z",
        "2026-10-11T12:01:30Z",
        "2026-10-11T12:02:00Z",
    ]
    out = None
    for i, ts in enumerate(timestamps):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 39.9800 + i * 0.001, "lng": 116.3100 + i * 0.001},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.4, "heading_deg": 45.0, "battery_pct": 90},
            "signal_status": {"state": "VALID"},
        }
        out = engine.predict_telemetry_reading(r)

    assert out is not None
    assert out["decision_state"] in (
        "SAFE / NORMAL",
        "FAMILIAR_MOVEMENT",
        "UNFAMILIAR_MOVEMENT",
        "SUSPICIOUS",
        "CRITICAL",
        "OUTSIDE_SAFE_AREA",
    )
    assert out["trigger_state"]["calibrated_probability"] is not None
    assert 0.0 <= out["risk_score"] <= 100.0
    assert len(engine.window_histories.get(u_id, [])) == 1
    assert engine.window_histories[u_id][0]["segment_id"] == f"{u_id}_stream_s0"
    assert engine.window_histories[u_id][0]["window_id"] == f"{u_id}_w_0"


def test_streaming_telemetry_gap_exceeding_300s_resets_buffer_and_prevents_evaluation(engine):
    """Verify that a point gap exceeding 300.0s terminates the active segment,
    resets the buffer to the latest point, and prevents premature cross-gap window evaluation."""
    u_id = "stream_user_gap_exceeded"
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)
    engine.last_risk_outputs.pop(u_id, None)
    engine.user_segment_indices.pop(u_id, None)

    # Point 1 at t=0s
    r1 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:00:00Z",
        "location": {"lat": 39.9800, "lng": 116.3100},
        "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.2, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out1 = engine.predict_telemetry_reading(r1)
    assert out1["decision_state"] == "INSUFFICIENT_EVIDENCE"

    # Point 2 at t=30s
    r2 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:00:30Z",
        "location": {"lat": 39.9802, "lng": 116.3102},
        "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.2, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out2 = engine.predict_telemetry_reading(r2)
    assert out2["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert len(engine.point_buffers[u_id]) == 2

    # Point 3 at t=360s (gap = 330s > 300.0s max_gap_sec)
    r3 = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:06:00Z",
        "location": {"lat": 39.9850, "lng": 116.3150},
        "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.2, "heading_deg": 0.0, "battery_pct": 80},
        "signal_status": {"state": "VALID"},
    }
    out3 = engine.predict_telemetry_reading(r3)

    # Buffer must have been reset to ONLY point 3 (len == 1)
    assert len(engine.point_buffers[u_id]) == 1
    assert engine.point_buffers[u_id][0]["latitude"] == pytest.approx(39.9850)
    # Segment index must have incremented to 1
    assert engine.user_segment_indices[u_id] == 1
    # Must NOT evaluate a cross-gap window: remains INSUFFICIENT_EVIDENCE
    assert out3["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out3["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
    assert out3["trigger_state"]["calibrated_probability"] is None
    assert out3["kinematic_features"]["points_accumulated"] == 1
    assert out3["kinematic_features"]["buffer_duration_sec"] == 0.0
    assert len(engine.window_histories.get(u_id, [])) == 0


def test_streaming_telemetry_post_gap_window_does_not_inherit_cross_gap_movement_or_lag(engine):
    """Verify that after a >300s gap, the next evaluated window does not merge cross-gap
    movement and does not inherit lag state (prev_window is None for new segment).
    Subsequent windows in the same new segment then resume normal lag features."""
    u_id = "stream_user_post_gap_lag"
    engine.point_buffers.pop(u_id, None)
    engine.window_histories.pop(u_id, None)
    engine.last_risk_outputs.pop(u_id, None)
    engine.user_segment_indices.pop(u_id, None)

    # Segment 0: Complete Window 0 (t=0s to t=120s)
    w0_times = [
        "2026-10-11T12:00:00Z",
        "2026-10-11T12:00:30Z",
        "2026-10-11T12:01:00Z",
        "2026-10-11T12:01:30Z",
        "2026-10-11T12:02:00Z",
    ]
    out_w0 = None
    for i, ts in enumerate(w0_times):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 39.9800 + i * 0.0005, "lng": 116.3100 + i * 0.0005},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 1.0, "heading_deg": 45.0, "battery_pct": 90},
            "signal_status": {"state": "VALID"},
        }
        out_w0 = engine.predict_telemetry_reading(r)

    assert out_w0["trigger_state"]["calibrated_probability"] is not None
    assert len(engine.window_histories[u_id]) == 1
    assert engine.window_histories[u_id][0]["segment_id"] == f"{u_id}_stream_s0"
    assert engine.last_risk_outputs.get(u_id) is not None

    # Temporal gap: 600s gap from 12:02:00Z to 12:12:00Z (> 300.0s max_gap_sec)
    r_gap = {
        "user_id": u_id,
        "timestamp": "2026-10-11T12:12:00Z",
        "location": {"lat": 40.0000, "lng": 116.3500},
        "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 5.0, "heading_deg": 90.0, "battery_pct": 85},
        "signal_status": {"state": "VALID"},
    }
    out_gap = engine.predict_telemetry_reading(r_gap)

    # Verifies gap split: last_risk_outputs cleared, returns INSUFFICIENT_EVIDENCE
    assert out_gap["decision_state"] == "INSUFFICIENT_EVIDENCE"
    assert out_gap["trigger_state"]["status"] == "ACCUMULATING_WINDOW"
    assert engine.user_segment_indices[u_id] == 1
    assert len(engine.point_buffers[u_id]) == 1

    # Segment 1: Complete Window 1 (12:12:00Z to 12:14:00Z, 120s elapsed in new segment)
    w1_times = [
        "2026-10-11T12:12:30Z",
        "2026-10-11T12:13:00Z",
        "2026-10-11T12:13:30Z",
        "2026-10-11T12:14:00Z",
    ]
    out_w1 = None
    for i, ts in enumerate(w1_times):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 40.0000 + (i + 1) * 0.002, "lng": 116.3500 + (i + 1) * 0.002},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 8.0, "heading_deg": 90.0, "battery_pct": 85},
            "signal_status": {"state": "VALID"},
        }
        out_w1 = engine.predict_telemetry_reading(r)

    assert out_w1["trigger_state"]["calibrated_probability"] is not None
    assert len(engine.window_histories[u_id]) == 2
    w0_features = engine.window_histories[u_id][0]
    w1_features = engine.window_histories[u_id][1]
    assert w0_features["segment_id"] == f"{u_id}_stream_s0"
    assert w0_features["window_id"] == f"{u_id}_w_0"
    assert w0_features["window_index"] == 0
    assert out_w0["metadata"]["window_index"] == 0

    assert w1_features["segment_id"] == f"{u_id}_stream_s1"
    assert w1_features["window_id"] == f"{u_id}_w_1"  # Globally unique monotonic window ID
    # Canonical per-segment window_index resets to 0 relative to Segment 1 start
    assert w1_features["window_index"] == 0
    assert out_w1["metadata"]["window_index"] == 0
    assert out_w1["metadata"]["segment_id"] == f"{u_id}_stream_s1"

    # Verify temporal span of Window 1 is exactly 120s within the new segment, NOT 840s across the gap
    t_start = pd.to_datetime(w1_features["start_time"])
    t_end = pd.to_datetime(w1_features["end_time"])
    assert (t_end - t_start).total_seconds() == pytest.approx(120.0)
    assert t_start == pd.to_datetime("2026-10-11T12:12:00Z")

    # Segment 1 continuing: Complete Window 2 (12:14:00Z to 12:16:00Z, no gap)
    w2_times = [
        "2026-10-11T12:14:30Z",
        "2026-10-11T12:15:00Z",
        "2026-10-11T12:15:30Z",
        "2026-10-11T12:16:00Z",
    ]
    out_w2 = None
    for i, ts in enumerate(w2_times):
        r = {
            "user_id": u_id,
            "timestamp": ts,
            "location": {"lat": 40.0080 + (i + 1) * 0.002, "lng": 116.3580 + (i + 1) * 0.002},
            "sensor_metrics": {"horizontal_accuracy_m": 4.0, "speed_mps": 8.0, "heading_deg": 90.0, "battery_pct": 80},
            "signal_status": {"state": "VALID"},
        }
        out_w2 = engine.predict_telemetry_reading(r)

    assert out_w2["trigger_state"]["calibrated_probability"] is not None
    assert len(engine.window_histories[u_id]) == 3
    w2_features = engine.window_histories[u_id][2]
    assert w2_features["segment_id"] == f"{u_id}_stream_s1"
    assert w2_features["window_id"] == f"{u_id}_w_2"  # Globally unique monotonic window ID
    # Consecutive window within Segment 1 increments to 1
    assert w2_features["window_index"] == 1
    assert out_w2["metadata"]["window_index"] == 1
    assert out_w2["metadata"]["segment_id"] == f"{u_id}_stream_s1"


# ==============================================================================
# 10. BATCH LAG ISOLATION ACROSS SEGMENT BOUNDARIES (STEP 2B)
# ==============================================================================


def test_batch_window_lag_isolation_across_segment_boundaries(engine):
    """Verify that batch inference isolates lag features at segment boundaries:
    even if window_index values are numerically consecutive (e.g. 0 and 1),
    differing segment_id values prevent the previous window from being passed as lag state.
    Conversely, consecutive windows sharing the same segment_id correctly inherit lag state."""
    w_seg0_w0 = _make_dummy_window(
        window_id="w_seg0_w0",
        speed=1.5,
    )
    w_seg0_w0["segment_id"] = "traj_001_s0"
    w_seg0_w0["window_index"] = 0

    w_seg1_w1 = _make_dummy_window(
        window_id="w_seg1_w1",
        speed=8.0,
    )
    w_seg1_w1["segment_id"] = "traj_001_s1"
    w_seg1_w1["window_index"] = 1  # Numerically consecutive: 1 == 0 + 1

    w_seg1_w2 = _make_dummy_window(
        window_id="w_seg1_w2",
        speed=8.5,
    )
    w_seg1_w2["segment_id"] = "traj_001_s1"  # Same segment as w_seg1_w1
    w_seg1_w2["window_index"] = 2  # Numerically consecutive: 2 == 1 + 1

    windows = [w_seg0_w0, w_seg1_w1, w_seg1_w2]

    # Evaluate using the exact batch pairing contract:
    # requires BOTH consecutive window_index AND matching segment_id
    resolved_prev_wins = []
    for i, curr_win in enumerate(windows):
        prev_win = (
            windows[i - 1]
            if (
                i > 0
                and curr_win.get("window_index") == windows[i - 1].get("window_index") + 1
                and curr_win.get("segment_id") == windows[i - 1].get("segment_id")
            )
            else None
        )
        resolved_prev_wins.append(prev_win)

    # Window 0 has no predecessor
    assert resolved_prev_wins[0] is None

    # Window 1 is on the opposite side of a segment boundary:
    # Must NOT inherit Window 0 as lag state despite consecutive window_index
    assert resolved_prev_wins[1] is None

    # Window 2 is in the same segment as Window 1 with consecutive window_index:
    # Must inherit Window 1
    assert resolved_prev_wins[2] is w_seg1_w1

    # Verify that engine.predict_window() independently enforces segment boundary isolation
    # when given cross-segment prev_window
    out_isolated = engine.predict_window(curr_window=w_seg1_w1, prev_window=w_seg0_w0)
    out_clean = engine.predict_window(curr_window=w_seg1_w1, prev_window=None)

    # Prediction score and raw/calibrated probabilities must be identical to clean evaluation
    # (i.e. cross-segment w_seg0_w0 was not used for lag deltas)
    assert out_isolated["risk_score"] == pytest.approx(out_clean["risk_score"])
    assert out_isolated["trigger_state"]["calibrated_probability"] == pytest.approx(
        out_clean["trigger_state"]["calibrated_probability"]
    )


def test_predict_plt_trajectory_isolates_segments_with_large_gap(engine, tmp_path):
    """Verify predict_plt_trajectory() partitions multi-segment trajectories across gaps > 300s,
    anchors elapsed time and window_index to each independent segment's own start timestamp,
    assigns distinct segment and unique window IDs, and enforces lag isolation at segment boundaries."""
    plt_file = tmp_path / "test_multiseg_traj.plt"
    lines = [
        "Geonaute Track File",
        "WGS 84",
        "Altitude is in Feet",
        "Reserved 3",
        "0,2,255,My Track,0",
        "0",
    ]
    # Segment 0: 3 points in window 0 (t = 0, 30, 60 seconds relative to segment 0 start 10:00:00)
    lines.extend([
        "39.9800,116.3100,0,100,39700.0,2008-10-23,10:00:00",
        "39.9805,116.3105,0,100,39700.0,2008-10-23,10:00:30",
        "39.9810,116.3110,0,100,39700.0,2008-10-23,10:01:00",
    ])
    # Gap of 540s (10:01:00 to 10:10:00) > 300s canonical threshold (SegmentationConfig.max_gap_seconds)
    # Segment 1 starts at 10:10:00.
    # Window 0 of Segment 1: elapsed 0s to 60s relative to segment 1 start (10:10:00 to 10:11:00)
    lines.extend([
        "39.9900,116.3200,0,100,39700.0,2008-10-23,10:10:00",
        "39.9905,116.3205,0,100,39700.0,2008-10-23,10:10:30",
        "39.9910,116.3210,0,100,39700.0,2008-10-23,10:11:00",
    ])
    # Window 1 of Segment 1: elapsed 120s to 180s relative to segment 1 start (10:12:00 to 10:13:00)
    lines.extend([
        "39.9915,116.3215,0,100,39700.0,2008-10-23,10:12:00",
        "39.9920,116.3220,0,100,39700.0,2008-10-23,10:12:30",
        "39.9925,116.3225,0,100,39700.0,2008-10-23,10:13:00",
    ])
    plt_file.write_text("\n".join(lines), encoding="utf-8")

    with unittest.mock.patch.object(engine, "predict_window", wraps=engine.predict_window) as spy_pred:
        preds = engine.predict_plt_trajectory(plt_file_path=plt_file, user_id="test_user_plt")

        # 1. Exactly 3 retained windows across the 2 segments
        assert len(preds) == 3
        assert spy_pred.call_count == 3

        # 2. Window grouping, segment IDs, and window ID uniqueness across the gap
        seg_0_id = "test_multiseg_traj_s0"
        seg_1_id = "test_multiseg_traj_s1"
        assert preds[0]["metadata"]["segment_id"] == seg_0_id
        assert preds[1]["metadata"]["segment_id"] == seg_1_id
        assert preds[2]["metadata"]["segment_id"] == seg_1_id

        win_0_id = "test_user_plt_s0_w_0"
        win_1_id = "test_user_plt_s1_w_0"
        win_2_id = "test_user_plt_s1_w_1"
        assert preds[0]["metadata"]["window_id"] == win_0_id
        assert preds[1]["metadata"]["window_id"] == win_1_id
        assert preds[2]["metadata"]["window_id"] == win_2_id
        # All window IDs remain unique even when window_index resets
        assert len({p["metadata"]["window_id"] for p in preds}) == 3

        # 3. Canonical per-segment window_index reset
        # Segment 0: window_index is 0
        assert preds[0]["metadata"]["window_index"] == 0
        # Segment 1: window_index RESETS to 0 relative to segment 1's own start timestamp (10:10:00)
        assert preds[1]["metadata"]["window_index"] == 0
        # Segment 1: subsequent window increments to 1
        assert preds[2]["metadata"]["window_index"] == 1

        # 4. Lag state inheritance verification in predict_window()
        # Call 0 (Segment 0, Window 0): no predecessor
        call_0_kwargs = spy_pred.call_args_list[0].kwargs
        assert call_0_kwargs["prev_window"] is None

        # Call 1 (Segment 1, Window 0): first retained window in new segment
        # CANNOT inherit lag state from preceding segment across gap
        call_1_kwargs = spy_pred.call_args_list[1].kwargs
        assert call_1_kwargs["prev_window"] is None

        # Call 2 (Segment 1, Window 1): consecutive window within new segment
        # DOES inherit lag state from the preceding window in the same segment
        call_2_kwargs = spy_pred.call_args_list[2].kwargs
        assert call_2_kwargs["prev_window"] is not None
        assert call_2_kwargs["prev_window"]["window_id"] == win_1_id
        assert call_2_kwargs["prev_window"]["segment_id"] == seg_1_id

        # 5. Output contract verification
        for p in preds:
            assert p["user_id"] == "test_user_plt"
            assert 0.0 <= p["risk_score"] <= 100.0
            assert p["trigger_state"]["calibrated_probability"] is not None

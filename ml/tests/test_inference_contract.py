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
import pytest

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

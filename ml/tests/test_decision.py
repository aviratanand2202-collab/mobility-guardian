"""Focused Unit Tests for Deterministic Decision Output Layer (Chunk 3).

Verifies all combinations and requirements:
1. SAFE / NORMAL:
   - Inside safe area with normal kinematics -> SAFE / NORMAL
   - Safe area unavailable with normal routine -> SAFE / NORMAL
2. OUTSIDE_SAFE_AREA:
   - Outside safe area alone with normal kinematics -> OUTSIDE_SAFE_AREA (never SUSPICIOUS/CRITICAL)
3. FAMILIAR_MOVEMENT:
   - Moving near familiar anchor when safe area unavailable -> FAMILIAR_MOVEMENT
4. UNFAMILIAR_MOVEMENT:
   - Moving in unfamiliar territory with normal kinematics -> UNFAMILIAR_MOVEMENT (never dangerous)
5. SUSPICIOUS:
   - Elevated behavioral risk -> SUSPICIOUS (regardless of location)
6. CRITICAL:
   - Critical behavioral excursion -> CRITICAL (location does NOT override severe behavioral anomaly)
7. EXPOSED FIELDS & METHODOLOGY:
   - Minimum required fields are present
   - Non-clinical disclaimers are attached to every decision
"""

from typing import Any, Dict

import pytest

from ml.src.decision import (
    DecisionOutput,
    DecisionState,
    interpret_decision,
)
from ml.src.inference import RiskInferenceEngine


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def base_risk_dict() -> Dict[str, Any]:
    """Base dictionary mimicking output of predict_window."""
    return {
        "user_id": "test_user_001",
        "timestamp": "2026-09-17T08:30:00Z",
        "risk_tier": "NORMAL_TRANSIT",
        "risk_score": 18.5,
        "polling_tier": 1,
        "predicted_lead_time_sec": None,
        "battery_override_active": False,
        "kinematic_features": {
            "mean_speed_mps": 1.2,
            "tortuosity_index": 1.1,
            "behavioral_indicator": "NORMAL",
            "behavioral_confidence": 0.95,
        },
        "trigger_state": {
            "window_seconds": 120,
            "horizon_sec": 120,
            "decision_threshold": 0.45,
            "calibrated_probability": 0.185,
            "raw_probability": 0.15,
            "binary_alert": False,
        },
        "explainability": {
            "top_features": [{"feature": "mean_speed_mps", "value": 1.2}],
            "behavior_classification": "NORMAL",
            "disclaimer": "Associational kinematic excursion prediction.",
        },
        "safe_area_state": "INSIDE_SAFE_AREA",
        "familiarity_state": "FAMILIAR_LOCATION",
        "geospatial_context": {
            "safe_area_state": "INSIDE_SAFE_AREA",
            "is_safe_area_available": True,
            "distance_to_safe_boundary_m": -150.0,
            "matched_safe_area_name": "Home Zone",
            "familiarity_state": "FAMILIAR_LOCATION",
            "nearest_anchor_id": "u001_anchor_0",
            "distance_to_nearest_anchor_m": 25.0,
            "is_caregiver_confirmed": False,
        },
        "metadata": {
            "profile_mode": "PERSONALIZED",
            "cold_start_status": "ML_DRIVEN",
            "trip_count": 15,
            "baseline_type": "PERSONALIZED",
        },
    }


# ==============================================================================
# 1. MAJOR DECISION STATE COMBINATION TESTS
# ==============================================================================


def test_decision_safe_normal_inside_safe_area(base_risk_dict):
    """Verify that INSIDE safe area + normal behavior yields SAFE / NORMAL."""
    base_risk_dict["risk_tier"] = "NORMAL_TRANSIT"
    base_risk_dict["safe_area_state"] = "INSIDE_SAFE_AREA"

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.SAFE_NORMAL.value
    assert out.severity_level == "NORMAL"
    assert "Routine Movement" in out.headline or "Safe Area" in out.headline
    assert "safe area" in out.reason.lower()


def test_decision_outside_safe_area_alone_not_suspicious_or_critical(base_risk_dict):
    """Verify that OUTSIDE safe area alone does NOT become SUSPICIOUS or CRITICAL."""
    base_risk_dict["risk_tier"] = "NORMAL_TRANSIT"
    base_risk_dict["safe_area_state"] = "OUTSIDE_SAFE_AREA"
    base_risk_dict["geospatial_context"]["distance_to_safe_boundary_m"] = 85.0
    base_risk_dict["geospatial_context"]["matched_safe_area_name"] = "Home Zone"

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.OUTSIDE_SAFE_AREA.value
    # CRITICAL INVARIANT: must NOT be promoted to SUSPICIOUS or CRITICAL
    assert out.decision_state != DecisionState.SUSPICIOUS.value
    assert out.decision_state != DecisionState.CRITICAL.value
    assert out.severity_level == "ADVISORY"
    assert "outside configured safe area" in out.reason.lower()
    assert "normal" in out.reason.lower() or "routine" in out.reason.lower()


def test_decision_familiar_movement_when_safe_area_unavailable(base_risk_dict):
    """Verify FAMILIAR_MOVEMENT state when safe area is unavailable but location is near an anchor."""
    base_risk_dict["risk_tier"] = "NORMAL_TRANSIT"
    base_risk_dict["safe_area_state"] = "SAFE_AREA_UNAVAILABLE"
    base_risk_dict["familiarity_state"] = "FAMILIAR_LOCATION"

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.FAMILIAR_MOVEMENT.value
    assert out.severity_level == "NORMAL"
    assert "Familiar Anchor" in out.headline or "Routine Movement" in out.headline
    assert "u001_anchor_0" in out.reason


def test_decision_unfamiliar_movement_alone_does_not_imply_danger(base_risk_dict):
    """Verify UNFAMILIAR_MOVEMENT state when location is unfamiliar but kinematics are normal."""
    base_risk_dict["risk_tier"] = "NORMAL_TRANSIT"
    base_risk_dict["safe_area_state"] = "SAFE_AREA_UNAVAILABLE"
    base_risk_dict["familiarity_state"] = "UNFAMILIAR_LOCATION"
    base_risk_dict["geospatial_context"]["distance_to_nearest_anchor_m"] = 2500.0

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.UNFAMILIAR_MOVEMENT.value
    # CRITICAL INVARIANT: unfamiliar location alone does NOT imply danger
    assert out.severity_level == "ADVISORY"
    assert out.decision_state != DecisionState.SUSPICIOUS.value
    assert out.decision_state != DecisionState.CRITICAL.value
    assert "does not imply danger" in out.reason.lower()
    assert "normal" in out.reason.lower()


def test_decision_suspicious_behavioral_risk_dominance(base_risk_dict):
    """Verify that SUSPICIOUS behavioral risk tier dominates regardless of safe area or familiarity."""
    # Even if inside safe area and at familiar anchor, suspicious kinematics trigger SUSPICIOUS
    base_risk_dict["risk_tier"] = "SUSPICIOUS"
    base_risk_dict["risk_score"] = 55.0
    base_risk_dict["safe_area_state"] = "INSIDE_SAFE_AREA"
    base_risk_dict["familiarity_state"] = "FAMILIAR_LOCATION"

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.SUSPICIOUS.value
    assert out.severity_level == "WARNING"
    assert "Elevated" in out.headline or "Suspicious" in out.headline


def test_decision_critical_behavioral_risk_dominance(base_risk_dict):
    """Verify that CRITICAL behavioral risk tier dominates even within safe area and familiar anchor."""
    # Familiar location alone must NOT imply safety if behavioral risk is high
    base_risk_dict["risk_tier"] = "CRITICAL"
    base_risk_dict["risk_score"] = 82.0
    base_risk_dict["safe_area_state"] = "INSIDE_SAFE_AREA"
    base_risk_dict["familiarity_state"] = "FAMILIAR_LOCATION"

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.CRITICAL.value
    assert out.severity_level == "ALERT"
    # Reason explicitly highlights anomaly despite being inside boundary
    assert "despite" in out.reason.lower() or "safe area" in out.reason.lower()


def test_decision_cold_start_new_user_safe_normal(base_risk_dict):
    """Verify SAFE / NORMAL state for new cold-start user with no history and normal kinematics."""
    base_risk_dict["risk_tier"] = "QUIESCENT"
    base_risk_dict["safe_area_state"] = "SAFE_AREA_UNAVAILABLE"
    base_risk_dict["familiarity_state"] = "NO_HISTORY"
    base_risk_dict["metadata"]["profile_mode"] = "COLD_START"
    base_risk_dict["metadata"]["trip_count"] = 0

    out: DecisionOutput = interpret_decision(base_risk_dict)

    assert out.decision_state == DecisionState.SAFE_NORMAL.value
    assert out.profile_mode == "COLD_START"
    assert out.trip_count == 0


# ==============================================================================
# 2. REQUIRED EXPOSED FIELDS & NON-CLINICAL NOTICE TESTS
# ==============================================================================


def test_decision_output_contains_all_required_fields(base_risk_dict):
    """Verify that DecisionOutput exposes all required fields."""
    out: DecisionOutput = interpret_decision(base_risk_dict)
    d = out.to_dict()

    required_keys = [
        "decision_state",
        "display_state",
        "headline",
        "reason",
        "severity_level",
        "profile_mode",
        "safe_area_state",
        "familiarity_state",
        "behavioral_tier",
        "risk_score",
        "calibrated_probability",
        "raw_probability",
        "horizon_sec",
        "binary_alert",
        "top_features",
        "distance_to_safe_boundary_m",
        "safe_area_name",
        "nearest_anchor_id",
        "distance_to_nearest_anchor_m",
        "is_caregiver_confirmed_anchor",
        "trip_count",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in d, f"Decision output dictionary missing required field: {key}"


def test_decision_disclaimer_contains_no_clinical_claims(base_risk_dict):
    """Verify that disclaimers explicitly disclaim clinical, dementia, and wandering diagnoses."""
    out: DecisionOutput = interpret_decision(base_risk_dict)
    disclaimer = out.disclaimer

    assert "NOT constitute clinical diagnosis" in disclaimer
    assert "dementia" in disclaimer
    assert "wandering" in disclaimer


# ==============================================================================
# 3. END-TO-END INFERENCE ENGINE INTEGRATION TESTS
# ==============================================================================


def test_inference_engine_predict_window_emits_decision():
    """Verify that RiskInferenceEngine.predict_window attaches valid decision output."""
    engine = RiskInferenceEngine()

    dummy_win = {
        "window_id": "w_test_dec",
        "user_id": "u_test_dec",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 1.5,
        "speed_std_dev": 0.3,
        "path_distance_m": 180.0,
        "straight_line_displacement_m": 160.0,
        "tortuosity_index": 1.12,
        "entropy_directional": 1.0,
        "turn_frequency": 0.02,
        "pacing_tendency": 0.0,
        "loop_metric": 0.01,
        "latitude": 39.9840,
        "longitude": 116.3180,
    }

    safe_area = {
        "type": "circle",
        "name": "Caregiver Safe Zone",
        "center": [39.9840, 116.3180],
        "radius_m": 300.0,
    }

    pred = engine.predict_window(dummy_win, horizon_sec=120, safe_area=safe_area)

    assert "decision_state" in pred
    assert "display_state" in pred
    assert "human_readable_reason" in pred
    assert "decision" in pred
    assert pred["decision_state"] in [s.value for s in DecisionState]
    assert pred["decision"]["is_caregiver_confirmed_anchor"] is False

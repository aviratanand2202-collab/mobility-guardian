"""Focused Test Suite for Geospatial Context & Familiarity Layer (Chunk 2).

Verifies:
1. SAFE-AREA STATE:
   - INSIDE_SAFE_AREA
   - OUTSIDE_SAFE_AREA
   - SAFE_AREA_UNAVAILABLE
   - Multiple safe zones & polygon safe areas
2. FAMILIARITY STATE:
   - FAMILIAR_LOCATION
   - UNFAMILIAR_LOCATION
   - INSUFFICIENT_EVIDENCE
   - NO_HISTORY
   - Algorithmic anchor review flag (not caregiver confirmed)
3. ORTHOGONALITY & SCIENTIFIC INTEGRITY:
   - Decoupled from XGBoost: identical risk scores and probabilities regardless of safe area
   - Outside safe area does NOT inherently trigger high risk or alert
   - Unfamiliar location does NOT inherently trigger high risk or alert
   - Centralized disclaimers & notes are attached
"""

from typing import Any, Dict

import pytest

from ml.src.geofence import (
    FamiliarityState,
    GeospatialConfig,
    SafeAreaState,
    evaluate_geospatial_context,
    evaluate_location_familiarity,
)
from ml.src.inference import RiskInferenceEngine


# ==============================================================================
# FIXTURES & HELPERS
# ==============================================================================


@pytest.fixture
def sample_anchors() -> list[Dict[str, Any]]:
    """Sample DBSCAN algorithmic anchor clusters for testing."""
    return [
        {
            "anchor_id": "u001_anchor_0",
            "name": "Home Anchor",
            "center_latitude": 39.9840,
            "center_longitude": 116.3180,
            "radius_m": 100.0,
            "observation_count": 12,
            "status": "CONFIRMED",  # Algorithmic DBSCAN confirmation, NOT caregiver
        },
        {
            "anchor_id": "u001_anchor_1",
            "name": "Activity Center",
            "center_latitude": 40.0000,
            "center_longitude": 116.3300,
            "radius_m": 80.0,
            "observation_count": 6,
            "status": "PENDING_CAREGIVER_REVIEW",
        },
    ]


@pytest.fixture
def circular_safe_area() -> Dict[str, Any]:
    """Sample circular safe area boundary."""
    return {
        "type": "circle",
        "name": "Home Primary Safe Zone",
        "center": [39.9840, 116.3180],
        "radius_m": 250.0,
    }


@pytest.fixture
def polygon_safe_area() -> Dict[str, Any]:
    """Sample polygonal safe area boundary around a neighborhood."""
    return {
        "type": "polygon",
        "name": "Neighborhood Geofence",
        "vertices": [
            [39.9800, 116.3100],
            [39.9900, 116.3100],
            [39.9900, 116.3300],
            [39.9800, 116.3300],
        ],
    }


# ==============================================================================
# 1. SAFE-AREA STATE TESTS
# ==============================================================================


def test_safe_area_unavailable_when_none_supplied():
    """Verify SAFE_AREA_UNAVAILABLE state when no safe area is configured."""
    res = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        safe_area=None,
        user_profile=None,
    )
    assert res.safe_area_state == SafeAreaState.SAFE_AREA_UNAVAILABLE
    assert not res.is_safe_area_available
    assert res.distance_to_safe_boundary_m is None
    assert res.matched_safe_area_name is None


def test_inside_safe_area_circular(circular_safe_area):
    """Verify INSIDE_SAFE_AREA state when point is within circular radius."""
    # Center of safe zone: distance is 0, margin is -250m (deep inside)
    res = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        safe_area=circular_safe_area,
    )
    assert res.safe_area_state == SafeAreaState.INSIDE_SAFE_AREA
    assert res.is_safe_area_available
    assert res.distance_to_safe_boundary_m is not None
    assert res.distance_to_safe_boundary_m < 0.0  # Inside has negative margin
    assert res.matched_safe_area_name == "Home Primary Safe Zone"


def test_outside_safe_area_circular(circular_safe_area):
    """Verify OUTSIDE_SAFE_AREA state when point is beyond circular radius."""
    # Point ~2.5km north
    res = evaluate_geospatial_context(
        latitude=40.0100,
        longitude=116.3180,
        safe_area=circular_safe_area,
    )
    assert res.safe_area_state == SafeAreaState.OUTSIDE_SAFE_AREA
    assert res.is_safe_area_available
    assert res.distance_to_safe_boundary_m is not None
    assert res.distance_to_safe_boundary_m > 0.0  # Outside has positive distance
    assert res.matched_safe_area_name == "Home Primary Safe Zone"


def test_polygon_safe_area_inside_and_outside(polygon_safe_area):
    """Verify INSIDE_SAFE_AREA and OUTSIDE_SAFE_AREA for polygonal boundaries."""
    # Point inside polygon [39.980 - 39.990, 116.310 - 116.330]
    res_inside = evaluate_geospatial_context(
        latitude=39.9850,
        longitude=116.3200,
        safe_area=polygon_safe_area,
    )
    assert res_inside.safe_area_state == SafeAreaState.INSIDE_SAFE_AREA
    assert res_inside.matched_safe_area_name == "Neighborhood Geofence"

    # Point outside polygon
    res_outside = evaluate_geospatial_context(
        latitude=39.9700,
        longitude=116.3200,
        safe_area=polygon_safe_area,
    )
    assert res_outside.safe_area_state == SafeAreaState.OUTSIDE_SAFE_AREA
    assert res_outside.distance_to_safe_boundary_m > 0.0


def test_multiple_safe_zones_union(circular_safe_area, polygon_safe_area):
    """Verify that multiple safe areas evaluate as INSIDE if inside ANY zone."""
    multi_boundary = [circular_safe_area, polygon_safe_area]

    # Point inside polygon but outside circle
    res = evaluate_geospatial_context(
        latitude=39.9880,
        longitude=116.3280,
        safe_area=multi_boundary,
    )
    assert res.safe_area_state == SafeAreaState.INSIDE_SAFE_AREA
    assert res.matched_safe_area_name == "Neighborhood Geofence"


def test_safe_area_with_nan_or_invalid_coordinates(circular_safe_area):
    """Verify safe degradation when coordinates are NaN."""
    res = evaluate_geospatial_context(
        latitude=float("nan"),
        longitude=116.3180,
        safe_area=circular_safe_area,
    )
    assert res.safe_area_state == SafeAreaState.SAFE_AREA_UNAVAILABLE
    assert res.familiarity_state == FamiliarityState.INSUFFICIENT_EVIDENCE


# ==============================================================================
# 2. FAMILIARITY STATE TESTS
# ==============================================================================


def test_familiarity_no_history_for_new_user():
    """Verify NO_HISTORY state for a cold-start user without historical profile."""
    # Case A: user_profile is None
    res_none = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        user_profile=None,
    )
    assert res_none.familiarity_state == FamiliarityState.NO_HISTORY

    # Case B: user_profile has trip_count = 0
    res_zero = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        user_profile={"trip_count": 0, "anchor_clusters": []},
    )
    assert res_zero.familiarity_state == FamiliarityState.NO_HISTORY


def test_familiarity_insufficient_evidence_when_low_trip_count(sample_anchors):
    """Verify INSUFFICIENT_EVIDENCE state when trips < 7 (below stabilization threshold)."""
    profile_low_trips = {
        "trip_count": 4,  # Below minimum 7 trips
        "anchor_clusters": sample_anchors,
    }
    state, anchor_id, dist_m, status, confirmed, trips, notes = evaluate_location_familiarity(
        lat=39.9840,
        lon=116.3180,
        user_profile=profile_low_trips,
    )
    assert state == FamiliarityState.INSUFFICIENT_EVIDENCE
    assert trips == 4
    assert any("below minimum threshold" in n for n in notes)


def test_familiarity_insufficient_evidence_when_zero_anchors():
    """Verify INSUFFICIENT_EVIDENCE state when trips >= 7 but no anchors detected."""
    profile_no_anchors = {
        "trip_count": 10,
        "anchor_clusters": [],  # No clusters formed
    }
    state, anchor_id, dist_m, status, confirmed, trips, notes = evaluate_location_familiarity(
        lat=39.9840,
        lon=116.3180,
        user_profile=profile_no_anchors,
    )
    assert state == FamiliarityState.INSUFFICIENT_EVIDENCE
    assert trips == 10
    assert any("no spatial anchor clusters" in n for n in notes)


def test_familiarity_familiar_location_within_anchor_catchment(sample_anchors):
    """Verify FAMILIAR_LOCATION when within learned anchor catchment."""
    profile = {
        "trip_count": 15,
        "anchor_clusters": sample_anchors,
    }
    # Coordinate exactly at Home Anchor (radius 100m)
    state, anchor_id, dist_m, status, confirmed, trips, notes = evaluate_location_familiarity(
        lat=39.9840,
        lon=116.3180,
        user_profile=profile,
    )
    assert state == FamiliarityState.FAMILIAR_LOCATION
    assert anchor_id == "u001_anchor_0"
    assert dist_m < 10.0
    assert status == "CONFIRMED"
    # Algorithmic confirmation is NOT caregiver confirmed!
    assert not confirmed


def test_familiarity_unfamiliar_location_outside_anchors(sample_anchors):
    """Verify UNFAMILIAR_LOCATION when coordinate is beyond all known anchors."""
    profile = {
        "trip_count": 15,
        "anchor_clusters": sample_anchors,
    }
    # Coordinate far away from both anchors (~10km)
    state, anchor_id, dist_m, status, confirmed, trips, notes = evaluate_location_familiarity(
        lat=39.9000,
        lon=116.3180,
        user_profile=profile,
    )
    assert state == FamiliarityState.UNFAMILIAR_LOCATION
    assert dist_m > 5000.0  # kilometers away
    assert any("outside all 2 known anchor clusters" in n for n in notes)


def test_algorithmic_anchors_not_caregiver_confirmed(sample_anchors):
    """Verify that DBSCAN algorithmic clusters are explicitly flagged as NOT caregiver-confirmed."""
    res = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        user_profile={"trip_count": 10, "anchor_clusters": sample_anchors},
    )
    assert res.familiarity_state == FamiliarityState.FAMILIAR_LOCATION
    assert not res.is_caregiver_confirmed
    assert "Algorithmic anchors are NOT caregiver-confirmed" in res.disclaimers["familiarity"]


# ==============================================================================
# 3. ORTHOGONALITY & SCIENTIFIC INTEGRITY TESTS
# ==============================================================================


def test_geospatial_context_does_not_alter_xgboost_predictions():
    """Verify that geospatial context is 100% orthogonal to XGBoost risk predictions.

    Predictions must have identical raw_probability, calibrated_probability,
    binary_alert, and risk_score regardless of safe area or familiarity state.
    """
    engine = RiskInferenceEngine()

    dummy_win = {
        "window_id": "test_win_ortho",
        "user_id": "u_test_ortho",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 2.5,
        "speed_std_dev": 0.5,
        "path_distance_m": 300.0,
        "straight_line_displacement_m": 250.0,
        "tortuosity_index": 1.2,
        "entropy_directional": 1.5,
        "latitude": 39.9840,
        "longitude": 116.3180,
    }

    # 1. Prediction without safe area (SAFE_AREA_UNAVAILABLE)
    pred_no_safe = engine.predict_window(dummy_win, horizon_sec=120)

    # 2. Prediction with safe area where location is OUTSIDE safe area
    far_safe_area = {
        "type": "circle",
        "name": "Distant Area",
        "center": [30.0000, 100.0000],
        "radius_m": 100.0,
    }
    pred_outside_safe = engine.predict_window(
        dummy_win, horizon_sec=120, safe_area=far_safe_area
    )

    # 3. Prediction with safe area where location is INSIDE safe area
    near_safe_area = {
        "type": "circle",
        "name": "Local Area",
        "center": [39.9840, 116.3180],
        "radius_m": 500.0,
    }
    pred_inside_safe = engine.predict_window(
        dummy_win, horizon_sec=120, safe_area=near_safe_area
    )

    # Verify states differ as expected
    assert pred_no_safe["safe_area_state"] == "SAFE_AREA_UNAVAILABLE"
    assert pred_outside_safe["safe_area_state"] == "OUTSIDE_SAFE_AREA"
    assert pred_inside_safe["safe_area_state"] == "INSIDE_SAFE_AREA"

    # CRITICAL INVARIANT: XGBoost kinematic scores must be EXACTLY IDENTICAL
    assert pred_no_safe["risk_score"] == pred_outside_safe["risk_score"] == pred_inside_safe["risk_score"]
    assert pred_no_safe["risk_tier"] == pred_outside_safe["risk_tier"] == pred_inside_safe["risk_tier"]
    assert (
        pred_no_safe["trigger_state"]["calibrated_probability"]
        == pred_outside_safe["trigger_state"]["calibrated_probability"]
        == pred_inside_safe["trigger_state"]["calibrated_probability"]
    )
    assert (
        pred_no_safe["trigger_state"]["binary_alert"]
        == pred_outside_safe["trigger_state"]["binary_alert"]
        == pred_inside_safe["trigger_state"]["binary_alert"]
    )


def test_outside_safe_area_does_not_force_high_risk():
    """Verify that being OUTSIDE safe area does NOT automatically elevate risk score to CRITICAL."""
    engine = RiskInferenceEngine()

    dummy_win = {
        "window_id": "test_win_normal",
        "user_id": "u_test_normal",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 1.2,  # Normal walking speed
        "speed_std_dev": 0.2,
        "path_distance_m": 140.0,
        "straight_line_displacement_m": 130.0,
        "tortuosity_index": 1.05,
        "entropy_directional": 0.5,
        "latitude": 40.5000,
        "longitude": 116.3180,
    }

    distant_safe_area = {
        "type": "circle",
        "name": "Home Safe Zone",
        "center": [39.9840, 116.3180],
        "radius_m": 200.0,
    }

    pred_with_safe = engine.predict_window(dummy_win, horizon_sec=120, safe_area=distant_safe_area)
    pred_without_safe = engine.predict_window(dummy_win, horizon_sec=120, safe_area=None)

    assert pred_with_safe["safe_area_state"] == "OUTSIDE_SAFE_AREA"
    # Safe area departure does NOT force CRITICAL tier
    assert pred_with_safe["risk_tier"] != "CRITICAL"
    # Safe area departure does NOT mutate or artificially elevate risk score
    assert pred_with_safe["risk_score"] == pred_without_safe["risk_score"]


def test_disclaimers_and_notes_present():
    """Verify that explicit disclaimers and notes are present in all geospatial context outputs."""
    cfg = GeospatialConfig()
    res = evaluate_geospatial_context(
        latitude=39.9840,
        longitude=116.3180,
        config=cfg,
    )
    assert "safe_area" in res.disclaimers
    assert "familiarity" in res.disclaimers
    assert "orthogonality" in res.disclaimers
    assert "GeoLife dataset does not provide ground-truth safe-zone labels" in res.disclaimers["safe_area"]


# ==============================================================================
# 4. ENGINE INTEGRATION TESTS
# ==============================================================================


def test_predict_telemetry_reading_with_safe_area(circular_safe_area):
    """Verify that predict_telemetry_reading exposes safe_area_state and familiarity_state."""
    engine = RiskInferenceEngine()

    reading = {
        "user_id": "test_user_telemetry",
        "timestamp": "2026-09-17T08:30:00Z",
        "location": {
            "lat": 39.9840,
            "lng": 116.3180,
            "altitude_m": 45.0,
        },
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": 1.2,
            "heading_deg": 45.0,
            "activity_type": "WALKING",
            "battery_pct": 85,
        },
        "signal_status": {"state": "VALID"},
    }

    # Reading with safe area configured
    out = engine.predict_telemetry_reading(reading, safe_area=circular_safe_area)

    assert "safe_area_state" in out
    assert "familiarity_state" in out
    assert "geospatial_context" in out
    assert out["safe_area_state"] == "INSIDE_SAFE_AREA"
    assert out["familiarity_state"] == "NO_HISTORY"  # Cold start user has no history
    assert out["geospatial_context"]["is_safe_area_available"] is True


def test_engine_user_safe_area_persistence(circular_safe_area):
    """Verify set_user_safe_area, get_user_safe_area, and clear_user_safe_area APIs."""
    engine = RiskInferenceEngine()
    u_id = "u_geo_test_999"

    # Initial state
    assert engine.get_user_safe_area(u_id) is None

    # Set safe area
    boundary = engine.set_user_safe_area(u_id, circular_safe_area)
    assert boundary is not None
    assert engine.get_user_safe_area(u_id) is not None

    # Clear safe area
    assert engine.clear_user_safe_area(u_id) is True
    assert engine.get_user_safe_area(u_id) is None

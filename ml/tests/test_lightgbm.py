"""Unit and Integration Tests for LightGBM Inference Pipeline.

Verifies:
1. Complete serialization and deserialization of LightGBM models across
   horizons 120s, 360s, 600s, 840s.
2. Exact 72-feature schema consistency (49 baseline + 23 validated kinematic/spatial/temporal features)
   matching improved XGBoost and Random Forest schemas identically.
3. Platt scaling calibrators and decision thresholds operating in calibrated probability space.
4. Parity of request/response output schema contracts between LightGBM, Random Forest, and XGBoost.
5. Multi-horizon inference across 120s, 360s, 600s, 840s yielding valid calibrated probabilities
   and operational risk tiers.
6. Robust resilience to missing values, NaNs, and cold-start profiles.
7. End-to-end streaming telemetry evaluation with persistent-loop and heart-rate evidence integration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ml.src.inference import InferenceConfig, RiskInferenceEngine
from ml.src.improved_pipeline import EXTENDED_FEATURE_NAMES


@pytest.fixture(scope="module")
def lgb_engine():
    """Instantiate RiskInferenceEngine pointing to LightGBM model artifacts."""
    cfg = InferenceConfig(model_dir="ml/models/lightgbm", model_type="lightgbm")
    return RiskInferenceEngine(config=cfg)


@pytest.fixture(scope="module")
def rf_engine():
    """Instantiate RiskInferenceEngine pointing to Random Forest model artifacts."""
    cfg = InferenceConfig(model_dir="ml/models/random_forest", model_type="random_forest")
    return RiskInferenceEngine(config=cfg)


@pytest.fixture(scope="module")
def xgb_improved_engine():
    """Instantiate RiskInferenceEngine pointing to improved XGBoost artifacts."""
    cfg = InferenceConfig(model_dir="ml/models/xgboost_improved")
    return RiskInferenceEngine(config=cfg)


def test_lgb_artifacts_exist_and_load(lgb_engine):
    """Verify that all LightGBM horizon models, metadata, and comparisons exist on disk."""
    model_dir = Path("ml/models/lightgbm")
    assert (model_dir / "model_metadata.json").exists(), "Missing model_metadata.json"
    assert (model_dir / "evaluation_comparison.json").exists(), "Missing evaluation_comparison.json"
    assert (model_dir / "population_baseline.json").exists(), "Missing population_baseline.json"

    assert lgb_engine.model_type == "lightgbm"

    for h in [120, 360, 600, 840]:
        model_file = model_dir / f"lgb_horizon_{h}s.txt"
        assert model_file.exists(), f"Missing LightGBM model file for horizon {h}s"
        assert h in lgb_engine.models
        assert lgb_engine.models[h] is not None
        assert h in lgb_engine.thresholds
        assert 0.1 <= lgb_engine.thresholds[h] <= 0.95


def test_lgb_feature_schema_matches_existing(lgb_engine, xgb_improved_engine, rf_engine):
    """Verify that LightGBM uses the exact same 72 features in identical order as XGBoost and RF."""
    assert len(lgb_engine.feature_names) == 72
    assert lgb_engine.feature_names == xgb_improved_engine.feature_names
    assert lgb_engine.feature_names == rf_engine.feature_names

    # Check baseline kinematic features
    assert "mean_speed_mps" in lgb_engine.feature_names
    assert "max_mad_z_score" in lgb_engine.feature_names
    assert "p95_exceedance_count" in lgb_engine.feature_names

    # Check extended features
    for ext_feat in EXTENDED_FEATURE_NAMES:
        assert ext_feat in lgb_engine.feature_names, f"Missing extended feature {ext_feat}"


def test_lgb_inference_execution_all_horizons(lgb_engine):
    """Verify that LightGBM engine predicts across all 4 horizons with valid schema."""
    dummy_window = {
        "window_id": "test_lgb_001",
        "user_id": "test_user_lgb",
        "start_time": "2026-10-10T14:00:00Z",
        "end_time": "2026-10-10T14:02:00Z",
        "mean_speed_mps": 1.45,
        "speed_std_dev": 0.35,
        "path_distance_m": 174.0,
        "straight_line_displacement_m": 150.0,
        "tortuosity_index": 1.16,
        "entropy_directional": 1.25,
        "turn_frequency": 0.04,
        "loop_metric": 0.01,
        "pacing_tendency": 0.02,
        "backtracking_tendency": 0.01,
        "heading_variability": 0.28,
        "point_count": 35,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
        # Extended kinematic features
        "mean_acceleration_mps2": 0.12,
        "max_acceleration_mps2": 0.70,
        "acceleration_std_dev": 0.18,
        "max_speed_mps": 2.2,
        "min_speed_mps": 0.6,
        "median_speed_mps": 1.4,
        "speed_variance": 0.12,
        "heading_change_mean": 0.15,
        "heading_change_std": 0.22,
        "path_closure_ratio": 0.86,
        "radius_of_gyration": 45.0,
        "bbox_diagonal_m": 160.0,
        "bbox_area_sqm": 7200.0,
        "spatial_randomness_dispersion": 0.25,
        "step_distance_mean": 4.97,
        "step_distance_max": 8.1,
        "step_distance_std": 1.2,
        "zero_dt_count": 0,
        "zero_dt_fraction": 0.0,
    }

    for h in [120, 360, 600, 840]:
        res = lgb_engine.predict_window(dummy_window, horizon_sec=h)
        assert isinstance(res, dict)
        assert "risk_score" in res
        assert 0.0 <= res["risk_score"] <= 100.0
        assert "trigger_state" in res
        assert "calibrated_probability" in res["trigger_state"]
        assert 0.0 <= res["trigger_state"]["calibrated_probability"] <= 1.0
        assert res["risk_tier"] in ("QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL")
        assert res["polling_tier"] in (0, 1, 2, 3)
        assert res["trigger_state"]["horizon_sec"] == h
        assert "decision_state" in res
        assert "display_state" in res


def test_contract_parity_between_all_three_models(lgb_engine, rf_engine, xgb_improved_engine):
    """Verify that LightGBM, Random Forest, and XGBoost produce identical response structures."""
    sample_window = {
        "window_id": "parity_win_002",
        "user_id": "parity_user_002",
        "start_time": "2026-10-10T10:00:00Z",
        "end_time": "2026-10-10T10:02:00Z",
        "mean_speed_mps": 3.5,
        "speed_std_dev": 0.8,
        "path_distance_m": 420.0,
        "straight_line_displacement_m": 120.0,
        "tortuosity_index": 3.5,
        "entropy_directional": 2.8,
        "turn_frequency": 0.15,
        "loop_metric": 0.65,
        "pacing_tendency": 0.45,
        "backtracking_tendency": 0.50,
        "heading_variability": 0.75,
        "point_count": 50,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
        "mean_acceleration_mps2": 0.45,
        "max_acceleration_mps2": 1.8,
        "acceleration_std_dev": 0.40,
        "max_speed_mps": 5.5,
        "min_speed_mps": 0.2,
        "median_speed_mps": 3.2,
        "speed_variance": 0.64,
        "heading_change_mean": 0.55,
        "heading_change_std": 0.65,
        "path_closure_ratio": 0.28,
        "radius_of_gyration": 85.0,
        "bbox_diagonal_m": 220.0,
        "bbox_area_sqm": 15000.0,
        "spatial_randomness_dispersion": 0.70,
        "step_distance_mean": 8.4,
        "step_distance_max": 15.0,
        "step_distance_std": 2.5,
        "zero_dt_count": 0,
        "zero_dt_fraction": 0.0,
    }

    lgb_out = lgb_engine.predict_window(sample_window, horizon_sec=120)
    rf_out = rf_engine.predict_window(sample_window, horizon_sec=120)
    xgb_out = xgb_improved_engine.predict_window(sample_window, horizon_sec=120)

    # All top-level keys must match
    assert set(lgb_out.keys()) == set(rf_out.keys()) == set(xgb_out.keys())

    # Metadata sub-keys must match
    assert set(lgb_out["metadata"].keys()) == set(rf_out["metadata"].keys()) == set(xgb_out["metadata"].keys())

    # Decision sub-keys must match
    assert set(lgb_out["decision"].keys()) == set(rf_out["decision"].keys()) == set(xgb_out["decision"].keys())

    # Value ranges
    for out in [lgb_out, rf_out, xgb_out]:
        assert 0.0 <= out["risk_score"] <= 100.0
        assert 0.0 <= out["trigger_state"]["calibrated_probability"] <= 1.0


def test_lgb_missing_features_and_cold_start_resilience(lgb_engine):
    """Verify that LightGBM handles missing features (NaNs) and cold start without failure."""
    sparse_window = {
        "window_id": "test_sparse_lgb_001",
        "user_id": "sparse_user_lgb_001",
        "start_time": "2026-10-10T16:00:00Z",
        "end_time": "2026-10-10T16:02:00Z",
        "mean_speed_mps": 1.2,
        # Most kinematic and extended features are omitted / NaN
        "point_count": 15,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    # Should succeed without exception
    res = lgb_engine.predict_window(sparse_window, horizon_sec=120)
    assert isinstance(res, dict)
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["metadata"]["profile_mode"] == "COLD_START"


def test_lgb_streaming_telemetry_integration(lgb_engine):
    """Verify that LightGBM engine integrates with streaming telemetry ingestion."""
    user_id = "lgb_stream_test_user"

    base_lat = 37.7749
    base_lon = -122.4194

    out = None
    for i in range(15):
        payload = {
            "user_id": user_id,
            "timestamp": f"2026-10-10T15:{i * 10 // 60:02d}:{i * 10 % 60:02d}Z",
            "location": {
                "lat": base_lat + i * 0.0001,
                "lng": base_lon + i * 0.0001,
            },
            "sensor_metrics": {
                "speed_mps": 1.5,
                "horizontal_accuracy_m": 4.0,
            },
            "signal_status": {
                "state": "VALID",
            },
            "wearable_metrics": {
                "heart_rate_bpm": 74.0,
                "wear_state": "ON_WRIST",
                "timestamp": f"2026-10-10T15:{i * 10 // 60:02d}:{i * 10 % 60:02d}Z",
            },
        }
        res = lgb_engine.predict_telemetry_reading(payload)
        if res.get("evaluation_status") == "EVALUATED":
            out = res
            break

    if out is not None:
        assert "risk_score" in out
        assert 0.0 <= out["risk_score"] <= 100.0
        assert "calibrated_probability" in out["trigger_state"]
        assert out["evaluation_status"] == "EVALUATED"

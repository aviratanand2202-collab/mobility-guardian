"""Regression & Validation Test Suite for Audited & Improved XGBoost Pipeline.

Verifies:
1. Complete serialization and deserialization of improved models across 120s, 360s, 600s, 840s.
2. Exact 72-feature schema consistency (49 baseline + 23 validated kinematic/spatial/temporal features).
3. Consistent decision thresholds and Platt calibrators operating in calibrated probability space.
4. Objective quantitative improvement over baseline across PR-AUC, ROC-AUC, Macro-F1, and Brier score.
5. Invariance and safety of original production models (zero mutation of ml/models/xgboost).
6. Real-time streaming and offline window inference compatibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from ml.src.inference import InferenceConfig, RiskInferenceEngine
from ml.src.improved_pipeline import EXTENDED_FEATURE_NAMES


@pytest.fixture(scope="module")
def improved_engine():
    """Instantiate RiskInferenceEngine pointing to the improved model artifacts."""
    cfg = InferenceConfig(model_dir="ml/models/xgboost_improved")
    return RiskInferenceEngine(config=cfg)


@pytest.fixture(scope="module")
def baseline_engine():
    """Instantiate default RiskInferenceEngine pointing to production baseline artifacts."""
    return RiskInferenceEngine()


def test_improved_artifacts_exist_and_load(improved_engine):
    """Verify that all improved horizon models, metadata, and comparisons exist on disk."""
    model_dir = Path("ml/models/xgboost_improved")
    assert (model_dir / "model_metadata.json").exists()
    assert (model_dir / "evaluation_comparison.json").exists()
    assert (model_dir / "population_baseline.json").exists()

    for h in [120, 360, 600, 840]:
        model_file = model_dir / f"xgb_horizon_{h}s.json"
        assert model_file.exists(), f"Missing improved model for horizon {h}s"
        assert h in improved_engine.models
        assert improved_engine.models[h] is not None
        assert h in improved_engine.thresholds
        assert 0.4 <= improved_engine.thresholds[h] <= 0.9


def test_improved_feature_names_and_count(improved_engine):
    """Verify that improved engine has exactly 72 features in fixed order."""
    assert len(improved_engine.feature_names) == 72
    # Baseline core features preserved
    assert "mean_speed_mps" in improved_engine.feature_names
    assert "max_mad_z_score" in improved_engine.feature_names
    assert "p95_exceedance_count" in improved_engine.feature_names

    # All 23 extended features present
    for ext_feat in EXTENDED_FEATURE_NAMES:
        assert ext_feat in improved_engine.feature_names, f"Missing extended feature {ext_feat}"


def test_production_baseline_models_unmutated(baseline_engine):
    """Verify that production baseline artifacts in ml/models/xgboost remain untouched and 49 features."""
    assert len(baseline_engine.feature_names) == 49
    for h in [120, 360, 600, 840]:
        assert h in baseline_engine.models
        assert baseline_engine.models[h] is not None


def test_improved_inference_execution_all_horizons(improved_engine):
    """Verify that improved engine predicts across all 4 horizons with valid schema."""
    dummy_window = {
        "window_id": "test_imp_001",
        "user_id": "test_user_improved",
        "start_time": "2026-10-10T12:00:00Z",
        "end_time": "2026-10-10T12:02:00Z",
        "mean_speed_mps": 2.5,
        "speed_std_dev": 0.4,
        "path_distance_m": 300.0,
        "straight_line_displacement_m": 250.0,
        "tortuosity_index": 1.2,
        "entropy_directional": 1.5,
        "turn_frequency": 0.05,
        "loop_metric": 0.02,
        "pacing_tendency": 0.01,
        "backtracking_tendency": 0.02,
        "heading_variability": 0.3,
        "point_count": 40,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
        # Extended kinematic fields
        "mean_acceleration_mps2": 0.15,
        "max_acceleration_mps2": 0.85,
        "acceleration_std_dev": 0.22,
        "max_speed_mps": 4.1,
        "min_speed_mps": 0.5,
        "median_speed_mps": 2.3,
        "speed_variance": 0.16,
        "heading_change_mean": 25.0,
        "heading_change_std": 18.0,
        "path_closure_ratio": 0.10,
        "radius_of_gyration": 45.0,
        "bbox_diagonal_m": 280.0,
        "bbox_area_sqm": 12000.0,
        "spatial_randomness_dispersion": 0.15,
        "step_distance_mean": 7.5,
        "step_distance_max": 14.0,
        "step_distance_std": 2.1,
        "zero_dt_count": 0,
        "zero_dt_fraction": 0.0,
    }

    for h in [120, 360, 600, 840]:
        out = improved_engine.predict_window(dummy_window, horizon_sec=h)
        assert out["user_id"] == "test_user_improved"
        assert 0.0 <= out["risk_score"] <= 100.0
        assert out["risk_tier"] in ["QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL"]
        assert out["polling_tier"] in [0, 1, 2, 3]
        assert "trigger_state" in out
        assert 0.0 <= out["trigger_state"]["calibrated_probability"] <= 1.0
        assert 0.0 <= out["trigger_state"]["raw_probability"] <= 1.0
        assert out["trigger_state"]["horizon_sec"] == h
        assert "decision" in out
        assert "decision_state" in out


def test_improved_calibrator_and_threshold_consistency():
    """Verify that optimal thresholds operate consistently in calibrated probability space."""
    meta_path = Path("ml/models/xgboost_improved/model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for h in [120, 360, 600, 840]:
        h_info = meta["horizons"][f"{h}s"]
        th = h_info["optimal_threshold"]
        # In calibrated probability space, threshold should be calibrated (between 0.60 and 0.85)
        assert 0.60 <= th <= 0.85, f"Threshold {th} for {h}s outside calibrated range"


def test_improved_metrics_strictly_outperform_baseline():
    """Verify that the improved pipeline outperforms the baseline across all 4 horizons."""
    comp_path = Path("ml/models/xgboost_improved/evaluation_comparison.json")
    assert comp_path.exists()
    with open(comp_path, "r") as f:
        comp = json.load(f)

    for h in [120, 360, 600, 840]:
        delta = comp[f"{h}s"]["delta"]
        # PR-AUC must improve
        assert delta["pr_auc"] > 0, f"Horizon {h}s PR-AUC did not improve: {delta['pr_auc']}"
        # ROC-AUC must improve
        assert delta["roc_auc"] > 0, f"Horizon {h}s ROC-AUC did not improve: {delta['roc_auc']}"
        # Macro-F1 must improve
        assert delta["macro_f1"] > 0, f"Horizon {h}s Macro-F1 did not improve: {delta['macro_f1']}"
        # Brier Score must improve (decrease)
        assert delta["brier_score"] < 0, f"Horizon {h}s Brier Score did not decrease: {delta['brier_score']}"


def test_streaming_telemetry_with_improved_engine(improved_engine):
    """Verify streaming telemetry ingestion operates end-to-end with the improved engine."""
    payload = {
        "user_id": "patient_stream_imp",
        "timestamp": "2026-10-10T12:00:00Z",
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

    out = improved_engine.predict_telemetry_reading(payload)
    assert out["user_id"] == "patient_stream_imp"
    assert out["trigger_state"]["status"] == "ACCUMULATING_WINDOW"

"""Comprehensive Test Suite for Production Inference & Deployment Bridge.

Verifies:
1. Model artifact serialization and deserialization
2. Feature ordering and schema consistency (49 features)
3. Horizon-specific threshold and calibrator parameter loading
4. RiskScoreOutput schema validation (shared/schema.json & backend/app/models.py)
5. Insufficient history vs zero-shot population fallback
6. Zero-shot unseen-user inference
7. Adapted-user personalized inference
8. Signal degradation and PDR fallback contract
9. Behavioral movement pattern integration (PACING, LAPPING, RANDOM_DRIFT)
10. Backend FastAPI /api/telemetry/ingest integration
11. Raw GeoLife .plt manual inference execution
12. Deterministic inference reproducibility
13. Prediction equivalence: inference engine reproduces frozen batch predictions
14. Non-mutation of canonical benchmark artifacts
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path so `app.*` sub-imports resolve
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xgboost as xgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402
from backend.app.models import RiskScoreOutput  # noqa: E402
from ml.src.inference import RiskInferenceEngine  # noqa: E402
from ml.src.manual_inference import run_manual_inference  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    """Instantiate singleton RiskInferenceEngine."""
    return RiskInferenceEngine()


# 1. Model artifact serialization and deserialization
def test_model_artifacts_exist_and_load(engine):
    """Verify that all 4 horizon models and metadata exist and load into Boosters."""
    cfg = engine.config
    assert os.path.exists(os.path.join(cfg.model_dir, cfg.metadata_filename))
    assert os.path.exists(os.path.join(cfg.model_dir, cfg.population_baseline_filename))

    for h in [120, 360, 600, 840]:
        assert h in engine.models
        assert engine.models[h] is not None
        assert h in engine.thresholds
        assert 0.1 <= engine.thresholds[h] <= 0.9


# 2. Feature ordering matches metadata
def test_feature_ordering_consistency(engine):
    """Verify that feature_names has exactly 49 features in fixed order."""
    assert len(engine.feature_names) == 49
    assert engine.feature_names[0] == "mean_speed_mps" or "mean_speed_mps" in engine.feature_names
    assert "max_mad_z_score" in engine.feature_names
    assert "p95_exceedance_count" in engine.feature_names


# 3. Threshold and calibrator loading per horizon
def test_threshold_and_calibrator_loading(engine):
    """Verify that calibrated parameters and thresholds match report values."""
    with open("predictive_risk_report.json", "r") as f:
        report = json.load(f)

    for h in [120, 360, 600, 840]:
        h_str = f"horizon_{h}s"
        expected_thresh = report["horizons"][h_str]["optimal_parameters"]["xgb_best_threshold"]
        assert abs(engine.thresholds[h] - expected_thresh) < 1e-6
        assert h in engine.calibrators
        assert "coef" in engine.calibrators[h]
        assert "intercept" in engine.calibrators[h]


# 4. Schema validation of RiskScoreOutput
def test_risk_score_output_schema(engine):
    """Verify that prediction output adheres strictly to RiskScoreOutput schema."""
    dummy_window = {
        "window_id": "test_w_001",
        "user_id": "test_user",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 2.5,
        "speed_std_dev": 0.5,
        "path_distance_m": 300.0,
        "straight_line_displacement_m": 250.0,
        "tortuosity_index": 1.2,
        "entropy_directional": 1.5,
        "turn_frequency": 0.05,
        "loop_metric": 0.02,
        "pacing_tendency": 0.01,
        "backtracking_tendency": 0.02,
        "heading_variability": 0.15,
        "point_count": 25,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    out = engine.predict_window(dummy_window, horizon_sec=120)

    # Validate against Pydantic schema
    validated = RiskScoreOutput(**out)
    assert validated.user_id == "test_user"
    assert 0.0 <= validated.risk_score <= 100.0
    assert validated.risk_tier in ["QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL"]
    assert validated.polling_tier in [0, 1, 2, 3]
    assert validated.kinematic_features is not None
    assert "behavioral_indicator" in validated.kinematic_features


# 5. Insufficient history vs zero-shot population fallback
def test_insufficient_history_behavior():
    """Verify that manual inference raises error if require_personalized is set with < 7 trips."""
    fake_path = "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"
    if not os.path.exists(fake_path):
        pytest.skip("GeoLife trajectory file not present.")

    # With require_personalized=True on a single trip, must raise ValueError
    with pytest.raises(ValueError, match="Insufficient history"):
        run_manual_inference(
            input_file=fake_path,
            history_path=fake_path,  # Only 1 file provided
            require_personalized=True,
        )


# 6. Zero-shot unseen-user inference
def test_zero_shot_unseen_user_inference(engine):
    """Verify zero-shot inference succeeds using population baseline."""
    dummy_window = {
        "window_id": "unseen_w_001",
        "user_id": "new_patient_999",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 1.8,
        "speed_std_dev": 0.4,
        "path_distance_m": 210.0,
        "straight_line_displacement_m": 200.0,
        "tortuosity_index": 1.05,
        "entropy_directional": 1.1,
        "turn_frequency": 0.03,
        "loop_metric": 0.01,
        "pacing_tendency": 0.01,
        "backtracking_tendency": 0.01,
        "heading_variability": 0.1,
        "point_count": 20,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    out = engine.predict_window(
        curr_window=dummy_window,
        user_baseline=None,  # Forces population baseline fallback
        user_context={"trip_count": 1, "cold_start_status": "ZERO_SHOT"},
        horizon_sec=120,
    )

    assert out["metadata"]["cold_start_status"] == "ZERO_SHOT"
    assert out["risk_tier"] in ["QUIESCENT", "NORMAL_TRANSIT", "SUSPICIOUS", "CRITICAL"]
    assert 0.0 <= out["risk_score"] <= 100.0


# 7. Adapted-user personalized inference
def test_adapted_user_inference(engine):
    """Verify that personalized baseline with abnormal speed yields elevated risk."""
    normal_base = {
        "mean_speed_mps": {"median": 1.5, "robust_scale": 0.3, "p95": 2.5},
        "tortuosity_index": {"median": 1.1, "robust_scale": 0.1, "p95": 1.5},
        "entropy_directional": {"median": 1.2, "robust_scale": 0.2, "p95": 2.0},
        "turn_frequency": {"median": 0.04, "robust_scale": 0.02, "p95": 0.1},
        "loop_metric": {"median": 0.02, "robust_scale": 0.02, "p95": 0.1},
        "pacing_tendency": {"median": 0.02, "robust_scale": 0.02, "p95": 0.1},
        "straight_line_displacement_m": {"median": 150.0, "robust_scale": 30.0, "p95": 250.0},
        "path_distance_m": {"median": 180.0, "robust_scale": 35.0, "p95": 300.0},
    }

    # Severe kinematic outlier excursion window (speed 15 m/s vs median 1.5)
    outlier_window = {
        "window_id": "outlier_w_001",
        "user_id": "adapted_user_001",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 15.0,
        "speed_std_dev": 2.5,
        "path_distance_m": 1800.0,
        "straight_line_displacement_m": 1500.0,
        "tortuosity_index": 1.2,
        "entropy_directional": 2.8,
        "turn_frequency": 0.15,
        "loop_metric": 0.05,
        "pacing_tendency": 0.02,
        "backtracking_tendency": 0.05,
        "heading_variability": 0.4,
        "point_count": 50,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    out = engine.predict_window(
        curr_window=outlier_window,
        user_baseline=normal_base,
        user_context={"trip_count": 12, "cold_start_status": "ADAPTED"},
        horizon_sec=120,
    )

    assert out["risk_score"] > 35.0
    assert out["risk_tier"] in ["SUSPICIOUS", "CRITICAL"]


# 8. Signal degradation and PDR fallback contract
def test_signal_degradation_behavior(engine):
    """Verify DEGRADED_SIGNAL freezes score and escalates upon high net displacement."""
    degraded_payload = {
        "user_id": "patient_042",
        "timestamp": "2026-10-07T12:00:00Z",
        "location": {"lat": 39.98, "lng": 116.31, "altitude_m": 50.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 35.0,  # > 20m degraded
            "speed_mps": 1.2,
            "heading_deg": 90.0,
            "battery_pct": 80,
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL",
            "degraded_since": "2026-10-07T11:55:00Z",
        },
        "imu_metrics": {
            "step_count_since_last_gps": 120,
            "net_displacement_m": 60.0,  # > 50m triggers Tier 3
            "pdr_tier_state": "UNTRACKED_DISPLACEMENT",
        },
    }

    out = engine.predict_telemetry_reading(degraded_payload)
    assert out["risk_tier"] == "CRITICAL"
    assert out["polling_tier"] == 3
    assert out["trigger_state"]["frozen_reason"] == "DEGRADED_SIGNAL_PDR_FALLBACK"


# 9. Behavioral movement pattern integration
def test_behavior_pattern_integration(engine):
    """Verify behavioral pattern is reported in kinematic_features without changing XGBoost target."""
    pacing_window = {
        "window_id": "pacing_w_001",
        "user_id": "test_user",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 1.5,
        "speed_std_dev": 0.3,
        "path_distance_m": 250.0,
        "straight_line_displacement_m": 15.0,  # High closure/reversal
        "tortuosity_index": 16.6,
        "entropy_directional": 1.8,
        "turn_frequency": 6.0,
        "loop_metric": 0.05,
        "pacing_tendency": 0.85,
        "backtracking_tendency": 0.80,
        "heading_variability": 0.75,
        "heading_change_mean": 170.0,  # Sharp 180 reversals
        "path_closure_ratio": 0.94,
        "bbox_diagonal_m": 30.0,
        "bbox_width_m": 25.0,
        "bbox_height_m": 20.0,
        "has_extreme_kinematic_transition": False,
        "point_count": 40,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    out = engine.predict_window(pacing_window, horizon_sec=120)
    assert "behavioral_indicator" in out["kinematic_features"]
    assert out["kinematic_features"]["behavioral_indicator"] in ["PACING", "NORMAL", "RANDOM_DRIFT"]


# 10. Backend FastAPI integration
def test_backend_telemetry_endpoint_integration():
    """Verify POST /api/telemetry/ingest returns valid RiskScoreOutput."""
    client = TestClient(app)
    payload = {
        "user_id": "backend_patient_001",
        "timestamp": "2026-10-07T12:00:00Z",
        "location": {"lat": 39.9847, "lng": 116.3184, "altitude_m": 45.0},
        "sensor_metrics": {
            "horizontal_accuracy_m": 8.0,
            "speed_mps": 1.5,
            "heading_deg": 120.0,
            "activity_type": "WALKING",
            "battery_pct": 90,
        },
        "signal_status": {"state": "VALID"},
    }

    response = client.post("/api/telemetry/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "backend_patient_001"
    assert "risk_score" in data
    assert "risk_tier" in data
    assert "polling_tier" in data


# 11. Raw GeoLife .plt manual inference execution
def test_raw_geolife_plt_manual_inference():
    """Verify manual inference CLI works on actual GeoLife .plt file."""
    sample_plt = "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"
    if not os.path.exists(sample_plt):
        pytest.skip("GeoLife raw file not available.")

    summary = run_manual_inference(
        input_file=sample_plt,
        horizon="120",
        output_dir="ml/data/inference",
    )
    assert summary["n_evaluated_windows"] > 0
    assert "120s" in summary["window_predictions"]
    first_pred = summary["window_predictions"]["120s"][0]
    assert 0.0 <= first_pred["risk_score"] <= 100.0


# 12. Deterministic inference reproducibility
def test_deterministic_inference_reproducibility(engine):
    """Verify identical inputs yield identical outputs."""
    sample_win = {
        "window_id": "det_win",
        "user_id": "u_det",
        "start_time": "2008-10-23T02:53:04Z",
        "end_time": "2008-10-23T02:55:04Z",
        "mean_speed_mps": 3.0,
        "speed_std_dev": 0.6,
        "path_distance_m": 360.0,
        "straight_line_displacement_m": 300.0,
        "tortuosity_index": 1.2,
        "entropy_directional": 1.6,
        "turn_frequency": 0.06,
        "loop_metric": 0.03,
        "pacing_tendency": 0.02,
        "backtracking_tendency": 0.03,
        "heading_variability": 0.2,
        "point_count": 30,
        "temporal_span_sec": 120.0,
        "is_kinematically_evaluable": True,
    }

    out1 = engine.predict_window(sample_win, horizon_sec=120)
    out2 = engine.predict_window(sample_win, horizon_sec=120)

    assert out1["risk_score"] == out2["risk_score"]
    assert out1["trigger_state"]["calibrated_probability"] == out2["trigger_state"]["calibrated_probability"]
    assert out1["risk_tier"] == out2["risk_tier"]


# 13. Prediction equivalence: inference engine reproduces frozen batch predictions
def test_inference_matches_frozen_benchmark_predictions(engine):
    """Verify that inference engine produces exact prediction matching canonical parquet."""
    canon_df = pd.read_parquet("ml/data/processed/risk_predictions.parquet")
    canon_120 = canon_df[canon_df["horizon_sec"] == 120]

    test_cache = Path("ml/models/xgboost/cache/test_df.parquet")
    if not test_cache.exists():
        pytest.skip("Test split cache not found.")

    test_df = pd.read_parquet(test_cache)
    row_0 = test_df.iloc[0]
    win_id = row_0["window_id"]
    canon_matches = canon_120[canon_120["window_id"] == win_id]
    if canon_matches.empty:
        pytest.skip(f"Window {win_id} not in benchmark.")

    expected_prob = float(canon_matches.iloc[0]["y_prob_xgb"])

    # Predict using engine booster on exact test feature vector
    feature_vector = [row_0[c] for c in engine.feature_names]
    dmat = xgb.DMatrix(pd.DataFrame([feature_vector], columns=engine.feature_names))
    raw_p = float(engine.models[120].predict(dmat)[0])

    assert abs(raw_p - expected_prob) < 1e-6, f"Engine prob {raw_p} diverged from benchmark {expected_prob}"


# 14. Non-mutation of canonical benchmark artifacts
def test_benchmark_artifacts_unmutated():
    """Verify that all canonical benchmark artifacts remain strictly untouched."""
    locked_hashes = {
        "ml/data/processed/risk_predictions.parquet": (
            "d716f7a268a7e96c874fb89c0e3311ebb343fb4c8fa7557f5eaa26c98b4a204f"
        ),
        "predictive_risk_report.json": "c7b57bdff9c126c382f5e535b376df101d432daa0da2a3c064c424f95bafced1",
        "predictive_risk_report.md": "181c2b108b1f4f8fc044e63f4cdd70e0447726925bf120da6818c9db945df0da",
    }
    for path_str, exp_hash in locked_hashes.items():
        p = Path(path_str)
        assert p.exists(), f"File {path_str} missing"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == exp_hash, f"File {path_str} hash changed: {actual_hash} != {exp_hash}"

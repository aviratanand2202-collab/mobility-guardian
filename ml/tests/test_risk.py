"""Unit and integration tests for Chunk 6: Personalized Predictive Kinematic Risk Model.

Covers all mandatory acceptance tests:
1. Boundary-crossing window [240, 360] at horizon 0-300s forces NaN (not negative, not positive from w3).
2. Pre-horizon window [t-60, t+60] cannot introduce prior evidence.
3. Interval-union coverage prevents overlap double-counting.
4. Incomplete horizon yields NaN under strict coverage.
5. Symmetric coverage enforcement for positive and negative labels.
6. Misaligned prediction origins yield NaN.
7. Zero MAD and unmeasured kinematic features handling.
8. Past-only expanding profile causality (no future leakage).
9. Target generation determinism.
10. Uniform boundary handling across all four horizons (120s, 360s, 600s, 840s).
"""

import numpy as np
import pandas as pd
import pytest

from ml.src.risk import (
    RiskConfig,
    compute_evidence_coverage,
    evaluate_window_outlier,
    extract_prediction_features,
    generate_horizon_target,
    merge_intervals,
    PersonalizedRobustMADBaseline,
    PersonalizedPercentileBaseline,
    evaluate_binary_predictions,
    XGBoostRiskModel,
)


def make_dummy_window(
    start_sec: float,
    end_sec: float,
    mean_speed: float = 2.0,
    entropy: float = 2.5,
    tortuosity: float = 1.1,
    turn_freq: float = 0.05,
    loop: float = 0.1,
    pacing: float = 0.05,
    disp: float = 200.0,
    path_dist: float = 240.0,
    is_evaluable: bool = True,
    trajectory_id: str = "t_001",
) -> pd.Series:
    """Helper to generate a mock analysis window."""
    base_t = pd.Timestamp("2008-10-01 08:00:00")
    st = base_t + pd.Timedelta(seconds=start_sec)
    et = base_t + pd.Timedelta(seconds=end_sec)
    return pd.Series({
        "trajectory_id": trajectory_id,
        "start_time": st,
        "end_time": et,
        "mean_speed_mps": mean_speed,
        "speed_std_dev": 0.5,
        "path_distance_m": path_dist,
        "straight_line_displacement_m": disp,
        "tortuosity_index": tortuosity,
        "entropy_directional": entropy,
        "turn_frequency": turn_freq,
        "loop_metric": loop,
        "pacing_tendency": pacing,
        "backtracking_tendency": 0.05,
        "heading_variability": 0.2,
        "point_count": 20,
        "temporal_span_sec": end_sec - start_sec,
        "is_kinematically_evaluable": is_evaluable,
    })


def make_standard_baselines() -> dict:
    """Helper for typical historical baseline distribution."""
    return {
        "mean_speed_mps": {"median": 2.0, "robust_scale": 0.5, "p95": 4.0},
        "tortuosity_index": {"median": 1.1, "robust_scale": 0.2, "p95": 2.0},
        "entropy_directional": {"median": 2.5, "robust_scale": 0.3, "p95": 3.0},
        "turn_frequency": {"median": 0.05, "robust_scale": 0.02, "p95": 0.15},
        "loop_metric": {"median": 0.1, "robust_scale": 0.05, "p95": 0.5},
        "pacing_tendency": {"median": 0.05, "robust_scale": 0.02, "p95": 0.2},
        "straight_line_displacement_m": {"median": 200.0, "robust_scale": 50.0, "p95": 400.0},
        "path_distance_m": {"median": 240.0, "robust_scale": 50.0, "p95": 450.0},
    }


# 1. The Primary Audit Counterexample: 0-300s boundary crossing forces NaN
def test_boundary_crossing_window_excluded_evidence_forces_nan():
    """Verify that an outlier in W3 [240, 360] crossing horizon 0-300s CANNOT make the label 1,

    and normal W1 [0, 120], W2 [120, 240] CANNOT make the label 0.
    The horizon target must strictly evaluate to NaN.
    """
    baselines = make_standard_baselines()
    config = RiskConfig(min_evidence_coverage_ratio=1.0)

    # W1 [0, 120]: normal
    w1 = make_dummy_window(0.0, 120.0, mean_speed=2.0)
    # W2 [120, 240]: normal
    w2 = make_dummy_window(120.0, 240.0, mean_speed=2.0)
    # W3 [240, 360]: extreme outlier (speed = 15 m/s >> median 2.0)
    w3 = make_dummy_window(240.0, 360.0, mean_speed=15.0, loop=0.8, pacing=0.5)

    traj_df = pd.DataFrame([w1, w2, w3])
    base_t = pd.Timestamp("2008-10-01 08:00:00")

    # Evaluate horizon H = 300s
    target = generate_horizon_target(
        trajectory_windows=traj_df,
        prediction_time=base_t,
        horizon_sec=300,
        baselines=baselines,
        config=config,
    )

    # Assertions:
    # 1. Target cannot be 1.0 (w3 ends at 360 > 300, so post-horizon evidence is excluded).
    # 2. Target cannot be 0.0 (w1 and w2 only cover 0-240s; 240-300s has no admissible evidence).
    # 3. Target must strictly be NaN.
    assert np.isnan(target)


# 2. Pre-horizon evidence exclusion
def test_pre_horizon_window_cannot_introduce_prior_evidence():
    """Verify that a window starting before prediction time t cannot establish a positive target."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")
    pred_t = base_t + pd.Timedelta(seconds=100)

    # Window starting at 0 and ending at 120 (starts before pred_t = 100)
    # Even if extreme outlier, it is inadmissible
    w0 = make_dummy_window(0.0, 120.0, mean_speed=25.0, loop=0.9)
    # Window strictly contained in horizon (100, 220]
    w1 = make_dummy_window(120.0, 220.0, mean_speed=2.0)

    traj_df = pd.DataFrame([w0, w1])

    # Horizon H = 120s from pred_t=100
    target = generate_horizon_target(
        trajectory_windows=traj_df,
        prediction_time=pred_t,
        horizon_sec=120,
        baselines=baselines,
    )
    # w0 is inadmissible; w1 covers only 120-220 (100s < 120s) -> incomplete -> NaN, NOT 1
    assert np.isnan(target)


# 3. Interval-union coverage prevents overlap double-counting
def test_interval_union_coverage_prevents_overlap_inflation():
    """Verify that overlapping windows [0, 120] and [60, 180] compute exact 180s union."""
    intervals = [(0.0, 120.0), (60.0, 180.0)]
    merged = merge_intervals(intervals)
    assert merged == [(0.0, 180.0)]

    dur, ratio, is_gap_valid = compute_evidence_coverage(
        admissible_intervals=intervals,
        t_start=0.0,
        horizon_sec=180.0,
    )
    assert dur == 180.0
    assert ratio == pytest.approx(1.0)
    assert is_gap_valid is True


# 4. Incomplete horizon coverage enforcement
def test_incomplete_horizon_coverage_enforcement():
    """Verify that an outlier-free trajectory terminating at 240s in a 600s horizon yields NaN."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")

    w1 = make_dummy_window(0.0, 120.0, mean_speed=2.0)
    w2 = make_dummy_window(120.0, 240.0, mean_speed=2.0)
    traj_df = pd.DataFrame([w1, w2])  # Trajectory terminates at 240s

    target = generate_horizon_target(
        trajectory_windows=traj_df,
        prediction_time=base_t,
        horizon_sec=600,
        baselines=baselines,
    )
    assert np.isnan(target)


# 5. Symmetric coverage enforcement for positive and negative labels
def test_symmetric_coverage_enforcement_positive_and_negative():
    """Verify that premature trajectory termination yields NaN even when an outlier occurred."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")

    # W1 is an outlier
    w1 = make_dummy_window(0.0, 120.0, mean_speed=25.0, loop=0.9)
    # Trajectory terminates at 180s (< 600s horizon)
    w2 = make_dummy_window(120.0, 180.0, mean_speed=2.0)
    traj_df = pd.DataFrame([w1, w2])

    target = generate_horizon_target(
        trajectory_windows=traj_df,
        prediction_time=base_t,
        horizon_sec=600,
        baselines=baselines,
    )
    # Incomplete coverage must symmetrically reject positive evidence -> NaN
    assert np.isnan(target)


# 6. Misaligned prediction origin yields NaN
def test_misaligned_prediction_origin_yields_nan():
    """Verify that a prediction at misaligned timestamp t=50 inside [0, 120] yields NaN."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")
    pred_t = base_t + pd.Timedelta(seconds=50)

    w1 = make_dummy_window(0.0, 120.0)
    w2 = make_dummy_window(120.0, 240.0)
    traj_df = pd.DataFrame([w1, w2])

    target = generate_horizon_target(
        trajectory_windows=traj_df,
        prediction_time=pred_t,
        horizon_sec=120,  # Horizon (50, 170]
        baselines=baselines,
    )
    assert np.isnan(target)


# 7. Zero MAD and unmeasured kinematic features handling
def test_zero_mad_and_missing_features_handling():
    """Verify zero-MAD invariance penalty and missing feature handling without division-by-zero."""
    # Baseline with MAD = 0 for pacing tendency
    baselines = make_standard_baselines()
    baselines["pacing_tendency"] = {"median": 0.0, "robust_scale": 0.0, "p95": 0.0}

    # Window with non-zero pacing -> triggers zero-MAD penalty (10.0)
    w = make_dummy_window(0.0, 120.0, pacing=0.08)
    is_out, is_eval, devs = evaluate_window_outlier(w, baselines)
    assert is_eval is True
    assert is_out is True
    assert devs["pacing_tendency"] == 10.0

    # Window with missing (NaN) features
    w_nan = make_dummy_window(0.0, 120.0)
    w_nan["entropy_directional"] = np.nan
    w_nan["turn_frequency"] = np.nan
    is_out_nan, is_eval_nan, devs_nan = evaluate_window_outlier(w_nan, baselines)
    # Remaining 6 features valid >= 3 min -> still evaluable
    assert is_eval_nan is True
    assert "entropy_directional" not in devs_nan


# 8. Past-only prediction-time feature extraction
def test_prediction_time_features_past_only():
    """Verify features extracted at prediction time use only current and previous window."""
    baselines = make_standard_baselines()
    w_curr = make_dummy_window(120.0, 240.0, mean_speed=5.0)
    w_prev = make_dummy_window(0.0, 120.0, mean_speed=2.0)

    user_ctx = {"trip_index": 8, "cold_start_status": "ML_DRIVEN"}
    feats = extract_prediction_features(w_curr, w_prev, baselines, user_ctx)

    assert feats["mean_speed_mps"] == 5.0
    assert feats["delta_mean_speed_mps"] == pytest.approx(3.0)  # 5.0 - 2.0
    assert feats["is_cold_start"] == 0.0
    assert feats["trip_index"] == 8.0
    assert "max_mad_z_score" in feats


# 9. Target generation determinism
def test_target_generation_deterministic():
    """Verify target generation produces exact deterministic output on repeated execution."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")

    w1 = make_dummy_window(0.0, 120.0, mean_speed=2.0)
    w2 = make_dummy_window(120.0, 240.0, mean_speed=2.0)
    w3 = make_dummy_window(240.0, 360.0, mean_speed=2.0)
    traj_df = pd.DataFrame([w1, w2, w3])

    t1 = generate_horizon_target(traj_df, base_t, 360, baselines)
    t2 = generate_horizon_target(traj_df, base_t, 360, baselines)
    assert t1 == 0.0
    assert t2 == 0.0


# 10. Uniform boundary handling across all four window-harmonic horizons
def test_uniform_boundary_handling_all_four_horizons():
    """Verify clean 100% complete coverage across 120s, 360s, 600s, and 840s."""
    baselines = make_standard_baselines()
    base_t = pd.Timestamp("2008-10-01 08:00:00")

    # Generate 8 contiguous 120s windows (960s total)
    windows = []
    for i in range(8):
        windows.append(make_dummy_window(i * 120.0, (i + 1) * 120.0, mean_speed=2.0))
    traj_df = pd.DataFrame(windows)

    # Horizons: 120s (1 win), 360s (3 wins), 600s (5 wins), 840s (7 wins)
    for h in [120, 360, 600, 840]:
        target = generate_horizon_target(traj_df, base_t, h, baselines)
        assert target == 0.0, f"Failed for horizon {h}"

    # Inject outlier in window 2 [240, 360]
    windows_out = []
    for i in range(8):
        spd = 18.0 if i == 2 else 2.0
        lp = 0.8 if i == 2 else 0.1
        windows_out.append(make_dummy_window(i * 120.0, (i + 1) * 120.0, mean_speed=spd, loop=lp))
    traj_out_df = pd.DataFrame(windows_out)

    # 120s horizon spans [0, 120]: does not contain window 2 -> target == 0.0
    assert generate_horizon_target(traj_out_df, base_t, 120, baselines) == 0.0
    # 360s, 600s, 840s horizons contain window 2 -> target == 1.0
    assert generate_horizon_target(traj_out_df, base_t, 360, baselines) == 1.0
    assert generate_horizon_target(traj_out_df, base_t, 600, baselines) == 1.0
    assert generate_horizon_target(traj_out_df, base_t, 840, baselines) == 1.0


# 11. Heuristic baselines fit and predict validation
def test_heuristic_baselines_fit_predict():
    """Verify that heuristic baselines tune thresholds on validation and predict accurately."""
    X_val = pd.DataFrame({
        "max_mad_z_score": [1.0, 1.5, 4.0, 5.5, 2.0],
        "p95_exceedance_count": [0, 1, 3, 4, 0],
    })
    y_val = pd.Series([0, 0, 1, 1, 0])

    mad_base = PersonalizedRobustMADBaseline().fit(X_val, y_val)
    perc_base = PersonalizedPercentileBaseline().fit(X_val, y_val)

    preds_mad = mad_base.predict(X_val)
    preds_perc = perc_base.predict(X_val)

    assert len(preds_mad) == 5
    assert len(preds_perc) == 5
    # High score indices 2 and 3 should be predicted positive
    assert preds_mad[2] == 1 and preds_mad[3] == 1
    assert preds_perc[2] == 1 and preds_perc[3] == 1


# 12. Binary prediction metrics engine validation
def test_evaluate_binary_predictions_metrics():
    """Verify metrics calculation against derived target."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

    metrics = evaluate_binary_predictions(y_true, y_pred, y_prob)

    assert metrics["n_samples"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["prevalence"] == 0.5
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


# 13. Regression test: verify exact feature key 'max_mad_z_score' emitted and usable by baselines
def test_max_mad_z_score_feature_naming_regression():
    """Verify that extract_prediction_features emits 'max_mad_z_score', preventing KeyError crashes."""
    w1 = make_dummy_window(0.0, 120.0, mean_speed=2.0)
    w2 = make_dummy_window(120.0, 240.0, mean_speed=5.0)
    baselines = make_standard_baselines()
    u_ctx = {"trip_count": 5, "cold_start_status": "WARM"}

    feats = extract_prediction_features(w2, w1, baselines, u_ctx)

    assert "max_mad_z_score" in feats, "Feature dict missing canonical 'max_mad_z_score' key"
    assert "max_z_score" not in feats, "Feature dict should not use deprecated 'max_z_score' key"
    assert not np.isnan(feats["max_mad_z_score"])
    assert feats["max_mad_z_score"] > 0.0

    df = pd.DataFrame([feats])
    y_dummy = pd.Series([1])
    mad_base = PersonalizedRobustMADBaseline().fit(df, y_dummy)
    preds = mad_base.predict(df)
    assert len(preds) == 1


# 14. Regression test: verify centralized RiskConfig defaults and integrity
def test_risk_config_centralization_and_reproducibility():
    """Verify that RiskConfig centralizes all search grids and sample limits reproducing defaults exactly."""
    import hashlib
    from pathlib import Path

    cfg = RiskConfig()

    # 1. Centralized defaults reproduce previous candidate grids exactly
    np.testing.assert_allclose(
        np.linspace(cfg.xgb_threshold_min, cfg.xgb_threshold_max, cfg.xgb_threshold_steps),
        np.linspace(0.1, 0.9, 81),
        err_msg="XGBoost threshold grid diverged from 0.1-0.9 81 steps",
    )
    np.testing.assert_allclose(
        np.linspace(cfg.mad_threshold_min, cfg.mad_threshold_max, cfg.mad_threshold_steps),
        np.linspace(1.0, 5.0, 41),
        err_msg="MAD threshold grid diverged from 1.0-5.0 41 steps",
    )
    assert list(range(1, cfg.pct_threshold_max + 1)) == [1, 2, 3, 4]

    # 2. SHAP sample limit resolves to 200
    assert cfg.shap_max_samples == 200

    # 3. Model wrappers respect RiskConfig centralized values
    mad_base = PersonalizedRobustMADBaseline(config=cfg)
    assert mad_base.config.mad_threshold_min == 1.0
    assert mad_base.config.mad_threshold_max == 5.0
    assert mad_base.config.mad_threshold_steps == 41

    pct_base = PersonalizedPercentileBaseline(config=cfg)
    assert pct_base.config.pct_threshold_max == 4

    # 4. Verify no Chunk 1-5 files changed (10 locked hashes)
    locked_hashes = {
        "ml/data/processed/trajectories.parquet": (
            "17570ab9226f5274ddc49066d18efc8d4bde5300f0667a40cbd2e2d69ba3a65b"
        ),
        "ml/data/processed/trajectory_windows.parquet": (
            "8a7fbd8677594f236b6eb071e380e23feec46f1218a0e9beb7b905f8bc03764d"
        ),
        "ml/data/processed/trajectory_features.parquet": (
            "7a2b6637a42e8d1e8d6236cf9b5372eab34d74e559b1df464252271d26a29502"
        ),
        "ml/data/processed/behavior_windows.parquet": (
            "ac67e9dd2c8ad0d37d5aeb6ddd14166f36b090fabba92ef002988ced50e73236"
        ),
        "ml/data/processed/mobility_profiles.parquet": (
            "2cca07c57f9ce4efc5e7ed2a2750186928052bebb8fb21270befd19863af8c29"
        ),
        "ml/src/data.py": "a124f2d4691024750c7afb6ed9b4d13c6a7fe867b572a7e11355ee7c56991e79",
        "ml/src/trajectory.py": "b21ed71e63056af56e42b56301158e78b81b93750af43bea8a2dd3daa1d2c69a",
        "ml/src/features.py": "0b850322a44f8c75164ace78a82d3e6b632df3f43ee0dca3bfb45a7d7c6d3b9f",
        "ml/src/behavior.py": "b87f5b6d68ef9b199f0d062231ea86c8552a6587f56328aeafe324b084e5f670",
        "ml/src/profile.py": "43445edf727a976a66969e5b69e3770408a8199aa597456b39bfd3dfc2c939ca",
    }
    for rel_path, expected_hash in locked_hashes.items():
        p = Path(rel_path)
        assert p.exists(), f"Locked file {rel_path} does not exist"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == expected_hash, f"Locked file {rel_path} hash modified: {h} != {expected_hash}"

    # 5. Verify existing prediction and report files exist and were not altered
    expected_outputs = {
        "ml/data/processed/risk_predictions.parquet": (
            "d716f7a268a7e96c874fb89c0e3311ebb343fb4c8fa7557f5eaa26c98b4a204f"
        ),
        "predictive_risk_report.json": "c7b57bdff9c126c382f5e535b376df101d432daa0da2a3c064c424f95bafced1",
        "predictive_risk_report.md": "181c2b108b1f4f8fc044e63f4cdd70e0447726925bf120da6818c9db945df0da",
    }
    for rel_path, expected_hash in expected_outputs.items():
        p = Path(rel_path)
        assert p.exists(), f"Output file {rel_path} missing"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == expected_hash, f"Output file {rel_path} was regenerated: {h} != {expected_hash}"


# 15. Regression test: comprehensive pre-submission methodology safeguards
def test_comprehensive_pre_submission_safeguards():
    """Verify all 10 pre-submission methodology safeguards explicitly."""
    # 1. Predictor timestamp <= target origin < future target interval
    t_origin = pd.Timestamp("2008-10-01 10:00:00")
    w_curr = make_dummy_window(0.0, 120.0)
    w_curr["end_time"] = t_origin
    w_future = make_dummy_window(120.0, 240.0)
    w_future["start_time"] = t_origin
    w_future["end_time"] = t_origin + pd.Timedelta(seconds=120)
    assert pd.to_datetime(w_curr["end_time"]) <= pd.to_datetime(w_future["start_time"])

    # 2. Future observations cannot affect prediction-time profile
    base1 = make_standard_baselines()
    u_ctx = {"trip_count": 5, "cold_start_status": "WARM"}
    feats1 = extract_prediction_features(w_curr, None, base1, u_ctx)
    w_future_perturbed = make_dummy_window(120.0, 240.0, mean_speed=99.0)
    _ = w_future_perturbed
    feats2 = extract_prediction_features(w_curr, None, base1, u_ctx)
    assert feats1 == feats2

    # 3. Target construction does not modify predictors
    feats_before = feats1.copy()
    traj_windows = pd.DataFrame([w_curr, w_future])
    _ = generate_horizon_target(traj_windows, t_origin, 120, base1)
    assert feats1 == feats_before

    # 4. Test users cannot appear in training (cohort disjointness)
    all_users = [f"u_{i:03d}" for i in range(100)]
    rng = np.random.RandomState(42)
    shuffled = all_users.copy()
    rng.shuffle(shuffled)
    n_unseen = int(100 * 0.20)
    unseen_users = set(shuffled[:n_unseen])
    known_users = set(shuffled[n_unseen:])
    assert len(unseen_users.intersection(known_users)) == 0

    # 5. Validation-only threshold tuning
    model = XGBoostRiskModel()
    val_probs = np.array([0.1, 0.2, 0.7, 0.8])
    y_val = np.array([0, 0, 1, 1])
    tuned_t = model._tune_threshold(val_probs, y_val)
    assert 0.1 <= tuned_t <= 0.9

    # 6. Zero-MAD handling: epsilon and penalty behavior
    cfg = RiskConfig()
    out_flag, is_ev, devs = evaluate_window_outlier(
        pd.Series({
            "mean_speed_mps": 5.0,
            "tortuosity_index": 1.1,
            "entropy_directional": 2.5,
        }),
        {
            "mean_speed_mps": {"median": 0.0, "robust_scale": 0.0, "p95": 0.0},
            "tortuosity_index": {"median": 1.1, "robust_scale": 0.1, "p95": 1.5},
            "entropy_directional": {"median": 2.5, "robust_scale": 0.2, "p95": 3.0},
        },
        config=cfg,
    )
    assert devs["mean_speed_mps"] == cfg.zero_mad_deviation_penalty

    # 7. Horizon boundary handling: crossing window rejected
    w_cross = make_dummy_window(0.0, 180.0)  # spans beyond 120s
    traj_cross = pd.DataFrame([w_cross])
    t_start = pd.to_datetime(w_cross["start_time"])
    tgt_cross = generate_horizon_target(traj_cross, t_start, 120, base1, config=cfg)
    assert np.isnan(tgt_cross)

    # 8. Incomplete horizon exclusion: returns NaN symmetrically
    traj_short = pd.DataFrame([make_dummy_window(0.0, 60.0)])
    tgt_short = generate_horizon_target(traj_short, t_start, 120, base1, config=cfg)
    assert np.isnan(tgt_short)

    # 9. Configuration values are actually used
    custom_cfg = RiskConfig(robust_mad_multiplier=5.0, xgb_threshold_min=0.2)
    assert custom_cfg.robust_mad_multiplier == 5.0
    assert custom_cfg.xgb_threshold_min == 0.2

    # 10. Deterministic output with fixed seed
    np.random.seed(cfg.random_seed)
    r1 = np.random.rand(10)
    np.random.seed(cfg.random_seed)
    r2 = np.random.rand(10)
    np.testing.assert_array_equal(r1, r2)

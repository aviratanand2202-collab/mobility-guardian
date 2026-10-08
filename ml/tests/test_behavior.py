"""Unit tests for trajectory behavioral pattern analysis and classification (Chunk 4).

Validates all required behavioral conditions:
1. Straight movement -> NORMAL
2. Repeated A->B->A->B pacing -> PACING
3. Closed lapping route -> LAPPING
4. Irregular random drift -> RANDOM_DRIFT
5. Zig-zag without spatial return -> NORMAL (negative control for PACING)
6. Stationary / dwell -> INSUFFICIENT_EVIDENCE
7. Insufficient observations -> INSUFFICIENT_EVIDENCE
8. Zero-dt transitions handling
9. Deterministic repeated execution
10. Controlled synthetic benchmark generation and evaluation
"""

import numpy as np
import pandas as pd
import pytest

from ml.src.behavior import (
    BehaviorConfig,
    SyntheticGeneratorConfig,
    classify_behavior_dataframe,
    evaluate_random_drift_regimes,
    evaluate_synthetic_benchmark,
    generate_random_drift_regimes,
    generate_synthetic_benchmark,
)
from ml.src.features import FeatureConfig, extract_features_from_dataframe


def create_window_points(
    lats: list[float] | np.ndarray,
    lons: list[float] | np.ndarray,
    dts: list[float] | np.ndarray | None = None,
    window_id: str = "w_test_001",
    user_id: str = "u001",
    traj_id: str = "t001",
    seg_id: str = "s001",
    window_index: int = 0,
    is_full_window: bool = True,
) -> pd.DataFrame:
    """Helper to construct a window points DataFrame for feature extraction."""
    n_pts = len(lats)
    if dts is None:
        dt_val = 120.0 / n_pts if n_pts > 0 else 0.0
        dts = [0.0] + [dt_val] * (n_pts - 1)

    t0 = pd.Timestamp("2026-10-05 10:00:00")
    timestamps = [t0]
    for d in dts[1:]:
        timestamps.append(timestamps[-1] + pd.Timedelta(seconds=float(d)))

    return pd.DataFrame(
        {
            "window_id": [window_id] * n_pts,
            "user_id": [user_id] * n_pts,
            "trajectory_id": [traj_id] * n_pts,
            "segment_id": [seg_id] * n_pts,
            "window_index": [window_index] * n_pts,
            "is_full_window": [is_full_window] * n_pts,
            "timestamp": timestamps,
            "latitude": list(lats),
            "longitude": list(lons),
            "dt": list(dts),
        }
    )


def extract_and_classify(points_df: pd.DataFrame, config: BehaviorConfig = BehaviorConfig()) -> pd.DataFrame:
    """Convenience helper to extract features and classify behavior."""
    feat_df = extract_features_from_dataframe(points_df, config=FeatureConfig())
    return classify_behavior_dataframe(feat_df, config=config)


# =============================================================================
# 1. STRAIGHT MOVEMENT -> NORMAL
# =============================================================================


def test_straight_movement():
    """Verify directed straight-line movement is classified as NORMAL."""
    n_pts = 60
    # ~300m linear progression northbound
    lats = [39.90 + i * (300.0 / 60) / 111000.0 for i in range(n_pts)]
    lons = [116.40 for _ in range(n_pts)]

    pts_df = create_window_points(lats, lons, window_id="w_straight")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    assert res.iloc[0]["behavior_class"] == "NORMAL"
    assert res.iloc[0]["path_closure_ratio"] < 0.20
    assert res.iloc[0]["straight_line_displacement_m"] > 250.0


# =============================================================================
# 2. REPEATED A->B->A->B PACING -> PACING
# =============================================================================


def test_repeated_abab_pacing():
    """Verify repeated linear back-and-forth movement is classified as PACING."""
    # 4 legs: A -> B -> A -> B along an 80m corridor
    # Returns near origin, has backtracking reversals, low directional entropy
    corridor_len_m = 80.0
    pts_per_leg = 20
    legs = [
        np.linspace(0, corridor_len_m, pts_per_leg),
        np.linspace(corridor_len_m, 0, pts_per_leg),
        np.linspace(0, corridor_len_m, pts_per_leg),
        np.linspace(corridor_len_m, 0, pts_per_leg),
    ]
    pos = np.concatenate(legs)
    lats = 39.90 + (pos / 111000.0)
    lons = 116.40 + np.zeros_like(lats)

    pts_df = create_window_points(lats, lons, window_id="w_pacing")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    row = res.iloc[0]
    assert row["behavior_class"] == "PACING"
    assert row["path_closure_ratio"] >= 0.60
    assert row["backtracking_tendency"] > 0.01
    assert row["entropy_directional"] <= 0.70


# =============================================================================
# 3. CLOSED LAPPING ROUTE -> LAPPING
# =============================================================================


def test_closed_lapping_route():
    """Verify closed circuit / loop route traversal is classified as LAPPING."""
    n_pts = 60
    radius_m = 60.0
    # 1 full circle
    angles = np.linspace(0, 2 * np.pi, n_pts)
    lats = 39.90 + (radius_m * np.cos(angles)) / 111000.0
    lons = 116.40 + (radius_m * np.sin(angles)) / (111000.0 * np.cos(np.radians(39.90)))

    pts_df = create_window_points(lats, lons, window_id="w_lapping")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    row = res.iloc[0]
    assert row["behavior_class"] == "LAPPING"
    assert row["path_closure_ratio"] >= 0.60
    assert row["loop_metric"] >= 0.45
    assert row["backtracking_tendency"] <= 0.05


# =============================================================================
# 4. RANDOM DRIFT -> RANDOM_DRIFT
# =============================================================================


def test_random_drift():
    """Verify high-entropy erratic 2D Brownian motion is classified as RANDOM_DRIFT."""
    # Seeded Brownian walk with frequent direction changes
    rng = np.random.RandomState(123)
    n_pts = 80
    step_sigma = 0.00008
    d_lats = rng.normal(0, step_sigma, n_pts)
    d_lons = rng.normal(0, step_sigma, n_pts)
    lats = 39.90 + np.cumsum(d_lats)
    lons = 116.40 + np.cumsum(d_lons)

    pts_df = create_window_points(lats, lons, window_id="w_random")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    row = res.iloc[0]
    assert row["behavior_class"] == "RANDOM_DRIFT"
    assert row["entropy_directional"] >= 0.70
    assert row["heading_variability"] >= 0.50
    assert row["turn_frequency"] >= 5.0


# =============================================================================
# 5. ZIG-ZAG WITHOUT SPATIAL RETURN -> NORMAL (NEGATIVE CONTROL)
# =============================================================================


def test_zigzag_without_spatial_return():
    """Verify forward progression with zig-zagging is NORMAL, NOT PACING."""
    n_pts = 60
    # Forward progress 250m with sinusoidal weaving (+- 15m)
    mainline = np.linspace(0, 250.0, n_pts)
    crossline = 15.0 * np.sin(np.linspace(0, 6 * np.pi, n_pts))
    lats = 39.90 + mainline / 111000.0
    lons = 116.40 + crossline / (111000.0 * np.cos(np.radians(39.90)))

    pts_df = create_window_points(lats, lons, window_id="w_zigzag")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    row = res.iloc[0]
    # Must NOT be classified as PACING because spatial displacement is large and closure is small
    assert row["behavior_class"] == "NORMAL"
    assert row["behavior_class"] != "PACING"
    assert row["path_closure_ratio"] < 0.50


# =============================================================================
# 6. STATIONARY DWELL -> INSUFFICIENT_EVIDENCE
# =============================================================================


def test_stationary_dwell():
    """Verify stationary window with minor GPS noise jitter is INSUFFICIENT_EVIDENCE."""
    rng = np.random.RandomState(42)
    n_pts = 40
    # Sub-meter GPS noise jitter around a fixed point
    lats = 39.90 + rng.normal(0, 0.000003, n_pts)
    lons = 116.40 + rng.normal(0, 0.000003, n_pts)

    pts_df = create_window_points(lats, lons, window_id="w_dwell")
    res = extract_and_classify(pts_df)

    assert len(res) == 1
    row = res.iloc[0]
    assert row["behavior_class"] == "INSUFFICIENT_EVIDENCE"
    assert row["path_distance_m"] < 10.0 or row["bbox_diagonal_m"] < 10.0


# =============================================================================
# 7. INSUFFICIENT OBSERVATIONS -> INSUFFICIENT_EVIDENCE
# =============================================================================


def test_insufficient_observations():
    """Verify windows with 1 point or tiny temporal span are INSUFFICIENT_EVIDENCE."""
    # 1 point window
    df1 = create_window_points([39.90], [116.40], dts=[0.0], window_id="w_1pt")
    res1 = extract_and_classify(df1)
    assert res1.iloc[0]["behavior_class"] == "INSUFFICIENT_EVIDENCE"

    # 2 points with short temporal span (10 seconds)
    df2 = create_window_points([39.90, 39.9001], [116.40, 116.4001], dts=[0.0, 10.0], window_id="w_2pt")
    res2 = extract_and_classify(df2)
    assert res2.iloc[0]["behavior_class"] == "INSUFFICIENT_EVIDENCE"


# =============================================================================
# 8. ZERO-DT TRANSITIONS HANDLING
# =============================================================================


def test_zero_dt_transitions():
    """Verify windows containing duplicate timestamps / dt=0.0 are handled safely."""
    # 10 points with some dt=0.0 transitions
    lats = [39.90 + i * 0.0001 for i in range(10)]
    lons = [116.40 + i * 0.0001 for i in range(10)]
    dts = [0.0, 0.0, 5.0, 5.0, 0.0, 10.0, 10.0, 0.0, 15.0, 15.0]

    pts_df = create_window_points(lats, lons, dts=dts, window_id="w_zero_dt")
    feat_df = extract_features_from_dataframe(pts_df, config=FeatureConfig())
    res = classify_behavior_dataframe(feat_df)

    assert len(res) == 1
    row = res.iloc[0]
    assert not pd.isna(row["behavior_class"])
    assert feat_df.iloc[0]["zero_dt_count"] == 3


# =============================================================================
# 9. DETERMINISTIC REPEATED EXECUTION
# =============================================================================


def test_deterministic_repeated_execution():
    """Verify identical inputs yield bit-identical behavioral classifications."""
    rng = np.random.RandomState(999)
    n_pts = 50
    lats = 39.90 + np.cumsum(rng.normal(0, 0.00005, n_pts))
    lons = 116.40 + np.cumsum(rng.normal(0, 0.00005, n_pts))

    pts_df = create_window_points(lats, lons, window_id="w_det")
    feat_df = extract_features_from_dataframe(pts_df, config=FeatureConfig())

    res1 = classify_behavior_dataframe(feat_df)
    res2 = classify_behavior_dataframe(feat_df)

    pd.testing.assert_frame_equal(res1, res2)


# =============================================================================
# 10. SYNTHETIC BENCHMARK EVALUATION
# =============================================================================


def test_synthetic_benchmark_evaluation():
    """Verify the controlled synthetic benchmark generator and evaluation metrics."""
    pts_df, feat_df = generate_synthetic_benchmark(n_per_class=10, seed=42)
    assert len(pts_df) > 0
    assert len(feat_df) == 50  # 5 classes * 10

    eval_res = evaluate_synthetic_benchmark(feat_df, config=BehaviorConfig())
    assert eval_res["total_synthetic_windows"] == 50
    assert "classification_report" in eval_res
    assert "confusion_matrix" in eval_res
    assert "sample_mismatches" in eval_res

    # Check that PACING precision is high (> 0.80) and INSUFFICIENT recall is 1.00
    cr = eval_res["classification_report"]
    assert cr["PACING"]["precision"] >= 0.80
    assert cr["INSUFFICIENT_EVIDENCE"]["recall"] == 1.00


# =============================================================================
# 11. CONFIG INTEGRITY AND JUSTIFICATION
# =============================================================================


def test_behavior_config_justification():
    """Verify BehaviorConfig parameters are centralized and within documented ranges."""
    cfg = BehaviorConfig()
    assert cfg.min_eval_points >= 3
    assert cfg.min_eval_span_sec >= 10.0
    assert cfg.min_movement_path_m == 10.0
    assert 0.0 < cfg.min_pacing_closure <= 1.0
    assert cfg.min_pacing_expansion > 1.0
    assert 0.0 <= cfg.min_pacing_backtracking <= 1.0
    assert 0.0 < cfg.min_lapping_loop_metric <= 1.0
    assert 0.0 < cfg.min_random_entropy <= 1.0
    assert cfg.max_random_loop_metric == 1.0

    gen_cfg = SyntheticGeneratorConfig()
    assert gen_cfg.diffusive_step_sigma > 0
    assert gen_cfg.window_duration_sec == 120.0


# =============================================================================
# 12. EXPLICIT SYNTHETIC CLASS MAPPING (PART A AUDIT)
# =============================================================================


def test_explicit_synthetic_class_mapping():
    """Verify explicit mapping of every synthetic archetype to its expected class label."""
    pts_df, feat_df = generate_synthetic_benchmark(n_per_class=10, seed=42)

    # Check generator window_id prefix to true_class mapping
    w_ids = feat_df["window_id"].tolist()
    classes = feat_df["true_class"].tolist()

    for wid, cls in zip(w_ids, classes):
        if wid.startswith("syn_norm_str_"):
            assert cls == "NORMAL"
        elif wid.startswith("syn_norm_zz_"):
            assert cls == "NORMAL"
        elif wid.startswith("syn_pace_"):
            assert cls == "PACING"
        elif wid.startswith("syn_lap_"):
            assert cls == "LAPPING"
        elif wid.startswith("syn_rand_"):
            assert cls == "RANDOM_DRIFT"
        elif wid.startswith("syn_stat_"):
            assert cls == "INSUFFICIENT_EVIDENCE"
        else:
            pytest.fail(f"Unexpected generator prefix in window_id: {wid}")


# =============================================================================
# 13. MULTI-REGIME RANDOM DRIFT VALIDATION (PART C AUDIT)
# =============================================================================


def test_random_drift_multi_regimes():
    """Verify behavioral descriptors respond in intended directions across 4 random drift regimes."""
    pts_df, feat_df = generate_random_drift_regimes(n_per_regime=15, seed=42)
    assert len(feat_df) == 60  # 4 regimes * 15

    regimes_eval = evaluate_random_drift_regimes(feat_df, config=BehaviorConfig())

    # 1. Strongly diffusive pure 2D Brownian motion
    sd = regimes_eval["strongly_diffusive"]
    assert sd["recall"] >= 0.80
    assert sd["median_entropy"] >= 0.85
    assert sd["median_heading_var"] >= 0.70

    # 2. High-turn frequency random walk
    ht = regimes_eval["high_turn"]
    assert ht["recall"] >= 0.90
    assert ht["median_turn_freq"] >= 10.0

    # 3. Spatially dispersed drift
    sp = regimes_eval["spatially_dispersed"]
    assert sp["recall"] >= 0.90

    # 4. Weakly persistent correlated walk: persistent forward momentum reduces random drift recall
    wp = regimes_eval["weakly_persistent"]
    assert wp["recall"] < sd["recall"]
    assert wp["median_displacement_m"] > sd["median_displacement_m"]


# =============================================================================
# 14. NEGATIVE CONTROLS RIGOROUS AUDIT (PART F AUDIT)
# =============================================================================


def test_negative_controls_rigorous():
    """Verify strict protection against false positive misclassifications."""
    pts_df, feat_df = generate_synthetic_benchmark(n_per_class=20, seed=42)
    classified = classify_behavior_dataframe(feat_df, config=BehaviorConfig())
    merged = feat_df.merge(classified[["window_id", "behavior_class"]], on="window_id")

    # 1. Straight transit: 0 false positives for PACING, LAPPING, or RANDOM_DRIFT
    str_windows = merged[merged["window_id"].str.startswith("syn_norm_str_")]
    assert (str_windows["behavior_class"] == "NORMAL").all()

    # 2. Forward zig-zag without return: 0 false positives for PACING
    zz_windows = merged[merged["window_id"].str.startswith("syn_norm_zz_")]
    assert not (zz_windows["behavior_class"] == "PACING").any()

    # 3. Stationary dwell: 100% INSUFFICIENT_EVIDENCE
    dwell_windows = merged[merged["window_id"].str.startswith("syn_stat_")]
    assert (dwell_windows["behavior_class"] == "INSUFFICIENT_EVIDENCE").all()

    # 4. Closed lapping routes: 0 false positives for PACING
    lap_windows = merged[merged["window_id"].str.startswith("syn_lap_")]
    assert not (lap_windows["behavior_class"] == "PACING").any()


# =============================================================================
# 15. DATA QUALITY & TEMPORAL INTEGRITY (PART G AUDIT)
# =============================================================================


def test_data_quality_and_temporal_integrity():
    """Verify extreme kinematic flags and corrupt observations produce INSUFFICIENT_EVIDENCE."""
    pts_df, feat_df = generate_synthetic_benchmark(n_per_class=5, seed=42)

    # Inject an extreme kinematic transition flag
    corrupt_feat = feat_df.copy()
    corrupt_feat.loc[0, "has_extreme_kinematic_transition"] = True
    corrupt_feat.loc[1, "is_kinematically_evaluable"] = False
    corrupt_feat.loc[2, "path_closure_ratio"] = np.nan

    classified = classify_behavior_dataframe(corrupt_feat, config=BehaviorConfig())

    assert classified.loc[0, "behavior_class"] == "INSUFFICIENT_EVIDENCE"
    assert classified.loc[1, "behavior_class"] == "INSUFFICIENT_EVIDENCE"
    assert classified.loc[2, "behavior_class"] == "INSUFFICIENT_EVIDENCE"

"""Unit tests for trajectory window feature engineering (Chunk 3).

Verifies all required test conditions:
1. 1-point window
2. 2-point window
3. 3+ point window
4. zero-dt observations
5. zero displacement
6. stationary observations
7. normal movement
8. longitude/latitude distance calculation (Haversine geodesic)
9. no infinity values in output
10. deterministic repeated execution
11. row/window count matching
12. no future-window information used
13. Anti-meridian spherical geometry (179.9 to -179.9 and -179.9 to 179.9)
14. Distinguishing true back-and-forth pacing from forward zig-zagging
15. Distinguishing 2D polygon looping from 1D back-and-forth pacing
"""

import math
import numpy as np
import pandas as pd
import pytest

from ml.src.features import (
    circular_longitude_span,
    extract_features_from_dataframe,
    haversine_np,
    spherical_centroid,
)


def create_synthetic_window_df(
    n_points: int,
    start_time: str = "2026-10-05 10:00:00",
    dt_seconds: float = 5.0,
    lat_start: float = 39.9042,
    lon_start: float = 116.4074,
    lat_step: float = 0.0001,
    lon_step: float = 0.0001,
    window_id: str = "test_w00001",
    user_id: str = "001",
    traj_id: str = "001_traj1",
    seg_id: str = "001_traj1_s001",
    is_full: bool = True,
) -> pd.DataFrame:
    """Helper to create a deterministic synthetic window DataFrame."""
    t0 = pd.Timestamp(start_time)
    ts = [t0 + pd.Timedelta(seconds=i * dt_seconds) for i in range(n_points)]
    lats = [lat_start + i * lat_step for i in range(n_points)]
    lons = [lon_start + i * lon_step for i in range(n_points)]
    dts = [0.0] + [dt_seconds] * (n_points - 1) if n_points > 1 else [0.0]

    return pd.DataFrame(
        {
            "window_id": [window_id] * n_points,
            "user_id": [user_id] * n_points,
            "trajectory_id": [traj_id] * n_points,
            "segment_id": [seg_id] * n_points,
            "window_index": [0] * n_points,
            "is_full_window": [is_full] * n_points,
            "timestamp": pd.to_datetime(ts),
            "latitude": np.array(lats, dtype=np.float64),
            "longitude": np.array(lons, dtype=np.float64),
            "dt": np.array(dts, dtype=np.float64),
        }
    )


# Test 1: 1-point window
def test_single_point_window():
    df = create_synthetic_window_df(n_points=1)
    feat_df = extract_features_from_dataframe(df)

    assert len(feat_df) == 1
    row = feat_df.iloc[0]
    assert row["point_count"] == 1
    assert row["temporal_span_sec"] == 0.0
    assert row["path_distance_m"] == 0.0
    assert row["straight_line_displacement_m"] == 0.0
    assert math.isnan(row["mean_speed_mps"])
    assert math.isnan(row["tortuosity_index"])
    assert math.isnan(row["entropy_directional"])
    assert math.isnan(row["heading_change_mean"])
    assert bool(row["has_displacement"]) is False
    assert bool(row["has_valid_kinematics"]) is False
    assert bool(row["is_kinematically_evaluable"]) is False


# Test 2: 2-point window
def test_two_point_window():
    df = create_synthetic_window_df(n_points=2, dt_seconds=10.0, lat_step=0.001, lon_step=0.0)
    feat_df = extract_features_from_dataframe(df)

    assert len(feat_df) == 1
    row = feat_df.iloc[0]
    assert row["point_count"] == 2
    assert row["temporal_span_sec"] == 10.0
    assert row["path_distance_m"] > 100.0  # ~111m for 0.001 deg lat
    assert row["straight_line_displacement_m"] > 100.0
    assert not math.isnan(row["mean_speed_mps"])
    assert row["mean_speed_mps"] > 0.0
    assert pytest.approx(row["tortuosity_index"], 0.01) == 1.0
    assert math.isnan(row["mean_acceleration_mps2"])
    assert math.isnan(row["heading_change_mean"])
    assert math.isnan(row["entropy_directional"])
    assert bool(row["has_displacement"]) is True
    assert bool(row["has_valid_kinematics"]) is True
    assert bool(row["has_acceleration"]) is False


# Test 3: 3+ point window
def test_three_plus_point_window():
    df = create_synthetic_window_df(n_points=5, dt_seconds=10.0, lat_step=0.0005, lon_step=0.0005)
    feat_df = extract_features_from_dataframe(df)

    assert len(feat_df) == 1
    row = feat_df.iloc[0]
    assert row["point_count"] == 5
    assert row["temporal_span_sec"] == 40.0
    assert row["path_distance_m"] > 0.0
    assert not math.isnan(row["mean_speed_mps"])
    assert not math.isnan(row["mean_acceleration_mps2"])
    assert not math.isnan(row["heading_change_mean"])
    assert bool(row["has_acceleration"]) is True
    assert bool(row["is_kinematically_evaluable"]) is True


# Test 4: zero-dt observations (same-second records)
def test_zero_dt_observations_no_division_by_zero():
    df = create_synthetic_window_df(n_points=3, dt_seconds=0.0, lat_step=0.0001, lon_step=0.0001)
    df["dt"] = [0.0, 0.0, 0.0]
    feat_df = extract_features_from_dataframe(df)

    assert len(feat_df) == 1
    row = feat_df.iloc[0]
    assert row["zero_dt_count"] == 2
    assert row["zero_dt_fraction"] == 1.0
    assert row["temporal_span_sec"] == 0.0
    # Speeds must be NaN because dt = 0 must never be divided by
    assert math.isnan(row["mean_speed_mps"])
    assert math.isnan(row["speed_variance"])
    assert bool(row["has_valid_kinematics"]) is False


# Test 5: zero displacement (start and end at the exact same location)
def test_zero_displacement_handling():
    lats = [39.9042, 39.9052, 39.9052, 39.9042]
    lons = [116.4074, 116.4074, 116.4084, 116.4074]
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    ts = [t0 + pd.Timedelta(seconds=i * 10) for i in range(4)]
    df = pd.DataFrame(
        {
            "window_id": ["loop_w1"] * 4,
            "user_id": ["001"] * 4,
            "trajectory_id": ["t1"] * 4,
            "segment_id": ["s1"] * 4,
            "window_index": [0] * 4,
            "is_full_window": [True] * 4,
            "timestamp": ts,
            "latitude": lats,
            "longitude": lons,
            "dt": [0.0, 10.0, 10.0, 10.0],
        }
    )

    feat_df = extract_features_from_dataframe(df)
    row = feat_df.iloc[0]
    assert row["path_distance_m"] > 100.0
    assert row["straight_line_displacement_m"] < 0.1
    assert math.isnan(row["tortuosity_index"])
    assert pytest.approx(row["path_closure_ratio"], 0.01) == 1.0


# Test 6: stationary observations
def test_stationary_observations():
    df = create_synthetic_window_df(n_points=5, dt_seconds=5.0, lat_step=0.0, lon_step=0.0)
    feat_df = extract_features_from_dataframe(df)

    row = feat_df.iloc[0]
    assert row["path_distance_m"] == 0.0
    assert row["straight_line_displacement_m"] == 0.0
    assert row["radius_of_gyration"] == 0.0
    assert row["bbox_diagonal_m"] == 0.0
    assert row["mean_speed_mps"] == 0.0
    assert math.isnan(row["tortuosity_index"])
    assert math.isnan(row["heading_change_mean"])
    assert math.isnan(row["entropy_directional"])


# Test 7: normal movement along straight line
def test_normal_straight_movement():
    df = create_synthetic_window_df(n_points=6, dt_seconds=5.0, lat_step=0.0001, lon_step=0.0)
    feat_df = extract_features_from_dataframe(df)

    row = feat_df.iloc[0]
    assert pytest.approx(row["mean_speed_mps"], 0.1) == 2.22
    assert pytest.approx(row["tortuosity_index"], 0.01) == 1.0
    assert pytest.approx(row["heading_variability"], 0.01) == 0.0
    assert pytest.approx(row["entropy_directional"], 0.01) == 0.0


# Test 8: Haversine distance accuracy
def test_haversine_distance_accuracy():
    dist = haversine_np(0.0, 0.0, 0.0, 1.0)
    assert 111000.0 < dist < 112000.0

    dist_lat = haversine_np(0.0, 0.0, 1.0, 0.0)
    assert 110000.0 < dist_lat < 112000.0

    assert haversine_np(39.9, 116.4, 39.9, 116.4) == 0.0


# Test 9: No infinity values in any output
def test_no_infinity_values():
    dfs = [
        create_synthetic_window_df(n_points=1),
        create_synthetic_window_df(n_points=2, dt_seconds=0.0),
        create_synthetic_window_df(n_points=4, lat_step=0.0, lon_step=0.0),
        create_synthetic_window_df(n_points=5, dt_seconds=10.0),
    ]
    combined = pd.concat(dfs, ignore_index=True)
    feat_df = extract_features_from_dataframe(combined)

    num_cols = feat_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        assert not np.any(np.isinf(feat_df[col])), f"Found infinity in column {col}"


# Test 10: Deterministic repeated execution
def test_deterministic_execution():
    df = create_synthetic_window_df(n_points=10, dt_seconds=2.0, lat_step=0.0001, lon_step=0.0002)
    res1 = extract_features_from_dataframe(df)
    res2 = extract_features_from_dataframe(df)

    pd.testing.assert_frame_equal(res1, res2)


# Test 11: Window count matches input
def test_window_count_matching():
    df1 = create_synthetic_window_df(n_points=4, window_id="w1")
    df2 = create_synthetic_window_df(n_points=2, window_id="w2")
    df3 = create_synthetic_window_df(n_points=1, window_id="w3")
    combined = pd.concat([df1, df2, df3], ignore_index=True)

    feat_df = extract_features_from_dataframe(combined)
    assert len(feat_df) == 3
    assert list(feat_df["window_id"]) == ["w1", "w2", "w3"]


# Test 12: No future-window information used
def test_no_future_window_information_used():
    df1 = create_synthetic_window_df(n_points=3, window_id="w1", lat_step=0.0, lon_step=0.0)
    df2 = create_synthetic_window_df(
        n_points=3,
        window_id="w2",
        start_time="2026-10-05 10:05:00",
        lat_step=0.01,
        lon_step=0.01,
    )

    combined = pd.concat([df1, df2], ignore_index=True)
    feat_df = extract_features_from_dataframe(combined)
    solo_feat = extract_features_from_dataframe(df1)

    row_combined = feat_df[feat_df["window_id"] == "w1"].iloc[0]
    row_solo = solo_feat.iloc[0]

    for col in solo_feat.columns:
        val_comb = row_combined[col]
        val_solo = row_solo[col]
        if isinstance(val_comb, float) and math.isnan(val_comb):
            assert math.isnan(val_solo)
        else:
            assert val_comb == val_solo


# Test 13: Anti-meridian spherical geometry (+-180 deg crossing)
def test_anti_meridian_global_geometry():
    # 179.9° -> -179.9°
    span_fwd = circular_longitude_span(np.array([179.9, -179.9]))
    assert pytest.approx(span_fwd, 0.001) == 0.2

    # -179.9° -> 179.9°
    span_rev = circular_longitude_span(np.array([-179.9, 179.9]))
    assert pytest.approx(span_rev, 0.001) == 0.2

    # Spherical centroid across Anti-Meridian
    clat, clon = spherical_centroid(np.array([50.0, 50.0]), np.array([179.9, -179.9]))
    assert pytest.approx(clat, 0.01) == 50.0
    assert pytest.approx(abs(clon), 0.01) == 180.0

    # Ensure Radius of Gyration is local (~7 km), NOT across the globe (>8,000 km)
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    df_am = pd.DataFrame(
        {
            "window_id": ["am_w1", "am_w1"],
            "user_id": ["001", "001"],
            "trajectory_id": ["t1", "t1"],
            "segment_id": ["s1", "s1"],
            "window_index": [0, 0],
            "is_full_window": [True, True],
            "timestamp": [t0, t0 + pd.Timedelta(seconds=60)],
            "latitude": [50.0, 50.0],
            "longitude": [179.9, -179.9],
            "dt": [0.0, 60.0],
        }
    )
    feat_am = extract_features_from_dataframe(df_am).iloc[0]
    assert feat_am["radius_of_gyration"] < 15000.0  # ~7.1 km, NOT 8,800,000 m
    assert feat_am["bbox_width_m"] < 25000.0  # ~14.3 km, NOT 40,000,000 m


# Test 14: Distinguish true back-and-forth pacing from generic zig-zagging
def test_pacing_vs_zigzag_discrimination():
    # Pacing: A -> B -> A -> B -> A (back-and-forth along same line)
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    pacing_df = pd.DataFrame(
        {
            "window_id": ["pace"] * 5,
            "user_id": ["001"] * 5,
            "trajectory_id": ["t1"] * 5,
            "segment_id": ["s1"] * 5,
            "window_index": [0] * 5,
            "is_full_window": [True] * 5,
            "timestamp": [t0 + pd.Timedelta(seconds=i * 10) for i in range(5)],
            "latitude": [39.900, 39.910, 39.900, 39.910, 39.900],
            "longitude": [116.400] * 5,
            "dt": [0.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    # Zig-zag: advancing forward while weaving slightly left and right (no reversals)
    zigzag_df = pd.DataFrame(
        {
            "window_id": ["zigzag"] * 5,
            "user_id": ["001"] * 5,
            "trajectory_id": ["t1"] * 5,
            "segment_id": ["s1"] * 5,
            "window_index": [0] * 5,
            "is_full_window": [True] * 5,
            "timestamp": [t0 + pd.Timedelta(seconds=i * 10) for i in range(5)],
            "latitude": [39.900, 39.905, 39.910, 39.915, 39.920],
            "longitude": [116.400, 116.403, 116.400, 116.403, 116.400],
            "dt": [0.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    f_pacing = extract_features_from_dataframe(pacing_df).iloc[0]
    f_zigzag = extract_features_from_dataframe(zigzag_df).iloc[0]

    # Pacing has 100% reversal rate and closure = 1.0 -> pacing_tendency = 1.0
    assert pytest.approx(f_pacing["pacing_tendency"], 0.01) == 1.0
    # Zig-zag has 0 reversals -> pacing_tendency must be 0.0
    assert pytest.approx(f_zigzag["pacing_tendency"], 0.01) == 0.0


# Test 15: Distinguish 2D polygon looping from 1D back-and-forth pacing
def test_loop_vs_pacing_discrimination():
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    # Circular / Square Loop: A -> B -> C -> D -> A
    loop_df = pd.DataFrame(
        {
            "window_id": ["loop"] * 5,
            "user_id": ["001"] * 5,
            "trajectory_id": ["t1"] * 5,
            "segment_id": ["s1"] * 5,
            "window_index": [0] * 5,
            "is_full_window": [True] * 5,
            "timestamp": [t0 + pd.Timedelta(seconds=i * 10) for i in range(5)],
            "latitude": [39.90, 39.91, 39.91, 39.90, 39.90],
            "longitude": [116.40, 116.40, 116.41, 116.41, 116.40],
            "dt": [0.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    # 1D Linear Pacing: A -> B -> A -> B -> A
    pacing_df = pd.DataFrame(
        {
            "window_id": ["pace"] * 5,
            "user_id": ["001"] * 5,
            "trajectory_id": ["t1"] * 5,
            "segment_id": ["s1"] * 5,
            "window_index": [0] * 5,
            "is_full_window": [True] * 5,
            "timestamp": [t0 + pd.Timedelta(seconds=i * 10) for i in range(5)],
            "latitude": [39.900, 39.910, 39.900, 39.910, 39.900],
            "longitude": [116.400] * 5,
            "dt": [0.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    f_loop = extract_features_from_dataframe(loop_df).iloc[0]
    f_pacing = extract_features_from_dataframe(pacing_df).iloc[0]

    # Both have high path closure (return near start)
    assert pytest.approx(f_loop["path_closure_ratio"], 0.01) == 1.0
    assert pytest.approx(f_pacing["path_closure_ratio"], 0.01) == 1.0

    # True 2D loop metric: High for polygon loop, 0.0 for 1D pacing
    assert f_loop["loop_metric"] > 0.5
    assert pytest.approx(f_pacing["loop_metric"], 0.01) == 0.0


# Test 16: Extreme kinematic transition quality flags
def test_extreme_kinematic_transition_quality_flag():
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    # Normal window: pedestrian speed ~1.1 m/s (< 340 m/s)
    normal_df = pd.DataFrame(
        {
            "window_id": ["norm"] * 3,
            "user_id": ["001"] * 3,
            "trajectory_id": ["t1"] * 3,
            "segment_id": ["s1"] * 3,
            "window_index": [0] * 3,
            "is_full_window": [True] * 3,
            "timestamp": [t0, t0 + pd.Timedelta(seconds=10), t0 + pd.Timedelta(seconds=20)],
            "latitude": [39.9000, 39.9001, 39.9002],
            "longitude": [116.4000, 116.4000, 116.4000],
            "dt": [0.0, 10.0, 10.0],
        }
    )
    f_norm = extract_features_from_dataframe(normal_df).iloc[0]
    assert not f_norm["has_extreme_kinematic_transition"]
    assert f_norm["extreme_transition_count"] == 0
    assert f_norm["extreme_transition_fraction"] == 0.0

    # Window with non-physical jump: 500 km in 2s (speed = 250,000 m/s > 340 m/s)
    extreme_df = pd.DataFrame(
        {
            "window_id": ["ext"] * 3,
            "user_id": ["001"] * 3,
            "trajectory_id": ["t1"] * 3,
            "segment_id": ["s1"] * 3,
            "window_index": [0] * 3,
            "is_full_window": [True] * 3,
            "timestamp": [t0, t0 + pd.Timedelta(seconds=2), t0 + pd.Timedelta(seconds=4)],
            "latitude": [34.0000, 39.0000, 39.0001],
            "longitude": [108.0000, 116.0000, 116.0000],
            "dt": [0.0, 2.0, 2.0],
        }
    )
    f_ext = extract_features_from_dataframe(extreme_df).iloc[0]
    assert f_ext["has_extreme_kinematic_transition"]
    assert f_ext["extreme_transition_count"] == 1
    assert pytest.approx(f_ext["extreme_transition_fraction"], 0.01) == 0.5


# Test 17: Loop-likeness continuous aspect ratio scaling
def test_loop_metric_aspect_ratio_scaling():
    t0 = pd.Timestamp("2026-10-05 10:00:00")
    # Thin rectangle loop: 1000m high, 100m wide (aspect ratio ~ 0.1)
    thin_loop_df = pd.DataFrame(
        {
            "window_id": ["thin"] * 5,
            "user_id": ["001"] * 5,
            "trajectory_id": ["t1"] * 5,
            "segment_id": ["s1"] * 5,
            "window_index": [0] * 5,
            "is_full_window": [True] * 5,
            "timestamp": [t0 + pd.Timedelta(seconds=i * 10) for i in range(5)],
            "latitude": [39.90, 39.909, 39.909, 39.90, 39.90],
            "longitude": [116.400, 116.400, 116.401, 116.401, 116.400],
            "dt": [0.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    f_thin = extract_features_from_dataframe(thin_loop_df).iloc[0]
    # Continuous aspect ratio scales loop_metric smoothly (~0.08 - 0.12)
    assert 0.05 < f_thin["loop_metric"] < 0.20

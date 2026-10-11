"""Unit and integration tests for Chunk 5: Personalized Mobility Profile & Baseline Modeling.

Covers all 25 required test cases:
1. single-user profile
2. multi-user isolation
3. DBSCAN anchors
4. anchor radius (robust derivation, not fixed, resistant to outlier)
5. zero-anchor case
6. insufficient observations
7. return-trip detection
8. no-return-trip case
9. CV calculation
10. zero-mean CV
11. Gamma fitting
12. invalid Gamma input
13. empirical baseline >=30 samples
14. P95/P99
15. MAD=0
16. MAD quarantine
17. legitimate long-distance travel preservation
18. cold-start <7 trips
19. delta_cv calculation
20. 3 consecutive stable trips
21. instability resetting stability
22. chronological no-future-leakage
23. deterministic repeated execution
24. schema validation
25. missing-value handling
"""

import json
import numpy as np
import pandas as pd
import pytest

from ml.src.profile import (
    ProfileConfig,
    build_user_profile,
    cluster_spatial_anchors,
    compute_cold_start_trajectory,
    compute_cv,
    compute_delta_cv,
    compute_mad,
    compute_trip_statistics,
    count_distinct_visits,
    detect_return_trips,
    evaluate_quarantine_observations,
    fit_feature_baseline,
)


# Helper fixtures & synthetic generators
def make_synthetic_trips(user_id="user_001", num_trips=10, base_dist=1000.0, std_dist=50.0):
    """Generate synthetic trips with controllable distances and timestamps."""
    records = []
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    rng = np.random.RandomState(42)

    for i in range(num_trips):
        start_t = base_time + pd.Timedelta(days=i)
        end_t = start_t + pd.Timedelta(hours=1)
        dist = max(base_dist + rng.normal(0, std_dist), 50.0)

        records.append({
            "user_id": user_id,
            "trajectory_id": f"{user_id}_traj_{i:03d}",
            "start_time": start_t,
            "end_time": end_t,
            "start_lat": 39.9042,
            "start_lon": 116.4074,
            "end_lat": 39.9042 + 0.001 * (i % 2),
            "end_lon": 116.4074 + 0.001 * (i % 2),
            "path_distance_m": dist,
        })
    return pd.DataFrame(records)


def make_synthetic_windows(user_id="user_001", num_windows=20, base_speed=5.0):
    """Generate synthetic window features for baseline testing."""
    records = []
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    rng = np.random.RandomState(42)

    for i in range(num_windows):
        st = base_time + pd.Timedelta(minutes=2 * i)
        et = st + pd.Timedelta(seconds=120)
        records.append({
            "user_id": user_id,
            "trajectory_id": f"{user_id}_traj_{i // 5:03d}",
            "window_id": f"{user_id}_win_{i:04d}",
            "start_time": st,
            "end_time": et,
            "mean_speed_mps": max(base_speed + rng.normal(0, 0.5), 0.1),
            "path_distance_m": 600.0,
            "straight_line_displacement_m": 500.0,
            "entropy_directional": 2.5 + rng.uniform(-0.2, 0.2),
            "tortuosity_index": 1.2 + rng.uniform(0.0, 0.1),
            "turn_frequency": 0.05 + rng.uniform(0.0, 0.02),
            "loop_metric": 0.1,
            "pacing_tendency": 0.05,
            "is_kinematically_evaluable": True,
            "has_extreme_kinematic_transition": False,
        })
    return pd.DataFrame(records)


# 1. Single-user profile test
def test_single_user_profile():
    trips_df = make_synthetic_trips(user_id="001", num_trips=8)
    windows_df = make_synthetic_windows(user_id="001", num_windows=15)
    config = ProfileConfig()

    profile = build_user_profile("001", windows_df, trips_df, config=config)
    assert profile.user_id == "001"
    assert profile.trip_count == 8
    assert profile.cold_start_status in ["RULE_BASED_FALLBACK", "ML_DRIVEN"]
    assert isinstance(profile.anchor_clusters, list)
    assert isinstance(profile.trip_stats, dict)
    assert isinstance(profile.baseline_distribution, dict)
    assert "mean_speed_mps" in profile.baseline_distribution


# 2. Multi-user isolation test
def test_multi_user_isolation():
    trips_u1 = make_synthetic_trips(user_id="user_A", num_trips=5, base_dist=100.0)
    trips_u2 = make_synthetic_trips(user_id="user_B", num_trips=12, base_dist=50000.0)
    combined_trips = pd.concat([trips_u1, trips_u2], ignore_index=True)

    win_u1 = make_synthetic_windows(user_id="user_A", num_windows=10, base_speed=1.5)
    win_u2 = make_synthetic_windows(user_id="user_B", num_windows=25, base_speed=25.0)
    combined_wins = pd.concat([win_u1, win_u2], ignore_index=True)

    profile_a = build_user_profile("user_A", combined_wins, combined_trips)
    profile_b = build_user_profile("user_B", combined_wins, combined_trips)

    assert profile_a.user_id == "user_A"
    assert profile_a.trip_count == 5
    assert profile_b.user_id == "user_B"
    assert profile_b.trip_count == 12

    # Verify baseline isolation
    speed_a = profile_a.baseline_distribution["mean_speed_mps"]["p95"]
    speed_b = profile_b.baseline_distribution["mean_speed_mps"]["p95"]
    assert speed_a < 5.0
    assert speed_b > 20.0


# 3. DBSCAN anchors test
def test_dbscan_anchors():
    # 6 points clustered near Beijing center (within 50m of each other)
    pts = []
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    for i in range(6):
        pts.append({
            "user_id": "test_user",
            "latitude": 39.9042 + 0.0001 * (i % 2),
            "longitude": 116.4074 + 0.0001 * (i // 2),
            "timestamp": base_time + pd.Timedelta(hours=i),
        })
    endpoints_df = pd.DataFrame(pts)
    config = ProfileConfig(anchor_eps_m=150.0, anchor_min_samples=3, anchor_confirmed_min_visits=5)

    anchors = cluster_spatial_anchors(endpoints_df, config=config)
    assert len(anchors) == 1
    assert anchors[0].observation_count == 6
    assert anchors[0].status == "CONFIRMED"
    assert pytest.approx(anchors[0].center_latitude, abs=0.01) == 39.9042
    assert pytest.approx(anchors[0].center_longitude, abs=0.01) == 116.4074


# 4. Anchor radius derivation (robust, not fixed, outlier resistance)
def test_anchor_radius_robust():
    # Cluster of 20 points within 30m, plus 1 outlier at 140m (still inside eps=150m)
    pts = []
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    for i in range(20):
        pts.append({
            "user_id": "u",
            "latitude": 39.9042 + 0.00005 * i,
            "longitude": 116.4074,
            "timestamp": base_time + pd.Timedelta(hours=i),
        })
    # Add 1 outlier
    pts.append({
        "user_id": "u",
        "latitude": 39.9042 + 0.0012,  # ~133 meters away
        "longitude": 116.4074,
        "timestamp": base_time + pd.Timedelta(hours=21),
    })
    endpoints_df = pd.DataFrame(pts)
    config = ProfileConfig(anchor_eps_m=150.0, anchor_min_samples=3, anchor_min_radius_m=25.0)

    anchors = cluster_spatial_anchors(endpoints_df, config=config)
    assert len(anchors) == 1
    # P95 should resist the single outlier at 133m
    assert anchors[0].radius_m < 120.0
    assert anchors[0].radius_m >= 25.0


# 5. Zero-anchor case
def test_zero_anchor_case():
    # Only 2 points widely separated (> 10 km apart), below min_samples=3
    pts = [
        {"user_id": "sparse_u", "latitude": 39.9042, "longitude": 116.4074, "timestamp": pd.Timestamp("2008-10-01")},
        {"user_id": "sparse_u", "latitude": 40.2042, "longitude": 116.8074, "timestamp": pd.Timestamp("2008-10-02")},
    ]
    endpoints_df = pd.DataFrame(pts)
    config = ProfileConfig(anchor_min_samples=3)

    anchors = cluster_spatial_anchors(endpoints_df, config=config)
    assert len(anchors) == 0


# 6. Insufficient observations handling
def test_insufficient_observations():
    trips_df = pd.DataFrame(columns=["user_id", "trajectory_id", "start_time", "end_time", "path_distance_m"])
    windows_df = pd.DataFrame(columns=["user_id", "trajectory_id", "window_id", "mean_speed_mps"])

    profile = build_user_profile("empty_u", windows_df, trips_df)
    assert profile.trip_count == 0
    assert profile.cold_start_status == "RULE_BASED_FALLBACK"
    assert profile.delta_cv is None
    assert profile.anchor_clusters == []
    assert profile.trip_stats["avg_return_trip_distance_m"] is None
    assert profile.baseline_distribution["mean_speed_mps"]["sample_size"] == 0
    assert profile.baseline_distribution["mean_speed_mps"]["distribution_type"] == "undefined"


# 7. Return-trip detection test
def test_return_trip_detection():
    # Start and end within 50m, path distance = 1500m -> genuine return trip
    trips = pd.DataFrame([{
        "user_id": "u",
        "trajectory_id": "t1",
        "start_lat": 39.9042,
        "start_lon": 116.4074,
        "end_lat": 39.9044,
        "end_lon": 116.4075,
        "path_distance_m": 1500.0,
    }])
    config = ProfileConfig(return_closure_threshold_m=200.0, return_min_path_m=200.0)
    ret = detect_return_trips(trips, config=config)
    assert len(ret) == 1
    assert ret.iloc[0]["trajectory_id"] == "t1"


# 8. No return-trip case
def test_no_return_trip_case():
    # One-way trip: start and end 10 km apart
    trips = pd.DataFrame([{
        "user_id": "u",
        "trajectory_id": "t1",
        "start_lat": 39.9042,
        "start_lon": 116.4074,
        "end_lat": 40.0042,
        "end_lon": 116.4074,
        "path_distance_m": 12000.0,
    }])
    stats = compute_trip_statistics(trips, anchors=[])
    assert stats.avg_return_trip_distance_m is None
    assert stats.cv_return_trip_distance is None


# 9. CV calculation test
def test_cv_calculation():
    vals = np.array([10.0, 10.0, 10.0, 10.0])
    # Std is 0 -> CV is 0
    cv = compute_cv(vals)
    assert cv == 0.0

    vals2 = np.array([10.0, 20.0])
    mean_val = 15.0
    std_val = np.std(vals2, ddof=1)
    expected_cv = std_val / mean_val
    assert compute_cv(vals2) == pytest.approx(expected_cv, rel=1e-5)


# 10. Zero-mean CV test
def test_zero_mean_cv():
    vals = np.array([0.0, 0.0, 0.0])
    cv = compute_cv(vals)
    assert cv is None  # Undefined, never zero-division


# 11. Gamma fitting test (< 30 samples, strictly positive)
def test_gamma_fitting():
    # 20 samples from Gamma(k=2, theta=5)
    rng = np.random.RandomState(42)
    sample = rng.gamma(shape=2.0, scale=5.0, size=20)
    baseline = fit_feature_baseline(sample, "test_feat")

    assert baseline.distribution_type == "gamma"
    assert baseline.sample_size == 20
    assert baseline.gamma_shape_k is not None
    assert baseline.gamma_scale_theta is not None
    assert baseline.gamma_shape_k > 0.0
    assert baseline.gamma_scale_theta > 0.0
    assert baseline.p95 is not None
    assert baseline.mad is not None


# 12. Invalid Gamma input (contains non-positive values or zero variance)
def test_invalid_gamma_input():
    # Sample containing zeros: Gamma is mathematically invalid
    sample = np.array([0.0, 0.0, 1.5, 2.0, 0.5])
    baseline = fit_feature_baseline(sample, "zero_containing")

    assert baseline.distribution_type == "empirical"
    assert baseline.gamma_shape_k is None
    assert baseline.gamma_scale_theta is None
    assert baseline.sample_size == 5

    # Sample with zero variance
    sample_flat = np.array([3.0, 3.0, 3.0, 3.0])
    baseline_flat = fit_feature_baseline(sample_flat, "flat")
    assert baseline_flat.distribution_type == "empirical"
    assert baseline_flat.gamma_shape_k is None


# 13. Empirical baseline >= 30 samples
def test_empirical_baseline_large_sample():
    rng = np.random.RandomState(42)
    sample = rng.normal(loc=10.0, scale=2.0, size=35)
    baseline = fit_feature_baseline(sample, "large_feat")

    assert baseline.distribution_type == "empirical"
    assert baseline.sample_size == 35
    assert baseline.gamma_shape_k is None
    assert baseline.gamma_scale_theta is None
    assert baseline.p95 is not None
    assert baseline.p99 is not None


# 14. P95/P99 calculation from observations
def test_p95_p99_calculation():
    sample = np.arange(1, 101, dtype=float)  # 1 to 100
    baseline = fit_feature_baseline(sample, "seq")
    assert baseline.p95 == pytest.approx(np.percentile(sample, 95))
    assert baseline.p99 == pytest.approx(np.percentile(sample, 99))


# 15. MAD = 0 handling
def test_mad_zero_handling():
    sample = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    mad_val, robust_scale = compute_mad(sample)
    assert mad_val == 0.0
    assert robust_scale == 0.0  # Explicit 0.0, no artificial noise


# 16. MAD quarantine detection
def test_mad_quarantine():
    windows = make_synthetic_windows(num_windows=20, base_speed=5.0)
    # Inject 1 unphysical extreme spike
    windows.loc[0, "mean_speed_mps"] = 500.0

    baselines = {"mean_speed_mps": {"mad": 0.5, "robust_scale": 0.74, "p95": 6.0}}
    config = ProfileConfig(quarantine_max_speed_mps=340.0)

    q_info = evaluate_quarantine_observations(windows, baselines, config=config)
    assert q_info["total_observations"] == 20
    assert q_info["quarantined_observations"] == 1
    assert q_info["quarantine_percentage"] == 5.0
    assert "unphysical_speed_burst" in q_info["quarantine_reasons"]


# 17. Legitimate long-distance travel preservation
def test_long_distance_preservation():
    # High-speed flight trajectory (250 m/s ~ 900 km/h, perfectly plausible airliner)
    trips = make_synthetic_trips(num_trips=2)
    # Trip 1: local, Trip 2: flight across 1000 km
    trips.loc[1, "end_lat"] = 30.0  # ~1100 km south
    trips.loc[1, "end_lon"] = 116.4074
    trips.loc[1, "path_distance_m"] = 1100000.0

    # Flight window: 240 m/s
    windows = make_synthetic_windows(num_windows=5, base_speed=240.0)
    config = ProfileConfig()

    stats = compute_trip_statistics(trips, anchors=[], windows_df=windows, config=config)
    assert stats.max_historical_radius_m > 1000000.0  # Preserved!

    # Verify not quarantined because 240 m/s < 340 m/s (sonic limit)
    baselines = {"mean_speed_mps": {"mad": 5.0, "robust_scale": 7.4}}
    q_info = evaluate_quarantine_observations(windows, baselines, config=config)
    assert q_info["quarantined_observations"] == 0


# 18. Cold-start < 7 trips requirement
def test_cold_start_under_7_trips():
    # 6 extremely stable trips
    trips = make_synthetic_trips(num_trips=6, base_dist=1000.0, std_dist=1.0)
    config = ProfileConfig(cold_start_min_trips=7)

    status, delta_cv, consec, hist = compute_cold_start_trajectory(trips, config=config)
    assert status == "RULE_BASED_FALLBACK"  # Must remain fallback because trips < 7


# 19. Delta CV calculation
def test_delta_cv_calculation():
    # Test formula: |CV_curr - CV_prev| / CV_prev
    assert compute_delta_cv(0.10, 0.20) == pytest.approx(0.50)
    assert compute_delta_cv(0.10, 0.0) == float("inf")
    assert compute_delta_cv(0.0, 0.0) == 0.0
    assert compute_delta_cv(None, 0.1) is None


# 20. 3 consecutive stable trips transition to ML_DRIVEN
def test_consecutive_stable_trips_transition():
    # 10 alternating trips where CV stabilizes with delta_cv < 0.05 for 3 consecutive trips
    dists = [1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 2000.0]
    trips = pd.DataFrame({
        "trajectory_id": [f"t_{i}" for i in range(len(dists))],
        "start_time": pd.date_range("2008-10-01", periods=len(dists), freq="D"),
        "path_distance_m": dists,
    })
    config = ProfileConfig(cold_start_min_trips=7, stability_consecutive_trips_required=3)

    status, delta_cv, consec, hist = compute_cold_start_trajectory(trips, config=config)
    assert consec >= 3
    assert status == "ML_DRIVEN"


# 21. Instability resetting stability counter
def test_instability_resets_stability():
    # Trips: first 4 very stable, 5th unstable, 6th, 7th stable
    dists = [1000.0, 1000.0, 1002.0, 1001.0, 50000.0, 50010.0, 50020.0]
    records = []
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    for i, d in enumerate(dists):
        records.append({
            "trajectory_id": f"t_{i}",
            "start_time": base_time + pd.Timedelta(days=i),
            "path_distance_m": d,
        })
    trips = pd.DataFrame(records)
    config = ProfileConfig(cold_start_min_trips=7, stability_consecutive_trips_required=3)

    status, delta_cv, consec, hist = compute_cold_start_trajectory(trips, config=config)
    # Trip 5 caused massive jump in CV -> reset counter to 0
    # Then trip 6 & 7 only achieved 2 consecutive stable trips -> cannot reach 3
    assert consec < 3
    assert status == "RULE_BASED_FALLBACK"


# 22. Chronological no-future-leakage test
def test_chronological_no_future_leakage():
    trips = make_synthetic_trips(num_trips=10)
    windows = make_synthetic_windows(num_windows=20)

    cutoff_time = pd.Timestamp("2008-10-03 23:59:59")  # First 3 days
    profile_t = build_user_profile(
        "001", windows, trips, as_of_time=cutoff_time
    )

    # Observations after cutoff_time must not be present
    assert profile_t.trip_count <= 4
    assert profile_t.trip_count < 10


# 23. Deterministic repeated execution
def test_deterministic_execution():
    trips = make_synthetic_trips(num_trips=8)
    windows = make_synthetic_windows(num_windows=15)
    ref_time = "2008-10-15T00:00:00"

    p1 = build_user_profile("001", windows, trips, as_of_time=ref_time).to_record()
    p2 = build_user_profile("001", windows, trips, as_of_time=ref_time).to_record()

    # Compare all core components
    for key in p1:
        assert p1[key] == p2[key]


# 24. Schema validation
def test_schema_validation():
    trips = make_synthetic_trips(num_trips=8)
    windows = make_synthetic_windows(num_windows=15)

    p_dict = build_user_profile("001", windows, trips).to_record()

    required_keys = [
        "user_id",
        "cold_start_status",
        "trip_count",
        "delta_cv",
        "consecutive_stable_trips",
        "anchor_clusters",
        "trip_stats",
        "baseline_distribution",
        "quarantine_stats",
        "stability_history",
        "last_updated",
    ]
    for k in required_keys:
        assert k in p_dict

    # Check that nested fields parse valid JSON
    assert isinstance(json.loads(p_dict["anchor_clusters"]), list)
    assert isinstance(json.loads(p_dict["trip_stats"]), dict)
    assert isinstance(json.loads(p_dict["baseline_distribution"]), dict)

    # Check trip_stats fields
    ts = json.loads(p_dict["trip_stats"])
    assert "avg_return_trip_distance_m" in ts
    assert "cv_return_trip_distance" in ts
    assert "avg_step_speed_mps" in ts
    assert "speed_std_dev" in ts
    assert "max_historical_radius_m" in ts


# 25. Missing value handling
def test_missing_value_handling():
    # User with single trip (no pairs, no return trips, std undefined)
    trips = pd.DataFrame([{
        "user_id": "solo_u",
        "trajectory_id": "solo_t",
        "start_time": pd.Timestamp("2008-10-01 08:00:00"),
        "end_time": pd.Timestamp("2008-10-01 09:00:00"),
        "start_lat": 39.9042,
        "start_lon": 116.4074,
        "end_lat": 40.0042,
        "end_lon": 116.4074,
        "path_distance_m": 12000.0,
    }])
    windows = pd.DataFrame(columns=["user_id", "trajectory_id", "window_id", "mean_speed_mps"])

    profile = build_user_profile("solo_u", windows, trips)
    # Verify missing fields are None, never 0.0
    assert profile.delta_cv is None
    assert profile.trip_stats["avg_return_trip_distance_m"] is None
    assert profile.trip_stats["cv_return_trip_distance"] is None
    assert profile.trip_stats["avg_step_speed_mps"] is None


# 26. Anchor confirmation requires distinct visits, preventing false inflation from duplicates
def test_anchor_distinct_visits_prevents_duplicate_confirmation():
    # 3 stationary trajectories (<200m) occurring consecutively at the same location
    # These produce 6 raw endpoints, but represent only 1 distinct visit episode
    base_time = pd.Timestamp("2008-10-01 08:00:00")
    trips_data = []
    for i in range(3):
        trips_data.append({
            "user_id": "stay_user",
            "trajectory_id": f"t_{i}",
            "start_time": base_time + pd.Timedelta(hours=2 * i),
            "end_time": base_time + pd.Timedelta(hours=2 * i + 1),
            "start_lat": 39.9042,
            "start_lon": 116.4074,
            "end_lat": 39.9042 + 0.0001,
            "end_lon": 116.4074 + 0.0001,
            "path_distance_m": 50.0,  # Stationary trajectory (<200m)
        })
    trips_df = pd.DataFrame(trips_data)
    windows_df = pd.DataFrame(columns=["user_id", "trajectory_id", "window_id", "mean_speed_mps"])

    profile = build_user_profile("stay_user", windows_df, trips_df)
    assert len(profile.anchor_clusters) == 1
    anchor = profile.anchor_clusters[0]

    # Raw endpoints = 6, but distinct visit episodes = 1
    assert anchor["raw_endpoint_count"] == 6
    assert anchor["observation_count"] == 1
    # Must remain PENDING_CAREGIVER_REVIEW because distinct visits (1) < 5
    assert anchor["status"] == "PENDING_CAREGIVER_REVIEW"


# 27. Cold-start CV does not substitute zeros for missing values
def test_cold_start_no_zero_substitution_for_missing():
    # 8 trips where 2 have missing (NaN) path_distance_m
    dists = [1000.0, 1010.0, np.nan, 1020.0, np.nan, 1015.0, 1025.0, 1030.0]
    trips = pd.DataFrame({
        "trajectory_id": [f"t_{i}" for i in range(len(dists))],
        "start_time": pd.date_range("2008-10-01", periods=len(dists), freq="D"),
        "path_distance_m": dists,
    })
    config = ProfileConfig(cold_start_min_trips=7)

    status, delta_cv, consec, hist = compute_cold_start_trajectory(trips, config=config)

    # 6 valid trips out of 8 -> valid trips < 7, so must remain RULE_BASED_FALLBACK
    assert len(hist) == 6
    assert status == "RULE_BASED_FALLBACK"
    # Verify no distance in history is 0.0 or NaN
    for record in hist:
        assert record["path_distance_m"] > 0.0
        assert not np.isnan(record["path_distance_m"])


# 28. Anchor visit consolidation boundary threshold (immediately below, at, and above)
def test_anchor_visit_consolidation_boundary_threshold():
    """Verify stationary trajectory consolidation strictly at the boundary threshold.

    Tests:
    - Immediately below default boundary (199.9m): consolidated as 1 visit episode.
    - Exactly at default boundary (200.0m): promoted to 2 distinct visit interactions (round-trip).
    - Immediately above default boundary (200.1m): promoted to 2 distinct visit interactions.
    - Configurable boundary test (e.g. 150.0m): dynamically respects configuration.
    """
    base_time = pd.Timestamp("2008-10-01 08:00:00")

    def make_cluster_endpoints(dist: float) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "user_id": "u1",
                "trajectory_id": "t_loop",
                "timestamp": base_time,
                "latitude": 39.9042,
                "longitude": 116.4074,
                "endpoint_type": "start",
                "path_distance_m": dist,
            },
            {
                "user_id": "u1",
                "trajectory_id": "t_loop",
                "timestamp": base_time + pd.Timedelta(minutes=30),
                "latitude": 39.9043,
                "longitude": 116.4075,
                "endpoint_type": "end",
                "path_distance_m": dist,
            },
        ])

    # 1. Immediately below boundary: 199.9 m -> stationary dwell (1 visit episode)
    df_below = make_cluster_endpoints(199.9)
    visits_below, raw_below = count_distinct_visits(df_below)
    assert raw_below == 2
    assert visits_below == 1

    # 2. Exactly at boundary: 200.0 m -> genuine departing & returning loop (2 visit interactions)
    df_at = make_cluster_endpoints(200.0)
    visits_at, raw_at = count_distinct_visits(df_at)
    assert raw_at == 2
    assert visits_at == 2

    # 3. Immediately above boundary: 200.1 m -> genuine departing & returning loop (2 visit interactions)
    df_above = make_cluster_endpoints(200.1)
    visits_above, raw_above = count_distinct_visits(df_above)
    assert raw_above == 2
    assert visits_above == 2

    # 4. Custom configured boundary (150.0 m)
    cfg_custom = ProfileConfig(anchor_stationary_path_threshold_m=150.0)
    df_custom_below = make_cluster_endpoints(149.9)
    df_custom_at = make_cluster_endpoints(150.0)
    df_custom_above = make_cluster_endpoints(150.1)

    assert count_distinct_visits(df_custom_below, config=cfg_custom)[0] == 1
    assert count_distinct_visits(df_custom_at, config=cfg_custom)[0] == 2
    assert count_distinct_visits(df_custom_above, config=cfg_custom)[0] == 2

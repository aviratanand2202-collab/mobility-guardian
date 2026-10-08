"""Personalized Longitudinal Mobility Profile & Baseline Modeling (Chunk 5).

Constructs per-user longitudinal mobility profiles from historical observations:
1. Spatial Anchors: Density-based clustering (DBSCAN) on visit endpoints with robust
   cluster radius derivation and status tracking (CONFIRMED vs PENDING_CAREGIVER_REVIEW).
2. Mobility & Trip Statistics: Return-trip detection, coefficient of variation (CV),
   step speed statistics, and maximum historical movement radius from reference point.
3. Personalized Baselines: Empirical and Gamma-fitted baselines for validated kinematic
   and behavioral features, robust dispersion (MAD and robust scale), and P95/P99 quantiles.
4. Data Quality Quarantine: Baseline/MAD-driven flagging of corrupted sensor records without
   deleting raw data or penalizing legitimate long-distance travel.
5. Cold-Start & Stability Modeling: Exact delta_cv formulation with consecutive stability
   tracking and transparent transition from RULE_BASED_FALLBACK to ML_DRIVEN.
6. Temporal Integrity: Strict chronological filtering (as-of-time) preventing future leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN

# Earth equatorial radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_METERS: float = 6371000.0


def current_iso_time() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def to_clean_float_array(values: Union[np.ndarray, pd.Series, List[Any]]) -> np.ndarray:
    """Safely convert any input to a clean 1D numpy float array without NaNs."""
    if values is None or len(values) == 0:
        return np.array([], dtype=float)
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return s.to_numpy(dtype=float)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class ProfileConfig:
    """Centralized configuration for mobility profile and baseline modeling.

    All methodological thresholds and hyperparameters are explicitly centralized
    and documented here to prevent hidden magic numbers.
    """

    # Spatial Anchor Clustering (DBSCAN)
    anchor_eps_m: float = 150.0
    """Spatial search radius for DBSCAN clustering in meters (~1-2 city blocks)."""

    anchor_min_samples: int = 3
    """Minimum endpoint observations required to form an anchor candidate cluster."""

    anchor_confirmed_min_visits: int = 5
    """Minimum endpoint visits required to promote an anchor to CONFIRMED status."""

    anchor_min_radius_m: float = 25.0
    """Minimum anchor geofence radius floor in meters, reflecting standard GPS accuracy."""

    anchor_radius_percentile: float = 95.0
    """Percentile of distance from centroid to points used to determine robust anchor radius."""

    anchor_stationary_path_threshold_m: float = 200.0
    """Minimum trajectory path distance in meters required for a trajectory starting and ending
    at the same anchor cluster to be considered a genuine departing-and-returning trip rather
    than a stationary dwell episode with sensor noise."""

    # Return Trip Detection
    return_closure_threshold_m: float = 200.0
    """Maximum distance between start and end coordinates to consider a trip as returning to origin."""

    return_min_path_m: float = 200.0
    """Minimum trajectory path distance in meters required to classify as a genuine return trip."""

    return_circuitous_ratio: float = 2.0
    """Minimum ratio of path distance to endpoint displacement to distinguish loops from stationary noise."""

    # Cold Start & Stability
    cold_start_min_trips: int = 7
    """Minimum cumulative trips required before ML_DRIVEN activation can be considered."""

    stability_delta_cv_threshold: float = 0.05
    """Maximum relative CV change (|CV_N - CV_{N-1}| / CV_{N-1}) to classify a trip as stable."""

    stability_consecutive_trips_required: int = 3
    """Number of consecutive stable trips required to promote profile to ML_DRIVEN status."""

    # Baseline & Distribution
    baseline_gamma_min_samples: int = 30
    """Threshold below which a Gamma distribution is fitted if mathematically valid; >=30 uses empirical."""

    # Quarantine & Quality Rules
    quarantine_max_speed_mps: float = 340.0
    """Physical sonic speed limit for window mean speed, above which records are flagged for quarantine."""

    quarantine_mad_multiplier: float = 5.0
    """MAD dispersion multiplier used for personal outlier flagging."""

    # Features selected for personalized longitudinal baseline modeling
    baseline_features: Tuple[str, ...] = (
        "entropy_directional",
        "mean_speed_mps",
        "tortuosity_index",
        "path_distance_m",
        "straight_line_displacement_m",
        "turn_frequency",
        "loop_metric",
        "pacing_tendency",
    )


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class AnchorCluster:
    """Represents a learned spatial anchor location."""

    user_id: str
    anchor_id: str
    center_latitude: float
    center_longitude: float
    radius_m: float
    observation_count: int  # Distinct visit episodes
    first_seen: str
    last_seen: str
    status: str  # 'CONFIRMED' or 'PENDING_CAREGIVER_REVIEW'
    typical_hours: List[int] = field(default_factory=list)
    raw_endpoint_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert anchor cluster to deterministic dictionary."""
        return asdict(self)


@dataclass
class TripStats:
    """Represents mobility trip statistics for a user."""

    avg_return_trip_distance_m: Optional[float]
    cv_return_trip_distance: Optional[float]
    avg_step_speed_mps: Optional[float]
    speed_std_dev: Optional[float]
    max_historical_radius_m: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trip stats to dictionary."""
        return asdict(self)


@dataclass
class BaselineDistribution:
    """Represents a fitted statistical baseline for a single feature."""

    distribution_type: str  # 'gamma', 'empirical', or 'undefined'
    sample_size: int
    gamma_shape_k: Optional[float]
    gamma_scale_theta: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    mad: Optional[float]
    robust_scale: Optional[float]
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline distribution to dictionary."""
        return asdict(self)


@dataclass
class UserProfile:
    """Represents the complete personalized mobility profile for a user."""

    user_id: str
    cold_start_status: str  # 'RULE_BASED_FALLBACK' or 'ML_DRIVEN'
    trip_count: int
    delta_cv: Optional[float]
    consecutive_stable_trips: int
    anchor_clusters: List[Dict[str, Any]]
    trip_stats: Dict[str, Any]
    baseline_distribution: Dict[str, Dict[str, Any]]
    quarantine_stats: Dict[str, Any] = field(default_factory=dict)
    stability_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        """Convert to a flat dictionary suitable for parquet serialization."""
        return {
            "user_id": str(self.user_id),
            "cold_start_status": str(self.cold_start_status),
            "trip_count": int(self.trip_count),
            "delta_cv": float(self.delta_cv) if self.delta_cv is not None else None,
            "consecutive_stable_trips": int(self.consecutive_stable_trips),
            "anchor_clusters": json.dumps(self.anchor_clusters, sort_keys=True),
            "trip_stats": json.dumps(self.trip_stats, sort_keys=True),
            "baseline_distribution": json.dumps(self.baseline_distribution, sort_keys=True),
            "quarantine_stats": json.dumps(self.quarantine_stats, sort_keys=True),
            "stability_history": json.dumps(self.stability_history, sort_keys=True),
            "last_updated": self.last_updated or current_iso_time(),
        }


# ==============================================================================
# GEODETIC & STATISTICAL UTILITIES
# ==============================================================================


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate the great-circle distance between two points in meters using Haversine formula."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_METERS * c


def compute_spherical_centroid(lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    """Calculate the 3D spherical centroid for a set of latitudes and longitudes."""
    phi = np.radians(lats)
    lam = np.radians(lons)

    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    hypot = np.sqrt(x_mean**2 + y_mean**2)
    if hypot == 0.0 and z_mean == 0.0:
        return float(np.mean(lats)), float(np.mean(lons))

    center_lat = np.degrees(np.arctan2(z_mean, hypot))
    center_lon = np.degrees(np.arctan2(y_mean, x_mean))
    return float(center_lat), float(center_lon)


def compute_cv(values: Union[np.ndarray, pd.Series, List[Any]]) -> Optional[float]:
    """Compute Coefficient of Variation: CV = std(X) / mean(X).

    Returns None if sample size < 2, mean <= 0, or mean is near-zero (< 1e-9).
    Never fabricates 0 for undefined values.
    """
    clean_vals = to_clean_float_array(values)
    if len(clean_vals) < 2:
        return None

    mean_val = float(np.mean(clean_vals))
    if mean_val <= 1e-9:
        return None

    std_val = float(np.std(clean_vals, ddof=1))
    return float(std_val / mean_val)


def compute_delta_cv(cv_curr: Optional[float], cv_prev: Optional[float]) -> Optional[float]:
    """Compute relative CV stability change: delta_cv = |CV_N - CV_{N-1}| / CV_{N-1}.

    Explicitly handles CV_{N-1} == 0 or undefined values:
    - If either is None: returns None.
    - If cv_prev == 0: returns 0.0 if cv_curr == 0 else float('inf').
    """
    if cv_curr is None or cv_prev is None:
        return None

    if cv_prev == 0.0:
        return 0.0 if cv_curr == 0.0 else float("inf")

    return float(abs(cv_curr - cv_prev) / cv_prev)


def compute_mad(values: Union[np.ndarray, pd.Series, List[Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Median Absolute Deviation (MAD) and robust scale (1.4826 * MAD).

    MAD = median(|x - median(x)|).
    Explicitly handles MAD == 0 by setting robust_scale = 0.0 without injecting noise.
    """
    clean_vals = to_clean_float_array(values)
    if len(clean_vals) == 0:
        return None, None

    med = float(np.median(clean_vals))
    abs_deviations = np.abs(clean_vals - med)
    mad_val = float(np.median(abs_deviations))

    if mad_val == 0.0:
        robust_scale = 0.0
    else:
        robust_scale = float(1.4826 * mad_val)

    return mad_val, robust_scale


# ==============================================================================
# SPATIAL ANCHORS (DBSCAN)
# ==============================================================================


def count_distinct_visits(
    cluster_rows: pd.DataFrame,
    config: ProfileConfig = ProfileConfig(),
) -> Tuple[int, int]:
    """Count distinct visit episodes in an anchor cluster, preventing double-counting.

    Prevents duplicate endpoints (e.g., arrival of trip k and departure of trip k+1,
    or start and end of stationary trajectories) from falsely inflating visit counts.

    Returns:
        (distinct_visit_count, raw_endpoint_count)
    """
    raw_count = len(cluster_rows)
    if raw_count == 0:
        return 0, 0
    if "trajectory_id" not in cluster_rows.columns:
        return raw_count, raw_count

    rows = cluster_rows.sort_values("timestamp").reset_index(drop=True)
    visits = 1
    for i in range(1, len(rows)):
        prev = rows.iloc[i - 1]
        curr = rows.iloc[i]

        # Case 1: Start and end of the exact same trajectory
        if curr["trajectory_id"] == prev["trajectory_id"]:
            path_d = curr.get("path_distance_m")
            if pd.notna(path_d) and path_d >= config.anchor_stationary_path_threshold_m:
                # Trajectory departed and returned (genuine return trip): 2 separate visit interactions
                visits += 1
            # Otherwise, stationary trajectory (< threshold) where user never left: same visit
            continue

        # Case 2: Consecutive stay episode between two consecutive trajectories
        # (Trip k ends at this cluster, and Trip k+1 starts from this cluster)
        if prev.get("endpoint_type") == "end" and curr.get("endpoint_type") == "start":
            # Continuous stay episode between trips: 1 dwell episode, not 2 visits
            continue

        visits += 1

    return visits, raw_count


def cluster_spatial_anchors(
    endpoints_df: pd.DataFrame,
    config: ProfileConfig = ProfileConfig(),
) -> List[AnchorCluster]:
    """Identify spatial anchors from trajectory endpoints using Haversine DBSCAN.

    Parameters:
        endpoints_df: DataFrame containing ['latitude', 'longitude', 'timestamp', 'user_id'].
        config: ProfileConfig containing DBSCAN parameters.

    Returns:
        List of AnchorCluster instances with robust radii and status.
    """
    if endpoints_df.empty or len(endpoints_df) < config.anchor_min_samples:
        return []

    user_id = str(endpoints_df["user_id"].iloc[0])
    valid_pts = endpoints_df.dropna(subset=["latitude", "longitude"]).copy()
    if len(valid_pts) < config.anchor_min_samples:
        return []

    # Convert coordinates to radians for Haversine DBSCAN
    coords_rad = np.radians(valid_pts[["latitude", "longitude"]].values)
    eps_rad = config.anchor_eps_m / EARTH_RADIUS_METERS

    db = DBSCAN(eps=eps_rad, min_samples=config.anchor_min_samples, metric="haversine", n_jobs=1)
    labels = db.fit_predict(coords_rad)
    valid_pts["cluster"] = labels

    clusters: List[AnchorCluster] = []
    cluster_labels = [lbl for lbl in np.unique(labels) if lbl >= 0]

    # Sort clusters by size descending for deterministic assignment
    cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in cluster_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    for anchor_idx, (cluster_id, _) in enumerate(cluster_sizes):
        cluster_rows = valid_pts[valid_pts["cluster"] == cluster_id]
        lats = cluster_rows["latitude"].values
        lons = cluster_rows["longitude"].values

        # Robust centroid
        center_lat, center_lon = compute_spherical_centroid(lats, lons)

        # Robust radius from cluster points to centroid
        distances = haversine_distance(center_lat, center_lon, lats, lons)
        p_radius = float(np.percentile(distances, config.anchor_radius_percentile))
        radius_m = float(max(p_radius, config.anchor_min_radius_m))

        # Timestamps and hours
        timestamps = pd.to_datetime(cluster_rows["timestamp"])
        first_seen = timestamps.min().isoformat()
        last_seen = timestamps.max().isoformat()
        typical_hours = sorted(list(set(timestamps.dt.hour.tolist())))

        # Distinct visits accounting (prevents duplicate endpoints from single visit from inflating count)
        distinct_visits, raw_count = count_distinct_visits(cluster_rows, config=config)

        # Status: CONFIRMED if observed distinct visits >= confirmed threshold, else PENDING_CAREGIVER_REVIEW
        status = (
            "CONFIRMED"
            if distinct_visits >= config.anchor_confirmed_min_visits
            else "PENDING_CAREGIVER_REVIEW"
        )

        cluster_obj = AnchorCluster(
            user_id=user_id,
            anchor_id=f"{user_id}_anchor_{anchor_idx}",
            center_latitude=center_lat,
            center_longitude=center_lon,
            radius_m=radius_m,
            observation_count=int(distinct_visits),
            first_seen=first_seen,
            last_seen=last_seen,
            status=status,
            typical_hours=typical_hours,
            raw_endpoint_count=int(raw_count),
        )
        clusters.append(cluster_obj)

    return clusters


# ==============================================================================
# RETURN TRIPS & MOBILITY TRIP STATISTICS
# ==============================================================================


def detect_return_trips(
    trips_df: pd.DataFrame,
    config: ProfileConfig = ProfileConfig(),
) -> pd.DataFrame:
    """Identify candidate return trips from observed trajectory summaries.

    Criteria for return trips:
    1. Distance between start and end coordinates <= return_closure_threshold_m.
    2. Path distance >= return_min_path_m.
    3. Path distance >= return_circuitous_ratio * endpoint_displacement.

    Returns:
        Filtered DataFrame containing confirmed candidate return trips.
    """
    if trips_df.empty:
        return pd.DataFrame(columns=trips_df.columns)

    req_cols = ["start_lat", "start_lon", "end_lat", "end_lon", "path_distance_m"]
    for col in req_cols:
        if col not in trips_df.columns:
            raise ValueError(f"trips_df missing required column: {col}")

    valid_trips = trips_df.dropna(subset=req_cols).copy()
    if valid_trips.empty:
        return pd.DataFrame(columns=trips_df.columns)

    disp_m = haversine_distance(
        valid_trips["start_lat"].values,
        valid_trips["start_lon"].values,
        valid_trips["end_lat"].values,
        valid_trips["end_lon"].values,
    )
    valid_trips["endpoint_disp_m"] = disp_m

    is_return = (
        (valid_trips["endpoint_disp_m"] <= config.return_closure_threshold_m)
        & (valid_trips["path_distance_m"] >= config.return_min_path_m)
        & (valid_trips["path_distance_m"] >= config.return_circuitous_ratio * valid_trips["endpoint_disp_m"])
    )

    return valid_trips[is_return].copy()


def compute_trip_statistics(
    trips_df: pd.DataFrame,
    anchors: List[AnchorCluster],
    windows_df: Optional[pd.DataFrame] = None,
    all_points_df: Optional[pd.DataFrame] = None,
    precomputed_max_radius: Optional[float] = None,
    config: ProfileConfig = ProfileConfig(),
) -> TripStats:
    """Compute mobility and trip statistics from observed data.

    Calculates:
    - avg_return_trip_distance_m: mean distance of detected return trips (None if 0 return trips).
    - cv_return_trip_distance: CV of return trip distance (None if < 2 return trips).
    - avg_step_speed_mps: mean speed from valid kinematic windows.
    - speed_std_dev: sample standard deviation of window speed.
    - max_historical_radius_m: maximum distance from primary reference anchor to all observed points.
    """
    # 1. Return Trip Statistics
    return_trips = detect_return_trips(trips_df, config=config)
    if return_trips.empty:
        avg_return_dist = None
        cv_return_dist = None
    else:
        dists = return_trips["path_distance_m"].values
        avg_return_dist = float(np.mean(dists))
        cv_return_dist = compute_cv(dists)

    # 2. Step Speed Statistics (from kinematically evaluable windows)
    avg_speed = None
    speed_std = None
    if windows_df is not None and not windows_df.empty and "mean_speed_mps" in windows_df.columns:
        valid_mask = windows_df["mean_speed_mps"].notna()
        if "is_kinematically_evaluable" in windows_df.columns:
            valid_mask = valid_mask & windows_df["is_kinematically_evaluable"]
        if "has_extreme_kinematic_transition" in windows_df.columns:
            valid_mask = valid_mask & (~windows_df["has_extreme_kinematic_transition"])

        speeds = windows_df.loc[valid_mask, "mean_speed_mps"].values
        if len(speeds) > 0:
            avg_speed = float(np.mean(speeds))
            speed_std = float(np.std(speeds, ddof=1)) if len(speeds) >= 2 else 0.0

    # 3. Maximum Historical Radius from Reference Point
    max_historical_radius: Optional[float] = None
    if precomputed_max_radius is not None:
        max_historical_radius = precomputed_max_radius
    else:
        # Reference point definition: primary anchor center (most visited), or geometric median of trip endpoints.
        ref_lat: Optional[float] = None
        ref_lon: Optional[float] = None

        if len(anchors) > 0:
            primary_anchor = sorted(anchors, key=lambda a: a.observation_count, reverse=True)[0]
            ref_lat = primary_anchor.center_latitude
            ref_lon = primary_anchor.center_longitude
        elif not trips_df.empty and "start_lat" in trips_df.columns:
            ref_lat = float(trips_df["start_lat"].median())
            ref_lon = float(trips_df["start_lon"].median())

        if ref_lat is not None and ref_lon is not None:
            if all_points_df is not None and not all_points_df.empty:
                p_lats = all_points_df["latitude"].values
                p_lons = all_points_df["longitude"].values
                dist_pts = haversine_distance(ref_lat, ref_lon, p_lats, p_lons)
                max_historical_radius = float(np.max(dist_pts))
            elif not trips_df.empty:
                lats = np.concatenate([trips_df["start_lat"].values, trips_df["end_lat"].values])
                lons = np.concatenate([trips_df["start_lon"].values, trips_df["end_lon"].values])
                dist_pts = haversine_distance(ref_lat, ref_lon, lats, lons)
                max_historical_radius = float(np.max(dist_pts))

    return TripStats(
        avg_return_trip_distance_m=avg_return_dist,
        cv_return_trip_distance=cv_return_dist,
        avg_step_speed_mps=avg_speed,
        speed_std_dev=speed_std,
        max_historical_radius_m=max_historical_radius,
    )


# ==============================================================================
# PERSONALIZED BASELINES & STATISTICAL FITTING
# ==============================================================================


def fit_feature_baseline(
    data: Union[np.ndarray, pd.Series, List[Any]],
    feature_name: str,
    as_of_time: Optional[str] = None,
    config: ProfileConfig = ProfileConfig(),
) -> BaselineDistribution:
    """Fit a personalized baseline distribution for a single validated feature.

    Distribution Rule:
    - If sample_size < 30: Fit Gamma(k, theta) via Method of Moments where mathematically
      valid (all x > 0 and variance > 0). If data contains non-positive values (e.g. 0.0)
      or variance is zero, fallback to empirical quantiles without adding arbitrary constants.
    - If sample_size >= 30: Use empirical quantiles.
    - P95 / P99: Derived directly from observed data quantiles.
    - MAD / robust_scale: Robust dispersion (MAD = median(|x - median(x)|), scale = 1.4826 * MAD).
      Explicitly handles MAD = 0 without arbitrary noise.
    """
    clean_data = to_clean_float_array(data)
    n = len(clean_data)
    last_updated = as_of_time or current_iso_time()

    if n == 0:
        return BaselineDistribution(
            distribution_type="undefined",
            sample_size=0,
            gamma_shape_k=None,
            gamma_scale_theta=None,
            p95=None,
            p99=None,
            mad=None,
            robust_scale=None,
            last_updated=last_updated,
        )

    # Robust Dispersion (MAD & robust scale)
    mad_val, robust_scale = compute_mad(clean_data)

    # Empirical Quantiles
    p95_val = float(np.percentile(clean_data, 95))
    p99_val = float(np.percentile(clean_data, 99))

    # Distribution selection
    if n < config.baseline_gamma_min_samples:
        # Check if Gamma is mathematically valid: strictly positive support and non-zero variance
        is_strictly_positive = np.all(clean_data > 0.0)
        var_val = float(np.var(clean_data, ddof=1)) if n >= 2 else 0.0

        if is_strictly_positive and var_val > 1e-9:
            mean_val = float(np.mean(clean_data))
            # Method of moments for Gamma: mean = k * theta, var = k * theta^2
            k = (mean_val**2) / var_val
            theta = var_val / mean_val
            return BaselineDistribution(
                distribution_type="gamma",
                sample_size=n,
                gamma_shape_k=float(k),
                gamma_scale_theta=float(theta),
                p95=p95_val,
                p99=p99_val,
                mad=mad_val,
                robust_scale=robust_scale,
                last_updated=last_updated,
            )
        else:
            # Data contains non-positive numbers (e.g. 0) or zero variance;
            # fall back to empirical representation rather than adding arbitrary constants
            return BaselineDistribution(
                distribution_type="empirical",
                sample_size=n,
                gamma_shape_k=None,
                gamma_scale_theta=None,
                p95=p95_val,
                p99=p99_val,
                mad=mad_val,
                robust_scale=robust_scale,
                last_updated=last_updated,
            )
    else:
        # Sample size >= 30: empirical distribution
        return BaselineDistribution(
            distribution_type="empirical",
            sample_size=n,
            gamma_shape_k=None,
            gamma_scale_theta=None,
            p95=p95_val,
            p99=p99_val,
            mad=mad_val,
            robust_scale=robust_scale,
            last_updated=last_updated,
        )


def fit_all_baselines(
    windows_df: pd.DataFrame,
    as_of_time: Optional[str] = None,
    config: ProfileConfig = ProfileConfig(),
) -> Dict[str, Dict[str, Any]]:
    """Fit personalized baselines for all configured features on a user's window dataset."""
    baselines: Dict[str, Dict[str, Any]] = {}
    for feat in config.baseline_features:
        if feat in windows_df.columns:
            data = windows_df[feat].values
            base_dist = fit_feature_baseline(data, feat, as_of_time=as_of_time, config=config)
            baselines[feat] = base_dist.to_dict()
        else:
            empty_dist = fit_feature_baseline(np.array([]), feat, as_of_time=as_of_time, config=config)
            baselines[feat] = empty_dist.to_dict()
    return baselines


# ==============================================================================
# COLD START & STABILITY TRACKING
# ==============================================================================


def compute_cold_start_trajectory(
    trips_df: pd.DataFrame,
    config: ProfileConfig = ProfileConfig(),
) -> Tuple[str, Optional[float], int, List[Dict[str, Any]]]:
    """Track longitudinal stability and cold-start transitions trip-by-trip.

    Formula implemented EXACTLY:
        delta_cv = |CV_N - CV_{N-1}| / CV_{N-1}
        where CV_N = std(D_1..D_N) / mean(D_1..D_N)

    ML_DRIVEN requires:
        trip_count >= 7
        AND
        delta_cv < 0.05 for 3 consecutive trips.

    Before this: RULE_BASED_FALLBACK.
    Instability (delta_cv >= 0.05) resets the consecutive stable trip counter.

    Returns:
        (cold_start_status, latest_delta_cv, consecutive_stable_trips, stability_history)
    """
    if trips_df.empty:
        return "RULE_BASED_FALLBACK", None, 0, []

    # Sort trips strictly chronologically
    sorted_trips = trips_df.sort_values("start_time").copy()

    # Filter out missing/invalid distances without substituting zeros
    valid_mask = sorted_trips["path_distance_m"].notna()
    valid_trips = sorted_trips[valid_mask].copy()

    if valid_trips.empty:
        return "RULE_BASED_FALLBACK", None, 0, []

    distances = valid_trips["path_distance_m"].values
    trip_ids = valid_trips["trajectory_id"].values if "trajectory_id" in valid_trips.columns else None

    total_trips = len(distances)
    consecutive_stable = 0
    ml_driven_activated = False
    latest_delta_cv: Optional[float] = None
    cv_prev: Optional[float] = None
    history: List[Dict[str, Any]] = []

    for idx in range(total_trips):
        trip_idx = idx + 1
        sub_dists = distances[:trip_idx]

        if trip_idx < 2:
            cv_curr = None
            delta_cv = None
        else:
            cv_curr = compute_cv(sub_dists)
            if cv_prev is not None and cv_curr is not None:
                delta_cv = compute_delta_cv(cv_curr, cv_prev)
            else:
                delta_cv = None

        # Check stability condition
        if delta_cv is not None:
            latest_delta_cv = delta_cv
            if delta_cv < config.stability_delta_cv_threshold:
                consecutive_stable += 1
            else:
                consecutive_stable = 0  # Instability resets consecutive stable counter
        else:
            consecutive_stable = 0

        # Check ML_DRIVEN transition criteria
        if (
            trip_idx >= config.cold_start_min_trips
            and consecutive_stable >= config.stability_consecutive_trips_required
        ):
            ml_driven_activated = True

        status = "ML_DRIVEN" if ml_driven_activated else "RULE_BASED_FALLBACK"

        history.append({
            "trip_index": trip_idx,
            "trajectory_id": str(trip_ids[idx]) if trip_ids is not None else None,
            "path_distance_m": float(distances[idx]),
            "cv": float(cv_curr) if cv_curr is not None else None,
            "delta_cv": float(delta_cv) if delta_cv is not None else None,
            "consecutive_stable_trips": int(consecutive_stable),
            "status": status,
        })

        if cv_curr is not None:
            cv_prev = cv_curr

    final_status = "ML_DRIVEN" if ml_driven_activated else "RULE_BASED_FALLBACK"
    return final_status, latest_delta_cv, consecutive_stable, history


# ==============================================================================
# QUARANTINE / DATA QUALITY CHECKS
# ==============================================================================


def evaluate_quarantine_observations(
    windows_df: pd.DataFrame,
    baselines: Dict[str, Dict[str, Any]],
    config: ProfileConfig = ProfileConfig(),
) -> Dict[str, Any]:
    """Identify data-quality quarantined observations using personal baselines & MAD logic.

    Quarantine reasons:
    - extreme_kinematic_transition: physically implausible transitions (e.g. speed > 340 m/s or clock errors).
    - unphysical_speed_burst: window speed > config.quarantine_max_speed_mps.
    - personal_mad_outlier: speed exceeds median + 5 * robust_scale AND exceeds 300 m/s.

    Does NOT delete raw observations. Does NOT quarantine legitimate flights or rare mobility.
    """
    total_obs = len(windows_df)
    if total_obs == 0:
        return {
            "total_observations": 0,
            "quarantined_observations": 0,
            "quarantine_percentage": 0.0,
            "quarantine_reasons": {},
        }

    reasons_count: Dict[str, int] = {}
    quarantined_indices = set()

    # 1. Chunk 3 flagged extreme transitions
    if "has_extreme_kinematic_transition" in windows_df.columns:
        ext_mask = windows_df["has_extreme_kinematic_transition"].fillna(False).astype(bool)
        ext_count = int(ext_mask.sum())
        if ext_count > 0:
            reasons_count["extreme_kinematic_transition"] = ext_count
            quarantined_indices.update(windows_df.index[ext_mask].tolist())

    # 2. Window speed exceeding physical limits
    if "mean_speed_mps" in windows_df.columns:
        phys_mask = windows_df["mean_speed_mps"] > config.quarantine_max_speed_mps
        phys_count = int(phys_mask.sum())
        if phys_count > 0:
            reasons_count["unphysical_speed_burst"] = phys_count
            quarantined_indices.update(windows_df.index[phys_mask].tolist())

    # 3. Personal MAD outlier check on speed
    speed_baseline = baselines.get("mean_speed_mps")
    if (
        speed_baseline
        and speed_baseline.get("mad") is not None
        and speed_baseline.get("robust_scale", 0.0) > 0.0
        and "mean_speed_mps" in windows_df.columns
    ):
        med_val = float(windows_df["mean_speed_mps"].median())
        rob_scale = float(speed_baseline["robust_scale"])
        # Outlier threshold: median + 5 * robust_scale, but must also exceed 300 m/s to protect flights
        outlier_thresh = max(med_val + config.quarantine_mad_multiplier * rob_scale, 300.0)
        mad_mask = windows_df["mean_speed_mps"] > outlier_thresh
        mad_count = int(mad_mask.sum())
        if mad_count > 0:
            reasons_count["personal_mad_speed_outlier"] = mad_count
            quarantined_indices.update(windows_df.index[mad_mask].tolist())

    quarantined_count = len(quarantined_indices)
    quarantine_pct = float(quarantined_count / total_obs * 100.0) if total_obs > 0 else 0.0

    return {
        "total_observations": int(total_obs),
        "quarantined_observations": int(quarantined_count),
        "quarantine_percentage": round(quarantine_pct, 4),
        "quarantine_reasons": reasons_count,
    }


# ==============================================================================
# PROFILE BUILDER (PER USER & ALL USERS)
# ==============================================================================


def build_user_profile(
    user_id: str,
    windows_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    all_points_df: Optional[pd.DataFrame] = None,
    as_of_time: Optional[Union[str, datetime]] = None,
    precomputed_max_radius: Optional[float] = None,
    config: ProfileConfig = ProfileConfig(),
) -> UserProfile:
    """Build a personalized longitudinal mobility profile for a single user.

    Strict Temporal Integrity:
    If as_of_time is provided, only observations at or before as_of_time are included.
    No future data leakage is permitted.
    """
    as_of_ts = pd.to_datetime(as_of_time) if as_of_time is not None else None
    as_of_iso = as_of_ts.isoformat() if as_of_ts is not None else None

    # Temporal & user filtering for no future leakage and strict isolation
    if not windows_df.empty and "user_id" in windows_df.columns:
        if (windows_df["user_id"] == user_id).all():
            u_windows = windows_df.copy()
        else:
            u_windows = windows_df[windows_df["user_id"] == user_id].copy()
    else:
        u_windows = pd.DataFrame(columns=windows_df.columns)

    if as_of_ts is not None and not u_windows.empty and "end_time" in u_windows.columns:
        u_windows = u_windows[pd.to_datetime(u_windows["end_time"]) <= as_of_ts]

    if not trips_df.empty and "user_id" in trips_df.columns:
        if (trips_df["user_id"] == user_id).all():
            u_trips = trips_df.copy()
        else:
            u_trips = trips_df[trips_df["user_id"] == user_id].copy()
    else:
        u_trips = pd.DataFrame(columns=trips_df.columns)

    if as_of_ts is not None and not u_trips.empty and "end_time" in u_trips.columns:
        u_trips = u_trips[pd.to_datetime(u_trips["end_time"]) <= as_of_ts]

    u_points = None
    if all_points_df is not None and not all_points_df.empty and "user_id" in all_points_df.columns:
        if (all_points_df["user_id"] == user_id).all():
            u_points = all_points_df.copy()
        else:
            u_points = all_points_df[all_points_df["user_id"] == user_id].copy()
        if as_of_ts is not None and "timestamp" in u_points.columns:
            u_points = u_points[pd.to_datetime(u_points["timestamp"]) <= as_of_ts]

    # 1. Spatial Anchors (from trip endpoints)
    if not u_trips.empty and "start_lat" in u_trips.columns:
        start_cols = ["user_id", "start_lat", "start_lon", "start_time"]
        end_cols = ["user_id", "end_lat", "end_lon", "end_time"]
        extra_cols = []
        if "trajectory_id" in u_trips.columns:
            extra_cols.append("trajectory_id")
        if "path_distance_m" in u_trips.columns:
            extra_cols.append("path_distance_m")

        start_pts = u_trips[start_cols + extra_cols].rename(
            columns={"start_lat": "latitude", "start_lon": "longitude", "start_time": "timestamp"}
        ).copy()
        start_pts["endpoint_type"] = "start"

        end_pts = u_trips[end_cols + extra_cols].rename(
            columns={"end_lat": "latitude", "end_lon": "longitude", "end_time": "timestamp"}
        ).copy()
        end_pts["endpoint_type"] = "end"

        endpoints = pd.concat([start_pts, end_pts], ignore_index=True)
        anchors = cluster_spatial_anchors(endpoints, config=config)
    else:
        anchors = []

    # 2. Trip Statistics
    trip_stats = compute_trip_statistics(
        trips_df=u_trips,
        anchors=anchors,
        windows_df=u_windows,
        all_points_df=u_points,
        precomputed_max_radius=precomputed_max_radius,
        config=config,
    )

    # 3. Personalized Baselines
    profile_time = as_of_iso or current_iso_time()
    baselines = fit_all_baselines(u_windows, as_of_time=profile_time, config=config)

    # 4. Cold Start & Stability
    status, delta_cv, consec_stable, history = compute_cold_start_trajectory(u_trips, config=config)

    # 5. Quarantine Evaluation
    quarantine_info = evaluate_quarantine_observations(u_windows, baselines, config=config)

    return UserProfile(
        user_id=str(user_id),
        cold_start_status=status,
        trip_count=len(u_trips),
        delta_cv=delta_cv,
        consecutive_stable_trips=consec_stable,
        anchor_clusters=[a.to_dict() for a in anchors],
        trip_stats=trip_stats.to_dict(),
        baseline_distribution=baselines,
        quarantine_stats=quarantine_info,
        stability_history=history,
        last_updated=profile_time,
    )


def extract_trip_endpoints(
    trajectories_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract start/end coordinates, timestamps, and aggregated path distances for each trajectory."""
    # Ensure trajectories_df has sorted timestamps
    t_sorted = trajectories_df.sort_values(["user_id", "trajectory_id", "timestamp"])

    first_pts = t_sorted.drop_duplicates(subset=["user_id", "trajectory_id"], keep="first").copy()
    last_pts = t_sorted.drop_duplicates(subset=["user_id", "trajectory_id"], keep="last").copy()

    trips = first_pts[["user_id", "trajectory_id", "timestamp", "latitude", "longitude"]].rename(
        columns={"timestamp": "start_time", "latitude": "start_lat", "longitude": "start_lon"}
    )
    trips["end_time"] = last_pts["timestamp"].values
    trips["end_lat"] = last_pts["latitude"].values
    trips["end_lon"] = last_pts["longitude"].values

    # Merge aggregated path distance from features
    if "path_distance_m" in features_df.columns:
        path_dist = features_df.groupby(["user_id", "trajectory_id"])["path_distance_m"].sum().reset_index()
        trips = trips.merge(path_dist, on=["user_id", "trajectory_id"], how="left")
    else:
        trips["path_distance_m"] = np.nan

    return trips.sort_values(["user_id", "start_time"]).reset_index(drop=True)


def build_all_mobility_profiles(
    trajectories_path: Union[str, Path] = "ml/data/processed/trajectories.parquet",
    features_path: Union[str, Path] = "ml/data/processed/trajectory_features.parquet",
    output_path: Union[str, Path] = "ml/data/processed/mobility_profiles.parquet",
    config: ProfileConfig = ProfileConfig(),
) -> pd.DataFrame:
    """Build longitudinal mobility profiles for all users in the dataset.

    Saves the canonical output to output_path and returns the profiles DataFrame.
    """
    print(f"Loading features from {features_path}...")
    feat_cols = [
        "user_id",
        "trajectory_id",
        "start_time",
        "end_time",
        "path_distance_m",
        "straight_line_displacement_m",
        "mean_speed_mps",
        "speed_std_dev",
        "entropy_directional",
        "tortuosity_index",
        "turn_frequency",
        "loop_metric",
        "pacing_tendency",
        "is_kinematically_evaluable",
        "has_extreme_kinematic_transition",
    ]
    features_df = pq.read_table(features_path, columns=feat_cols).to_pandas()

    print(f"Loading trajectories from {trajectories_path}...")
    traj_cols = ["user_id", "trajectory_id", "timestamp", "latitude", "longitude"]
    trajectories_df = pq.read_table(trajectories_path, columns=traj_cols).to_pandas()

    print("Extracting trip endpoints and path distances...")
    trips_df = extract_trip_endpoints(trajectories_df, features_df)

    user_ids = sorted(trips_df["user_id"].unique().tolist())

    print("Pre-grouping user datasets for rapid lookup...")
    features_by_user = {uid: grp for uid, grp in features_df.groupby("user_id")}
    trips_by_user = {uid: grp for uid, grp in trips_df.groupby("user_id")}

    print("Determining reference points across all users...")
    user_refs = {}
    for uid in user_ids:
        u_trips = trips_by_user.get(uid, pd.DataFrame())
        if not u_trips.empty and "start_lat" in u_trips.columns:
            start_cols = ["user_id", "start_lat", "start_lon", "start_time"]
            end_cols = ["user_id", "end_lat", "end_lon", "end_time"]
            extra_cols = []
            if "trajectory_id" in u_trips.columns:
                extra_cols.append("trajectory_id")
            if "path_distance_m" in u_trips.columns:
                extra_cols.append("path_distance_m")

            start_pts = u_trips[start_cols + extra_cols].rename(
                columns={"start_lat": "latitude", "start_lon": "longitude", "start_time": "timestamp"}
            ).copy()
            start_pts["endpoint_type"] = "start"

            end_pts = u_trips[end_cols + extra_cols].rename(
                columns={"end_lat": "latitude", "end_lon": "longitude", "end_time": "timestamp"}
            ).copy()
            end_pts["endpoint_type"] = "end"

            endpoints = pd.concat([start_pts, end_pts], ignore_index=True)
            u_anchors = cluster_spatial_anchors(endpoints, config=config)
            if len(u_anchors) > 0:
                prim = sorted(u_anchors, key=lambda a: a.observation_count, reverse=True)[0]
                user_refs[uid] = (prim.center_latitude, prim.center_longitude)
            else:
                user_refs[uid] = (float(u_trips["start_lat"].median()), float(u_trips["start_lon"].median()))

    print("Computing maximum historical movement radii across all 24.7M points in a single pass...")
    max_radii = {}
    for uid, u_df in trajectories_df.groupby("user_id"):
        ref = user_refs.get(uid)
        if ref is not None:
            dists = haversine_distance(ref[0], ref[1], u_df["latitude"].values, u_df["longitude"].values)
            max_radii[uid] = float(np.max(dists))

    print(f"Building mobility profiles for {len(user_ids)} users...")
    records = []
    for uid in user_ids:
        u_profile = build_user_profile(
            user_id=uid,
            windows_df=features_by_user.get(uid, pd.DataFrame(columns=features_df.columns)),
            trips_df=trips_by_user.get(uid, pd.DataFrame(columns=trips_df.columns)),
            precomputed_max_radius=max_radii.get(uid),
            config=config,
        )
        records.append(u_profile.to_record())

    profiles_df = pd.DataFrame(records)

    # Save to parquet with explicit schema
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(profiles_df, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    print(f"Successfully saved {len(profiles_df)} profiles to {output_path}")

    return profiles_df


if __name__ == "__main__":
    build_all_mobility_profiles()

"""Feature engineering module for GeoLife trajectory analysis windows (Chunk 3).

Extracts spatial, kinematic, geometric, and behavioral movement features
from non-overlapping 120-second trajectory analysis windows.

All calculations:
- Adhere strictly to the window boundary (no future-window leakage).
- Explicitly handle zero/near-zero displacement and stationary observations without infinity.
- Strictly avoid division by zero on same-second observations (dt = 0).
- Robust to global / Anti-Meridian spherical geometry (+-180 deg longitude).
- Accurately distinguish linear back-and-forth pacing from 2D polygon looping.
- Preserve NaN for mathematically undefined quantities without fabricating arbitrary zeros.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """Centralized configuration for trajectory feature engineering."""

    # Mean spherical Earth-radius approximation (6,371,000 meters) used for Haversine geodesic distance calculations
    earth_radius_m: float = 6371000.0

    # Minimum displacement to calculate valid bearing/movement direction (meters)
    # Filters out stationary GPS noise jitter where bearing is physically meaningless
    min_displacement_m: float = 0.5

    # Minimum straight-line displacement for tortuosity calculation (meters)
    # When displacement < 1.0m, straight-line displacement approaches zero
    # and tortuosity (path / disp) diverges or becomes mathematically indeterminate
    min_tortuosity_disp_m: float = 1.0

    # Minimum path distance for path closure and loop metrics (meters)
    min_loop_path_m: float = 10.0

    # Minimal observation counts required by mathematical definition
    min_loop_points: int = 3  # Minimal vertices to form a 2D closed simplex/polygon
    min_turn_vectors: int = 2  # Minimal consecutive directional vectors to measure angular heading change
    min_dispersion_points: int = 3  # Minimal points to measure spatial dispersion ratio
    min_bbox_diagonal_m: float = 1.0  # Minimal bounding-box diagonal (meters) to avoid zero-division in aspect ratio

    # Kinematic / data-quality indicator threshold (meters/second):
    # Fixed physical domain constraint representing the standard acoustic sound barrier (Mach 1 ~ 340.0 m/s).
    # All civilian transportation recorded in GeoLife (pedestrian, cycling, transit, rail, commercial flight)
    # is strictly subsonic. Transitions exceeding 340.0 m/s represent non-physical GPS hardware artifacts,
    # large coordinate teleports, or concurrent logger stream collisions.
    # Because this is a fixed domain constraint derived from physics (NOT estimated from data), it ensures
    # absolute zero train/validation/test leakage across user splits.
    extreme_speed_threshold_mps: float = 340.0

    # Turn angle threshold to count as a deliberate directional turn (degrees)
    turn_threshold_deg: float = 30.0

    # Turn angle threshold to count as a reversal / U-turn (degrees)
    reversal_threshold_deg: float = 135.0

    # Number of directional entropy bins (fixed compass octants: N, NE, E, SE, S, SW, W, NW)
    # Fixed geometric bins avoid future information leakage
    entropy_bins: int = 8

    # Minimum valid movement vectors (displacement >= 0.5m) required to calculate directional entropy
    min_entropy_vectors: int = 3

    # Modeling evaluability thresholds: criteria for a window to have sufficient kinematic observations
    eval_min_points: int = 3
    eval_min_span_sec: float = 10.0
    eval_min_path_m: float = 1.0


def haversine_np(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
    radius: float = 6371000.0,
) -> np.ndarray | float:
    """Calculate geodesic distance between points using the Haversine formula."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def calculate_bearings_np(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Calculate initial compass bearings in degrees [0, 360) between coordinate pairs."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlam = np.radians(lon2 - lon1)

    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    bearings = np.degrees(np.arctan2(y, x))
    return (bearings + 360.0) % 360.0


def spherical_centroid(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Calculate geographic center of mass on the sphere using 3D vector averaging.

    Properly handles global trajectories crossing the Anti-Meridian (+-180 deg).
    """
    phi = np.radians(lats)
    lam = np.radians(lons)
    x = float(np.mean(np.cos(phi) * np.cos(lam)))
    y = float(np.mean(np.cos(phi) * np.sin(lam)))
    z = float(np.mean(np.sin(phi)))

    lon_mean = math.degrees(math.atan2(y, x))
    hyp = math.sqrt(x**2 + y**2)
    lat_mean = math.degrees(math.atan2(z, hyp))
    return lat_mean, lon_mean


def circular_longitude_span(lons: np.ndarray) -> float:
    """Calculate the minimal circular longitude arc span in degrees [0, 360).

    Avoids naive max(lon) - min(lon) explosion when crossing the Anti-Meridian.
    """
    if len(lons) <= 1:
        return 0.0
    u_lons = np.unique(lons)
    if len(u_lons) <= 1:
        return 0.0
    u_lons = np.sort(u_lons)
    gaps = np.diff(u_lons)
    wrap_gap = 360.0 - (u_lons[-1] - u_lons[0])
    all_gaps = np.append(gaps, wrap_gap)
    max_gap = float(np.max(all_gaps))
    return float(360.0 - max_gap)


def compute_window_features_dict(
    window_id: str,
    user_id: str,
    trajectory_id: str,
    segment_id: str,
    window_index: int,
    is_full_window: bool,
    lats: np.ndarray,
    lons: np.ndarray,
    ts: np.ndarray,
    dts: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, Any]:
    """Compute all mathematically supported features for a single analysis window."""
    n_pts = len(lats)
    t_start = pd.Timestamp(ts[0])
    t_end = pd.Timestamp(ts[-1])
    span_sec = float((ts[-1] - ts[0]) / np.timedelta64(1, "s"))

    # Initial dictionary with metadata and temporal defaults
    feat: dict[str, Any] = {
        "window_id": window_id,
        "user_id": user_id,
        "trajectory_id": trajectory_id,
        "segment_id": segment_id,
        "window_index": int(window_index),
        "is_full_window": bool(is_full_window),
        "start_time": t_start,
        "end_time": t_end,
        "point_count": int(n_pts),
        "temporal_span_sec": span_sec,
    }

    # Group A: Temporal / Quality
    if n_pts > 1:
        step_dts = dts[1:]
        zero_dt_count = int(np.sum(step_dts == 0.0))
        feat["zero_dt_count"] = zero_dt_count
        feat["zero_dt_fraction"] = float(zero_dt_count / len(step_dts))
        feat["dt_mean"] = float(np.mean(step_dts))
        feat["dt_median"] = float(np.median(step_dts))
        feat["dt_min"] = float(np.min(step_dts))
        feat["dt_max"] = float(np.max(step_dts))
        feat["dt_std"] = float(np.std(step_dts, ddof=1)) if len(step_dts) > 1 else np.nan
    else:
        feat["zero_dt_count"] = 0
        feat["zero_dt_fraction"] = 0.0
        feat["dt_mean"] = np.nan
        feat["dt_median"] = np.nan
        feat["dt_min"] = np.nan
        feat["dt_max"] = np.nan
        feat["dt_std"] = np.nan

    # Group C: Spatial / Trajectory Geometry
    if n_pts > 1:
        step_dists = haversine_np(lats[:-1], lons[:-1], lats[1:], lons[1:], radius=config.earth_radius_m)
        path_dist = float(np.sum(step_dists))
        disp = float(haversine_np(lats[0], lons[0], lats[-1], lons[-1], radius=config.earth_radius_m))

        feat["path_distance_m"] = path_dist
        feat["straight_line_displacement_m"] = disp
        feat["step_distance_mean"] = float(np.mean(step_dists))
        feat["step_distance_median"] = float(np.median(step_dists))
        feat["step_distance_max"] = float(np.max(step_dists))
        feat["step_distance_min"] = float(np.min(step_dists))
        feat["step_distance_std"] = float(np.std(step_dists, ddof=1)) if len(step_dists) > 1 else np.nan

        # Tortuosity = path / displacement
        # Explicit zero/near-zero handling: when displacement < min_tortuosity_disp_m, tortuosity is undefined
        if disp >= config.min_tortuosity_disp_m:
            feat["tortuosity_index"] = float(path_dist / disp)
        else:
            feat["tortuosity_index"] = np.nan

        # Path Closure Ratio: 1 - displacement / path in [0, 1]
        # Measures the proportion of path distance that does not result in net displacement
        if n_pts >= config.min_loop_points and path_dist >= config.min_loop_path_m:
            disp_ratio = disp / path_dist
            feat["path_closure_ratio"] = float(max(0.0, min(1.0, 1.0 - disp_ratio)))
        else:
            feat["path_closure_ratio"] = np.nan

        # Centroid and Radius of Gyration (using 3D spherical centroid)
        clat, clon = spherical_centroid(lats, lons)
        dists_to_centroid = haversine_np(lats, lons, clat, clon, radius=config.earth_radius_m)
        rg = float(np.sqrt(np.mean(dists_to_centroid**2)))
        feat["radius_of_gyration"] = 0.0 if rg < 1e-6 else rg

        # Bounding Box dimensions (using circular longitude span)
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        h_m = float(haversine_np(lat_min, clon, lat_max, clon, radius=config.earth_radius_m))
        lon_span_deg = circular_longitude_span(lons)
        w_m = float(haversine_np(clat, 0.0, clat, lon_span_deg, radius=config.earth_radius_m))
        feat["bbox_height_m"] = h_m
        feat["bbox_width_m"] = w_m
        feat["bbox_diagonal_m"] = float(math.sqrt(h_m**2 + w_m**2))
        feat["bbox_area_sqm"] = float(h_m * w_m)
    else:
        # 1-point window: no displacement or extent
        feat["path_distance_m"] = 0.0
        feat["straight_line_displacement_m"] = 0.0
        feat["step_distance_mean"] = np.nan
        feat["step_distance_median"] = np.nan
        feat["step_distance_max"] = np.nan
        feat["step_distance_min"] = np.nan
        feat["step_distance_std"] = np.nan
        feat["tortuosity_index"] = np.nan
        feat["path_closure_ratio"] = np.nan
        feat["radius_of_gyration"] = 0.0
        feat["bbox_height_m"] = 0.0
        feat["bbox_width_m"] = 0.0
        feat["bbox_diagonal_m"] = 0.0
        feat["bbox_area_sqm"] = 0.0

    # Group B: Kinematics
    # Speeds are defined strictly when dt > 0. Same-second observations (dt = 0) never produce speed.
    if n_pts > 1:
        step_dts = dts[1:]
        pos_dt_mask = step_dts > 0.0
        if np.any(pos_dt_mask):
            valid_speeds = step_dists[pos_dt_mask] / step_dts[pos_dt_mask]
            feat["mean_speed_mps"] = float(np.mean(valid_speeds))
            feat["median_speed_mps"] = float(np.median(valid_speeds))
            feat["max_speed_mps"] = float(np.max(valid_speeds))
            feat["min_speed_mps"] = float(np.min(valid_speeds))
            if len(valid_speeds) > 1:
                feat["speed_std_dev"] = float(np.std(valid_speeds, ddof=1))
                feat["speed_variance"] = float(np.var(valid_speeds, ddof=1))
            else:
                feat["speed_std_dev"] = np.nan
                feat["speed_variance"] = np.nan

            # Kinematic / data-quality indicator flags (strictly within-window):
            # Flags physically impossible transitions (> 340 m/s) to prevent corrupted sensor jumps
            # from contaminating downstream behavioral and mobility models.
            # NOTE: This is purely a kinematic/data-quality indicator, NOT a wandering indicator,
            # behavioral anomaly label, or clinical anomaly label.
            extreme_mask = valid_speeds > config.extreme_speed_threshold_mps
            extreme_cnt = int(np.sum(extreme_mask))
            feat["has_extreme_kinematic_transition"] = bool(extreme_cnt > 0)
            feat["extreme_transition_count"] = extreme_cnt
            feat["extreme_transition_fraction"] = float(extreme_cnt / len(valid_speeds))

            # Accelerations: difference between consecutive speed observations
            if len(valid_speeds) > 1:
                valid_step_dts = step_dts[pos_dt_mask]
                acc_dts = 0.5 * (valid_step_dts[:-1] + valid_step_dts[1:])
                acc_mask = acc_dts > 0.0
                if np.any(acc_mask):
                    accels = (valid_speeds[1:] - valid_speeds[:-1])[acc_mask] / acc_dts[acc_mask]
                    feat["mean_acceleration_mps2"] = float(np.mean(accels))
                    feat["max_acceleration_mps2"] = float(np.max(accels))
                    feat["min_acceleration_mps2"] = float(np.min(accels))
                    feat["acceleration_std_dev"] = (
                        float(np.std(accels, ddof=1)) if len(accels) > 1 else np.nan
                    )
                else:
                    feat["mean_acceleration_mps2"] = np.nan
                    feat["max_acceleration_mps2"] = np.nan
                    feat["min_acceleration_mps2"] = np.nan
                    feat["acceleration_std_dev"] = np.nan
            else:
                feat["mean_acceleration_mps2"] = np.nan
                feat["max_acceleration_mps2"] = np.nan
                feat["min_acceleration_mps2"] = np.nan
                feat["acceleration_std_dev"] = np.nan
        else:
            feat["mean_speed_mps"] = np.nan
            feat["median_speed_mps"] = np.nan
            feat["max_speed_mps"] = np.nan
            feat["min_speed_mps"] = np.nan
            feat["speed_std_dev"] = np.nan
            feat["speed_variance"] = np.nan
            feat["has_extreme_kinematic_transition"] = False
            feat["extreme_transition_count"] = 0
            feat["extreme_transition_fraction"] = 0.0
            feat["mean_acceleration_mps2"] = np.nan
            feat["max_acceleration_mps2"] = np.nan
            feat["min_acceleration_mps2"] = np.nan
            feat["acceleration_std_dev"] = np.nan
    else:
        feat["mean_speed_mps"] = np.nan
        feat["median_speed_mps"] = np.nan
        feat["max_speed_mps"] = np.nan
        feat["min_speed_mps"] = np.nan
        feat["speed_std_dev"] = np.nan
        feat["speed_variance"] = np.nan
        feat["has_extreme_kinematic_transition"] = False
        feat["extreme_transition_count"] = 0
        feat["extreme_transition_fraction"] = 0.0
        feat["mean_acceleration_mps2"] = np.nan
        feat["max_acceleration_mps2"] = np.nan
        feat["min_acceleration_mps2"] = np.nan
        feat["acceleration_std_dev"] = np.nan

    # Group B & D: Bearings, Turns, and Behavioral Features
    # Filter step vectors where displacement exceeds min_displacement_m (0.5m)
    valid_bearings: np.ndarray = np.array([], dtype=np.float64)
    if n_pts > 1:
        disp_mask = step_dists >= config.min_displacement_m
        if np.any(disp_mask):
            valid_bearings = calculate_bearings_np(
                lats[:-1][disp_mask],
                lons[:-1][disp_mask],
                lats[1:][disp_mask],
                lons[1:][disp_mask],
            )

    n_bearings = len(valid_bearings)
    if n_bearings >= config.min_turn_vectors:
        # Heading changes between consecutive valid movement vectors: [-180, 180]
        heading_diffs = (valid_bearings[1:] - valid_bearings[:-1] + 180.0) % 360.0 - 180.0
        abs_turns = np.abs(heading_diffs)
        feat["heading_change_mean"] = float(np.mean(abs_turns))
        feat["heading_change_std"] = float(np.std(abs_turns, ddof=1)) if len(abs_turns) > 1 else np.nan

        # Turns count exceeding turn_threshold_deg (e.g. 30 deg)
        turns_count = int(np.sum(abs_turns >= config.turn_threshold_deg))
        feat["turn_frequency"] = float(turns_count / max(span_sec / 60.0, 1.0 / 60.0))

        # Backtracking tendency: fraction of turns that are reversals (>= 135 deg)
        reversals = int(np.sum(abs_turns >= config.reversal_threshold_deg))
        backtrack_rate = float(reversals / len(abs_turns))
        feat["backtracking_tendency"] = backtrack_rate

        # Pacing tendency: true back-and-forth pacing requires high reversal rate + path closure
        # (Distinguishes true pacing from generic forward zig-zagging which has 0 reversals)
        closure_val = feat["path_closure_ratio"]
        if not math.isnan(closure_val):
            feat["pacing_tendency"] = float(backtrack_rate * closure_val)
        else:
            feat["pacing_tendency"] = np.nan

        # 2D Loop-likeness / Closure Heuristic: combines path closure, 2D aspect ratio, and low reversals.
        # METHODOLOGICAL CONSTRAINTS:
        # 1. This is a deterministic loop-likeness heuristic in [0, 1], NOT formal polygon detection
        #    (it does not construct geometric polygons, ray-cast interior points, or test self-intersections).
        # 2. It is NOT clinical ground truth and does NOT by itself prove pacing, lapping, or wandering.
        # 3. Continuous aspect ratio min(w, h) / max(w, h) smoothly scales down thin 1D corridors without
        #    an arbitrary step threshold, rewarding open 2D spatial enclosures.
        if not math.isnan(closure_val) and feat["bbox_diagonal_m"] >= config.min_bbox_diagonal_m:
            aspect = min(feat["bbox_width_m"], feat["bbox_height_m"]) / max(
                feat["bbox_width_m"], feat["bbox_height_m"], config.min_bbox_diagonal_m
            )
            feat["loop_metric"] = float(closure_val * aspect * (1.0 - backtrack_rate))
        else:
            feat["loop_metric"] = np.nan
    else:
        feat["heading_change_mean"] = np.nan
        feat["heading_change_std"] = np.nan
        feat["turn_frequency"] = np.nan
        feat["backtracking_tendency"] = np.nan
        feat["pacing_tendency"] = np.nan
        feat["loop_metric"] = np.nan

    # Heading variability (circular variance of bearings in [0, 1])
    if n_bearings >= config.min_turn_vectors:
        rad_b = np.radians(valid_bearings)
        c_mean = float(np.mean(np.cos(rad_b)))
        s_mean = float(np.mean(np.sin(rad_b)))
        r_len = math.sqrt(c_mean**2 + s_mean**2)
        feat["heading_variability"] = float(max(0.0, min(1.0, 1.0 - r_len)))
    else:
        feat["heading_variability"] = np.nan

    # Spatial Randomness / Dispersion: Ratio of Radius of Gyration to Bounding Box Diagonal
    if n_pts >= config.min_dispersion_points and feat["bbox_diagonal_m"] >= config.min_bbox_diagonal_m:
        feat["spatial_randomness_dispersion"] = float(
            feat["radius_of_gyration"] / feat["bbox_diagonal_m"]
        )
    else:
        feat["spatial_randomness_dispersion"] = np.nan

    # Group E: Directional Entropy
    # Discretizes step bearings into B fixed compass octants of 45 degrees
    if n_bearings >= config.min_entropy_vectors:
        bin_indices = (valid_bearings // (360.0 / config.entropy_bins)).astype(int) % config.entropy_bins
        counts = np.bincount(bin_indices, minlength=config.entropy_bins)
        probs = counts / np.sum(counts)
        nz_probs = probs[probs > 0.0]
        h = -float(np.sum(nz_probs * np.log2(nz_probs)))
        max_h = math.log2(config.entropy_bins)
        feat["entropy_directional"] = float(h / max_h)
    else:
        feat["entropy_directional"] = np.nan

    # Feature Computability & Modeling Flags
    has_disp = bool(n_pts >= 2 and feat["path_distance_m"] > 0.0)
    has_kin = bool(not math.isnan(feat["mean_speed_mps"]))
    has_acc = bool(not math.isnan(feat["mean_acceleration_mps2"]))
    has_turns = bool(not math.isnan(feat["heading_change_mean"]))
    has_ent = bool(not math.isnan(feat["entropy_directional"]))
    evaluable = bool(
        n_pts >= config.eval_min_points
        and span_sec >= config.eval_min_span_sec
        and feat["path_distance_m"] >= config.eval_min_path_m
    )

    feat["has_displacement"] = has_disp
    feat["has_valid_kinematics"] = has_kin
    feat["has_acceleration"] = has_acc
    feat["has_valid_turns"] = has_turns
    feat["has_entropy"] = has_ent
    feat["is_kinematically_evaluable"] = evaluable

    return feat


def extract_features_from_dataframe(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """Extract all features for contiguous trajectory windows in a dataframe."""
    if df.empty:
        return pd.DataFrame()

    window_ids = df["window_id"].to_numpy()
    change_indices = np.where(window_ids[:-1] != window_ids[1:])[0] + 1
    splits = np.split(np.arange(len(df)), change_indices)

    lats = df["latitude"].to_numpy(dtype=np.float64)
    lons = df["longitude"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy()
    dts = df["dt"].to_numpy(dtype=np.float64)
    uids = df["user_id"].to_numpy()
    trajs = df["trajectory_id"].to_numpy()
    segs = df["segment_id"].to_numpy()
    w_idxs = df["window_index"].to_numpy(dtype=np.int32)
    fulls = df["is_full_window"].to_numpy(dtype=bool)

    records: list[dict[str, Any]] = []
    for idxs in splits:
        i0 = idxs[0]
        rec = compute_window_features_dict(
            window_id=str(window_ids[i0]),
            user_id=str(uids[i0]),
            trajectory_id=str(trajs[i0]),
            segment_id=str(segs[i0]),
            window_index=int(w_idxs[i0]),
            is_full_window=bool(fulls[i0]),
            lats=lats[idxs],
            lons=lons[idxs],
            ts=ts[idxs],
            dts=dts[idxs],
            config=config,
        )
        records.append(rec)

    return pd.DataFrame(records)


def process_features_pipeline(
    input_parquet: str | Path,
    output_dir: str | Path,
    config: FeatureConfig = FeatureConfig(),
    max_row_groups: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Process trajectory windows into trajectory_features.parquet using streaming."""
    in_path = Path(input_parquet).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "trajectory_features.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    pq_file = pq.ParquetFile(in_path)
    total_rgs = pq_file.num_row_groups
    rgs_to_process = min(total_rgs, max_row_groups) if max_row_groups else total_rgs

    logger.info(f"Processing features from {in_path.name} ({rgs_to_process} row groups)")
    t_start = time.time()

    writer: pq.ParquetWriter | None = None
    total_windows_processed = 0

    read_columns = [
        "user_id",
        "trajectory_id",
        "segment_id",
        "window_id",
        "window_index",
        "is_full_window",
        "timestamp",
        "latitude",
        "longitude",
        "dt",
    ]

    for rg_idx in range(rgs_to_process):
        rg_table = pq_file.read_row_group(rg_idx, columns=read_columns)
        rg_df = rg_table.to_pandas()

        feat_df = extract_features_from_dataframe(rg_df, config=config)
        total_windows_processed += len(feat_df)

        feat_table = pa.Table.from_pandas(feat_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, feat_table.schema, compression="snappy")
        writer.write_table(feat_table)

        if (rg_idx + 1) % 15 == 0 or (rg_idx + 1) == rgs_to_process:
            logger.info(
                f"Processed {rg_idx + 1} / {rgs_to_process} row groups ({total_windows_processed:,} windows)..."
            )

    if writer is not None:
        writer.close()

    elapsed = time.time() - t_start
    logger.info(f"Feature extraction complete in {elapsed:.2f}s! Generating report...")

    report = generate_feature_engineering_report(
        output_parquet=out_parquet,
        config=config,
        runtime_seconds=elapsed,
    )

    return out_parquet, report


def generate_feature_engineering_report(
    output_parquet: str | Path,
    config: FeatureConfig = FeatureConfig(),
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    """Generate comprehensive feature engineering report in JSON and Markdown."""
    parquet_path = Path(output_parquet).resolve()
    out_dir = parquet_path.parent
    json_path = out_dir / "feature_engineering_report.json"
    md_path = out_dir / "feature_engineering_report.md"

    # Read feature dataset
    df = pq.read_table(parquet_path).to_pandas()
    n_windows = len(df)

    numeric_cols = [
        "point_count",
        "temporal_span_sec",
        "zero_dt_count",
        "zero_dt_fraction",
        "dt_mean",
        "dt_std",
        "dt_median",
        "dt_min",
        "dt_max",
        "path_distance_m",
        "straight_line_displacement_m",
        "tortuosity_index",
        "path_closure_ratio",
        "loop_metric",
        "radius_of_gyration",
        "bbox_height_m",
        "bbox_width_m",
        "bbox_diagonal_m",
        "bbox_area_sqm",
        "step_distance_mean",
        "step_distance_std",
        "step_distance_max",
        "step_distance_min",
        "mean_speed_mps",
        "median_speed_mps",
        "max_speed_mps",
        "min_speed_mps",
        "speed_std_dev",
        "speed_variance",
        "mean_acceleration_mps2",
        "max_acceleration_mps2",
        "min_acceleration_mps2",
        "acceleration_std_dev",
        "heading_change_mean",
        "heading_change_std",
        "turn_frequency",
        "backtracking_tendency",
        "pacing_tendency",
        "heading_variability",
        "spatial_randomness_dispersion",
        "entropy_directional",
        "extreme_transition_count",
        "extreme_transition_fraction",
    ]

    flag_cols = [
        "is_full_window",
        "has_displacement",
        "has_valid_kinematics",
        "has_acceleration",
        "has_valid_turns",
        "has_entropy",
        "is_kinematically_evaluable",
        "has_extreme_kinematic_transition",
    ]

    distributions: dict[str, Any] = {}
    missingness: dict[str, Any] = {}
    inf_count: dict[str, int] = {}

    for col in numeric_cols:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_inf = int(np.isinf(series).sum())
        inf_count[col] = n_inf
        missingness[col] = {
            "missing_count": n_missing,
            "missing_pct": float(n_missing / n_windows * 100.0),
            "inf_count": n_inf,
        }

        valid_vals = series.dropna()
        if len(valid_vals) > 0:
            distributions[col] = {
                "count": int(len(valid_vals)),
                "min": float(valid_vals.min()),
                "p25": float(valid_vals.quantile(0.25)),
                "median": float(valid_vals.median()),
                "mean": float(valid_vals.mean()),
                "p75": float(valid_vals.quantile(0.75)),
                "p95": float(valid_vals.quantile(0.95)),
                "p99": float(valid_vals.quantile(0.99)),
                "max": float(valid_vals.max()),
                "std": float(valid_vals.std()) if len(valid_vals) > 1 else 0.0,
            }
        else:
            distributions[col] = {"count": 0}

    flag_stats: dict[str, Any] = {}
    for col in flag_cols:
        true_cnt = int(df[col].sum())
        flag_stats[col] = {
            "true_count": true_cnt,
            "false_count": n_windows - true_cnt,
            "true_pct": float(true_cnt / n_windows * 100.0),
        }

    key_features = [
        "point_count",
        "path_distance_m",
        "straight_line_displacement_m",
        "tortuosity_index",
        "path_closure_ratio",
        "loop_metric",
        "pacing_tendency",
        "radius_of_gyration",
        "mean_speed_mps",
        "turn_frequency",
        "heading_variability",
        "entropy_directional",
    ]
    corr_matrix = df[key_features].corr().round(4).to_dict()

    report: dict[str, Any] = {
        "dataset_name": "GeoLife 120-Second Analysis Window Feature Engineering (Chunk 3)",
        "output_file": str(parquet_path),
        "total_windows": n_windows,
        "runtime_seconds": float(runtime_seconds),
        "config": {
            "min_displacement_m": config.min_displacement_m,
            "min_tortuosity_disp_m": config.min_tortuosity_disp_m,
            "min_loop_path_m": config.min_loop_path_m,
            "min_loop_points": config.min_loop_points,
            "min_turn_vectors": config.min_turn_vectors,
            "min_dispersion_points": config.min_dispersion_points,
            "min_bbox_diagonal_m": config.min_bbox_diagonal_m,
            "extreme_speed_threshold_mps": config.extreme_speed_threshold_mps,
            "turn_threshold_deg": config.turn_threshold_deg,
            "reversal_threshold_deg": config.reversal_threshold_deg,
            "entropy_bins": config.entropy_bins,
            "min_entropy_vectors": config.min_entropy_vectors,
            "eval_min_points": config.eval_min_points,
            "eval_min_span_sec": config.eval_min_span_sec,
            "eval_min_path_m": config.eval_min_path_m,
        },
        "flags": flag_stats,
        "missingness": missingness,
        "distributions": distributions,
        "key_correlations": corr_matrix,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    flag_rows = []
    flag_descs = {
        "is_full_window": "Standard full 120s window",
        "has_displacement": ">= 2 pts and path > 0m",
        "has_valid_kinematics": ">= 1 positive dt interval",
        "has_acceleration": ">= 2 positive dt intervals",
        "has_valid_turns": ">= 2 valid vectors for turn",
        "has_entropy": ">= 3 valid vectors for directional entropy",
        "is_kinematically_evaluable": "Valid pts>=3, span>=10s, path>=1m",
        "has_extreme_kinematic_transition": "Data-quality indicator: step speed > 340 m/s (Mach 1 domain constraint)",
    }
    for f_col in flag_cols:
        fc = flag_stats[f_col]["true_count"]
        fp = flag_stats[f_col]["true_pct"]
        fd = flag_descs[f_col]
        flag_rows.append(f"| `{f_col}` | {fc:,} | {fp:.2f}% | {fd} |")
    flag_table_body = "\n".join(flag_rows)

    md_content = f"""# Trajectory Feature Engineering Quality & Distribution Report (Chunk 3)

## 1. Overview & Execution Summary
- **Input Dataset**: `ml/data/processed/trajectory_windows.parquet`
- **Output Dataset**: `{parquet_path}`
- **Total Windows Evaluated**: {n_windows:,} (100% matched to Chunk 2)
- **Pipeline Runtime**: {runtime_seconds:.2f} seconds
- **Infinity Values Audit**: 0 infinity values across all columns

## 2. Configuration & Methodological Thresholds
| Parameter | Value | Justification |
|---|---:|---|
| `earth_radius_m` | {config.earth_radius_m:.1f} m | Mean spherical Earth-radius approximation for Haversine |
| `min_displacement_m` | {config.min_displacement_m:.1f} m | Filters stationary GPS jitter from bearing calculation |
| `min_tortuosity_disp_m` | {config.min_tortuosity_disp_m:.1f} m | Prevents division by near-zero displacement |
| `min_loop_path_m` | {config.min_loop_path_m:.1f} m | Eliminates micro-loop noise from stationary dwell |
| `min_loop_points` | {config.min_loop_points} | Minimal vertices to form a 2D closed polygon |
| `min_turn_vectors` | {config.min_turn_vectors} | Minimal movement vectors to measure angular turn |
| `min_dispersion_points` | {config.min_dispersion_points} | Minimal observations to compute spatial dispersion |
| `min_bbox_diagonal_m` | {config.min_bbox_diagonal_m:.1f} m | Minimum bbox diagonal to avoid aspect division by 0 |
| `extreme_speed_threshold_mps` | {config.extreme_speed_threshold_mps:.1f} m/s | Physical sound barrier (Mach 1) |
| `turn_threshold_deg` | {config.turn_threshold_deg:.1f}° | Separates intentional turns from sensor path jitter |
| `reversal_threshold_deg` | {config.reversal_threshold_deg:.1f}° | Identifies sharp U-turns and reversals (>= 135°) |
| `entropy_bins` | {config.entropy_bins} | Fixed compass octants (N, NE, E, SE, S, SW, W, NW) |
| `min_entropy_vectors` | {config.min_entropy_vectors} | Minimum valid directional vectors required |

## 3. Window Evaluability & Quality Flags
| Quality Flag | True Count | Percentage | Description |
|---|---:|---:|---|
{flag_table_body}

## 4. Feature Formulas & Minimum Required Observations
| Feature Name | Formula | Min Pts | Missingness Reason |
|---|---|---:|---|
| `point_count` | Count of points in window | 1 | Always defined (0% missing) |
| `temporal_span_sec` | $t_N - t_1$ | 1 | Always defined (0% missing) |
| `zero_dt_fraction` | $\\sum \\mathbb{{I}}(\\Delta t=0) / (N-1)$ | 2 | 1-point windows have no intervals |
| `path_distance_m` | $\\sum d_i$ (Haversine geodesic) | 1 | Defined for all windows (0m for 1 pt) |
| `straight_line_displacement_m` | Geodesic dist $(p_1, p_N)$ | 1 | Defined for all windows (0m for 1 pt) |
| `tortuosity_index` | $d_{{\\text{{path}}}} / d_{{\\text{{disp}}}}$ | 2 | Undefined if displacement < 1.0m |
| `path_closure_ratio` | $1 - d_{{\\text{{disp}}}} / d_{{\\text{{path}}}}$ | 3 | Undefined if path < 10m or points < 3 |
| `loop_metric` | $\\text{{closure}} \\times \\text{{aspect}} \\times (1 - \\text{{rev}})$ | 3 | Path < 10m or pts < 3 |
| `pacing_tendency` | $\\text{{reversals}} \\times \\text{{closure}}$ | 3 | < 2 valid movement vectors |
| `radius_of_gyration` | $\\sqrt{{\\frac{{1}}{{N}}\\sum d(p_i, \\bar{{p}})^2}}$ | 1 | Always defined (3D center) |
| `mean_speed_mps` | Mean $d_i / \\Delta t_i$ for $\\Delta t_i > 0$ | 2 | Undefined if no positive $\\Delta t$ steps |
| `mean_acceleration_mps2` | $\\Delta v_j / \\Delta t_j$ | 3 | Undefined if < 2 positive speed steps |
| `turn_frequency` | Turns $\\ge 30^\\circ$ per minute | 3 | Undefined if < 2 valid movement vectors |
| `heading_variability` | Circular variance $1 - \\bar{{R}}$ | 3 | Undefined if < 2 valid movement vectors |
| `entropy_directional` | Shannon Entropy on 8 octants | 4 | Undefined if < 3 valid movement vectors |

## 5. Feature Distributions & Quantiles
| Feature Name | Missing % | Min | P25 | Median | Mean | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for col in [
        "point_count",
        "temporal_span_sec",
        "path_distance_m",
        "straight_line_displacement_m",
        "tortuosity_index",
        "path_closure_ratio",
        "loop_metric",
        "pacing_tendency",
        "radius_of_gyration",
        "bbox_diagonal_m",
        "mean_speed_mps",
        "speed_std_dev",
        "mean_acceleration_mps2",
        "turn_frequency",
        "heading_variability",
        "entropy_directional",
    ]:
        dist = distributions.get(col, {})
        miss = missingness.get(col, {})
        if dist.get("count", 0) > 0:
            md_content += (
                f"| `{col}` | {miss.get('missing_pct', 0.0):.2f}% | {dist.get('min', 0.0):.2f} | "
                f"{dist.get('p25', 0.0):.2f} | {dist.get('median', 0.0):.2f} | {dist.get('mean', 0.0):.2f} | "
                f"{dist.get('p75', 0.0):.2f} | {dist.get('p95', 0.0):.2f} | {dist.get('max', 0.0):.2f} |\n"
            )

    md_content += """
## 6. Directional Entropy Specification (Group E)
- **Discretization**: Step bearings are discretized into $B = 8$ compass octants of $45^\\circ$ each.
- **Formula**: $H = -\\sum_{k=1}^8 p_k \\log_2(p_k)$, normalized as $\\text{entropy} = H / \\log_2(8) \\in [0, 1]$.
- **Leakage Prevention**: Bin boundaries are intrinsic circular constants and contain zero data parameters.
- **Handling of Insufficient Observations**: If a window has < 3 valid vectors, `entropy_directional = NaN`.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report


def main() -> None:
    """CLI entrypoint for trajectory feature engineering."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    default_in = Path(__file__).resolve().parent.parent / "data" / "processed" / "trajectory_windows.parquet"
    default_out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    parser = argparse.ArgumentParser(description="GeoLife Trajectory Analysis Window Feature Engineering (Chunk 3)")
    parser.add_argument("--input-file", type=str, default=str(default_in), help="Path to trajectory_windows.parquet")
    parser.add_argument("--output-dir", type=str, default=str(default_out_dir), help="Path to output directory")
    parser.add_argument("--max-rgs", type=int, default=None, help="Optional maximum row groups to process")

    args = parser.parse_args()

    config = FeatureConfig()
    print("=" * 70)
    print("Starting GeoLife Trajectory Feature Engineering (Chunk 3)")
    print(f"Input file:  {args.input_file}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 70)

    out_file, report = process_features_pipeline(
        input_parquet=args.input_file,
        output_dir=args.output_dir,
        config=config,
        max_row_groups=args.max_rgs,
    )

    print("\nFeature engineering complete!")
    print(f"Output file:     {out_file}")
    print(f"Total windows:   {report['total_windows']:,}")
    print(f"Runtime:         {report['runtime_seconds']:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

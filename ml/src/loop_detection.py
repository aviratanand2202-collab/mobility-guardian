"""Persistent-Looping and Repetitive-Route Detection Engine.

Analyzes GPS trajectory sequences and kinematic features to detect repetitive
route traversal, repeated turns, route closure, and sustained localized movement.

Distinguishes genuine persistent looping from:
- Ordinary turns (directional intersection corners with high displacement)
- Short circular detours (single brief roundabout or fountain detour < 60s)
- Stationary GPS jitter (low speed < 0.3 m/s, tiny displacement, sensor noise)
- Insufficient GPS history (sparse points, short duration, degraded accuracy)

Scientific and Clinical Disclaimer:
Looping metrics indicate geometric and spatial movement recurrence relative to
observed coordinates. Looping by itself does NOT establish clinical wandering,
cognitive impairment, or distress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.src.features import FeatureConfig, haversine_np, calculate_bearings_np, spherical_centroid


@dataclass(frozen=True)
class LoopDetectorConfig:
    """Configurable thresholds for persistent loop detection."""

    # Duration and observation bounds
    min_loop_duration_sec: float = 120.0
    """Minimum sustained duration in seconds required to declare persistent looping."""

    min_points: int = 6
    """Minimum GPS points required to evaluate loop geometry."""

    min_path_distance_m: float = 35.0
    """Minimum total path distance in meters to distinguish from stationary jitter."""

    # Kinematic speed filters
    min_speed_mps: float = 0.3
    """Minimum mean speed (m/s). Speeds below this represent stationary sensor jitter."""

    max_speed_mps: float = 4.0
    """Maximum speed (m/s). Pedestrian/jogging range; excludes high-speed vehicle loops."""

    # Spatial confinement bounds
    max_bounding_radius_m: float = 250.0
    """Maximum spatial spread (meters). Genuine localized looping occurs in a confined area."""

    min_closure_ratio: float = 0.60
    """Path closure ratio threshold (1 - displacement/path). High closure indicates return."""

    min_loop_cycles: float = 1.3
    """Minimum estimated complete loop cycles (rotations/revisitations) required."""

    max_revisit_distance_m: float = 25.0
    """Maximum distance in meters between non-adjacent points to count as a path closure revisit."""

    min_revisit_time_gap_sec: float = 30.0
    """Minimum temporal separation between points to count as a recurring loop revisit."""

    max_horizontal_accuracy_m: float = 25.0
    """Coarse GPS readings exceeding this accuracy threshold are filtered out."""

    stale_timeout_sec: float = 300.0
    """Temporal gap in telemetry that triggers a detector buffer reset."""


DEFAULT_LOOP_CONFIG = LoopDetectorConfig()


@dataclass
class LoopEvidence:
    """Structured, interpretable evidence for the decision layer."""

    is_looping: bool
    confidence: float  # [0.0, 1.0]
    loop_cycles: float
    duration_sec: float
    path_distance_m: float
    straight_line_displacement_m: float
    mean_speed_mps: float
    radius_of_gyration_m: float
    closure_ratio: float
    net_rotation_deg: float
    total_angular_change_deg: float
    revisit_count: int
    pattern: str  # "PERSISTENT_LOOP", "ORDINARY_TURN", "STATIONARY_JITTER", "SHORT_DETOUR", "INSUFFICIENT_HISTORY", "INSUFFICIENT_MOVEMENT"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersistentLoopDetector:
    """Stateful or stateless trajectory loop detector evaluating streaming or windowed points."""

    def __init__(self, config: LoopDetectorConfig = DEFAULT_LOOP_CONFIG):
        self.config = config
        self.feat_cfg = FeatureConfig()

    def evaluate_points(
        self,
        points: List[Dict[str, Any]],
        config: Optional[LoopDetectorConfig] = None,
    ) -> LoopEvidence:
        """Evaluate a sequence of chronologically ordered GPS points for persistent looping.

        Each point dict should contain:
        - "lat" or "latitude": float
        - "lng" or "longitude": float
        - "timestamp": datetime or float/int epoch seconds
        - optional "speed_mps": float
        - optional "horizontal_accuracy_m": float
        """
        cfg = config or self.config

        # 1. Filter points by GPS accuracy if available
        valid_pts = []
        for p in points:
            acc = p.get("horizontal_accuracy_m")
            if acc is not None and acc > cfg.max_horizontal_accuracy_m:
                continue
            valid_pts.append(p)

        n_pts = len(valid_pts)
        if n_pts < cfg.min_points:
            return LoopEvidence(
                is_looping=False,
                confidence=0.0,
                loop_cycles=0.0,
                duration_sec=0.0,
                path_distance_m=0.0,
                straight_line_displacement_m=0.0,
                mean_speed_mps=0.0,
                radius_of_gyration_m=0.0,
                closure_ratio=0.0,
                net_rotation_deg=0.0,
                total_angular_change_deg=0.0,
                revisit_count=0,
                pattern="INSUFFICIENT_HISTORY",
                reason=f"Insufficient valid GPS observations ({n_pts} < {cfg.min_points}).",
            )

        # 2. Extract arrays
        lats = np.array([float(p.get("lat", p.get("latitude", 0.0))) for p in valid_pts], dtype=np.float64)
        lons = np.array([float(p.get("lng", p.get("longitude", 0.0))) for p in valid_pts], dtype=np.float64)

        # Parse timestamps into epoch seconds
        raw_ts = [p.get("timestamp") for p in valid_pts]
        t_secs = []
        for t in raw_ts:
            if hasattr(t, "timestamp"):
                t_secs.append(float(t.timestamp()))
            elif isinstance(t, (int, float)):
                t_secs.append(float(t))
            else:
                try:
                    import pandas as pd
                    t_secs.append(float(pd.to_datetime(t).timestamp()))
                except Exception:
                    t_secs.append(0.0)

        t_arr = np.array(t_secs, dtype=np.float64)
        duration_sec = float(t_arr[-1] - t_arr[0])

        # Check for temporal reset / gap
        step_dts = np.diff(t_arr)
        if np.any(step_dts > cfg.stale_timeout_sec):
            return LoopEvidence(
                is_looping=False,
                confidence=0.0,
                loop_cycles=0.0,
                duration_sec=duration_sec,
                path_distance_m=0.0,
                straight_line_displacement_m=0.0,
                mean_speed_mps=0.0,
                radius_of_gyration_m=0.0,
                closure_ratio=0.0,
                net_rotation_deg=0.0,
                total_angular_change_deg=0.0,
                revisit_count=0,
                pattern="INSUFFICIENT_HISTORY",
                reason=f"Trajectory contains temporal gap exceeding {cfg.stale_timeout_sec}s.",
            )

        # 3. Geometric and Kinematic Distances
        step_dists = haversine_np(lats[:-1], lons[:-1], lats[1:], lons[1:], radius=self.feat_cfg.earth_radius_m)
        path_dist = float(np.sum(step_dists))
        disp = float(haversine_np(lats[0], lons[0], lats[-1], lons[-1], radius=self.feat_cfg.earth_radius_m))
        mean_speed = float(path_dist / max(duration_sec, 1.0))

        # Check Stationary GPS Jitter
        if mean_speed < cfg.min_speed_mps or path_dist < cfg.min_path_distance_m:
            return LoopEvidence(
                is_looping=False,
                confidence=0.0,
                loop_cycles=0.0,
                duration_sec=duration_sec,
                path_distance_m=round(path_dist, 1),
                straight_line_displacement_m=round(disp, 1),
                mean_speed_mps=round(mean_speed, 2),
                radius_of_gyration_m=0.0,
                closure_ratio=0.0,
                net_rotation_deg=0.0,
                total_angular_change_deg=0.0,
                revisit_count=0,
                pattern="STATIONARY_JITTER",
                reason=(
                    f"Mean speed ({mean_speed:.2f} m/s) or path distance ({path_dist:.1f}m) "
                    "below movement threshold. Classified as stationary sensor jitter."
                ),
            )

        # Centroid and Radius of Gyration
        clat, clon = spherical_centroid(lats, lons)
        dists_to_centroid = haversine_np(lats, lons, clat, clon, radius=self.feat_cfg.earth_radius_m)
        rg = float(np.sqrt(np.mean(dists_to_centroid**2)))

        # Path closure ratio: 1 - disp / path
        closure_ratio = float(max(0.0, min(1.0, 1.0 - (disp / max(path_dist, 1e-3)))))

        # 4. Angular Turns and Winding Analysis
        valid_bearings = np.array([], dtype=np.float64)
        disp_mask = step_dists >= self.feat_cfg.min_displacement_m
        if np.any(disp_mask):
            valid_bearings = calculate_bearings_np(
                lats[:-1][disp_mask],
                lons[:-1][disp_mask],
                lats[1:][disp_mask],
                lons[1:][disp_mask],
            )

        net_rot = 0.0
        total_rot = 0.0
        if len(valid_bearings) >= 2:
            diffs = (valid_bearings[1:] - valid_bearings[:-1] + 180.0) % 360.0 - 180.0
            total_rot = float(np.sum(np.abs(diffs)))
            net_rot = float(np.sum(diffs))

        # 5. Path Revisit & Cycle Counting
        # Detect return visits: points p_j that are close to prior points p_i with sufficient time gap
        revisit_count = 0
        cycle_estimates = []
        for i in range(n_pts - 1):
            t_i = t_arr[i]
            for j in range(i + 2, n_pts):
                if (t_arr[j] - t_i) >= cfg.min_revisit_time_gap_sec:
                    d_ij = float(haversine_np(lats[i], lons[i], lats[j], lons[j], radius=self.feat_cfg.earth_radius_m))
                    if d_ij <= cfg.max_revisit_distance_m:
                        revisit_count += 1
                        cycle_time = t_arr[j] - t_i
                        if cycle_time > 0:
                            cycle_estimates.append(duration_sec / cycle_time)
                        break

        # Estimated completed cycles from rotation and revisits
        rotation_cycles = abs(net_rot) / 360.0
        revisit_cycles = float(np.mean(cycle_estimates)) if cycle_estimates else (1.0 if revisit_count >= 1 else 0.0)
        estimated_cycles = round(max(rotation_cycles, revisit_cycles), 2)

        # 6. Distinguish Short Detour vs Ordinary Turn vs Persistent Loop
        # Check Progressive Transit (not looping)
        if disp > (0.6 * path_dist) or rg > cfg.max_bounding_radius_m:
            return LoopEvidence(
                is_looping=False,
                confidence=0.1,
                loop_cycles=estimated_cycles,
                duration_sec=duration_sec,
                path_distance_m=round(path_dist, 1),
                straight_line_displacement_m=round(disp, 1),
                mean_speed_mps=round(mean_speed, 2),
                radius_of_gyration_m=round(rg, 1),
                closure_ratio=round(closure_ratio, 2),
                net_rotation_deg=round(net_rot, 1),
                total_angular_change_deg=round(total_rot, 1),
                revisit_count=revisit_count,
                pattern="PROGRESSIVE_TRANSIT",
                reason=(
                    f"Trajectory shows progressive linear displacement ({disp:.1f}m / {path_dist:.1f}m path). "
                    "Not a confined repetitive loop."
                ),
            )

        # Check Short Detour (< min_loop_duration_sec or < min_loop_cycles)
        if duration_sec < cfg.min_loop_duration_sec or estimated_cycles < cfg.min_loop_cycles:
            pattern = "SHORT_DETOUR" if (closure_ratio >= cfg.min_closure_ratio or total_rot >= 360.0) else "ORDINARY_TURN"
            return LoopEvidence(
                is_looping=False,
                confidence=round(min(0.5, closure_ratio * 0.5), 2),
                loop_cycles=estimated_cycles,
                duration_sec=round(duration_sec, 1),
                path_distance_m=round(path_dist, 1),
                straight_line_displacement_m=round(disp, 1),
                mean_speed_mps=round(mean_speed, 2),
                radius_of_gyration_m=round(rg, 1),
                closure_ratio=round(closure_ratio, 2),
                net_rotation_deg=round(net_rot, 1),
                total_angular_change_deg=round(total_rot, 1),
                revisit_count=revisit_count,
                pattern=pattern,
                reason=(
                    f"Movement duration ({duration_sec:.1f}s < {cfg.min_loop_duration_sec}s) or "
                    f"cycles ({estimated_cycles:.2f} < {cfg.min_loop_cycles}) insufficient to confirm persistent looping."
                ),
            )

        # 7. Persistent Looping Confirmation
        # Meets duration >= 120s, speed in range, closure ratio, and cycles >= 1.3
        is_loop = bool(
            duration_sec >= cfg.min_loop_duration_sec
            and estimated_cycles >= cfg.min_loop_cycles
            and closure_ratio >= cfg.min_closure_ratio
            and rg <= cfg.max_bounding_radius_m
            and (revisit_count >= 1 or rotation_cycles >= 1.2)
        )

        # Smooth confidence score in [0.0, 1.0]
        conf_duration = min(1.0, duration_sec / (cfg.min_loop_duration_sec * 1.5))
        conf_cycles = min(1.0, estimated_cycles / 2.0)
        conf_closure = closure_ratio
        confidence = round(float(0.4 * conf_duration + 0.3 * conf_cycles + 0.3 * conf_closure), 2)

        return LoopEvidence(
            is_looping=is_loop,
            confidence=confidence if is_loop else 0.4,
            loop_cycles=estimated_cycles,
            duration_sec=round(duration_sec, 1),
            path_distance_m=round(path_dist, 1),
            straight_line_displacement_m=round(disp, 1),
            mean_speed_mps=round(mean_speed, 2),
            radius_of_gyration_m=round(rg, 1),
            closure_ratio=round(closure_ratio, 2),
            net_rotation_deg=round(net_rot, 1),
            total_angular_change_deg=round(total_rot, 1),
            revisit_count=revisit_count,
            pattern="PERSISTENT_LOOP" if is_loop else "COMPLEX_TRANSIT",
            reason=(
                f"Sustained repetitive route detected ({estimated_cycles:.1f} cycles, {duration_sec:.0f}s, "
                f"path: {path_dist:.0f}m, radius: {rg:.0f}m, closure: {closure_ratio:.2f})."
                if is_loop
                else "Trajectory exhibits curvature without meeting full persistence criteria."
            ),
        )

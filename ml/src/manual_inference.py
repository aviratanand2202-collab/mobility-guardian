"""Manual Inference CLI for Single-Trajectory and GeoLife .plt Files.

Allows manual end-to-end inference testing on the frozen serialized XGBoost models
WITHOUT invoking or rerunning the full training pipeline.

Usage:
    # Zero-shot inference on a new GeoLife .plt file using frozen population prior:
    py -m ml.src.manual_inference --input "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"

    # Personalized inference with historical trajectories:
    py -m ml.src.manual_inference --input "ml/data/raw/geolife/000/Trajectory/20081024020959.plt" \\
        --history "ml/data/raw/geolife/000/Trajectory"

    # Evaluate across all 4 prediction horizons:
    py -m ml.src.manual_inference --input "ml/data/raw/geolife/000/Trajectory/20081023025304.plt" --horizon all
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ml.src.data import parse_plt_file
from ml.src.features import FeatureConfig, compute_window_features_dict
from ml.src.inference import RiskInferenceEngine
from ml.src.profile import (
    ProfileConfig,
    cluster_spatial_anchors,
    haversine_distance,
)
from ml.src.risk import RiskConfig, compute_empirical_baseline_from_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manual_inference")


class BaselineBuildResult(tuple):
    """Result tuple supporting backward-compatible 3-item unpacking: (baseline, trip_count, mode).

    Also provides structured access to discovered spatial anchors via .anchor_clusters attribute.
    """

    def __new__(cls, baseline, trip_count, mode, anchor_clusters=None):
        return super().__new__(cls, (baseline, trip_count, mode))

    def __init__(self, baseline, trip_count, mode, anchor_clusters=None):
        self.baseline = baseline
        self.trip_count = trip_count
        self.mode = mode
        self.anchor_clusters = anchor_clusters if anchor_clusters is not None else []
        self.anchors = self.anchor_clusters


def build_personalized_baseline_from_history(
    history_path: Union[str, Path],
    cfg: RiskConfig,
    feat_cfg: FeatureConfig,
    min_trips: int = 7,
    current_trajectory_file: Optional[Union[str, Path]] = None,
    as_of_time: Optional[Union[str, pd.Timestamp]] = None,
    profile_cfg: Optional[ProfileConfig] = None,
    user_id: Optional[str] = None,
) -> BaselineBuildResult:
    """Parse historical .plt files, construct empirical baseline, and discover spatial anchors.

    Strict Isolation and Zero Future-Data Leakage:
    - Current trajectory is excluded if present in the history folder.
    - If as_of_time is provided, observations at or after as_of_time are excluded.
    - Empty or invalid history safely returns BaselineBuildResult(None, 0, "COLD_START", anchor_clusters=[]).
    - If valid trips >= min_trips, discovers spatial anchors via frozen DBSCAN methodology
      (eps=150.0m, min_samples=3) without fabricating anchors.
    """
    if not history_path:
        return BaselineBuildResult(None, 0, "COLD_START", anchor_clusters=[])

    p = Path(history_path)
    if p.is_dir():
        plt_files = sorted(list(p.glob("*.plt")))
    elif p.is_file():
        plt_files = [p]
    else:
        logger.warning(f"History path not found or invalid: {history_path}")
        return BaselineBuildResult(None, 0, "COLD_START", anchor_clusters=[])

    # Strict isolation: exclude current trajectory file from historical profile data
    if current_trajectory_file is not None:
        curr_p = Path(current_trajectory_file).resolve()
        plt_files = [f for f in plt_files if f.resolve() != curr_p and f.stem != curr_p.stem]

    if not plt_files:
        logger.warning("No historical trajectory files remain after isolating current trajectory.")
        return BaselineBuildResult(None, 0, "COLD_START", anchor_clusters=[])

    all_windows = []
    all_endpoints = []
    valid_trip_count = 0
    as_of_ts = pd.to_datetime(as_of_time) if as_of_time is not None else None

    for f in plt_files:
        try:
            df_pts, _ = parse_plt_file(f)
            if df_pts is None or len(df_pts) < 3:
                continue
            df_pts["timestamp"] = pd.to_datetime(df_pts["timestamp"])

            # Temporal integrity: exclude future data at or after current trajectory start
            if as_of_ts is not None:
                if df_pts["timestamp"].dt.tz is not None and as_of_ts.tz is None:
                    effective_as_of = as_of_ts.tz_localize("UTC")
                elif df_pts["timestamp"].dt.tz is None and as_of_ts.tz is not None:
                    effective_as_of = as_of_ts.tz_localize(None)
                else:
                    effective_as_of = as_of_ts

                df_pts = df_pts[df_pts["timestamp"] < effective_as_of]
                if len(df_pts) < 3:
                    continue

            df_pts = df_pts.sort_values("timestamp").reset_index(drop=True)
            t0 = df_pts["timestamp"].iloc[0]
            elapsed = (df_pts["timestamp"] - t0).dt.total_seconds()
            df_pts["window_index"] = (elapsed // 120.0).astype(int)

            traj_id = f.stem
            u_name = f.parent.parent.name if f.parent.name.lower() == "trajectory" else "user"
            eff_uid = user_id or u_name
            file_had_window = False

            for w_idx, w_df in df_pts.groupby("window_index"):
                if len(w_df) >= 3:
                    dt_s = (w_df["timestamp"] - w_df["timestamp"].shift(1)).dt.total_seconds().fillna(0.0)
                    time_diffs = dt_s.to_numpy(dtype=np.float64)
                    w_feats = compute_window_features_dict(
                        window_id=f"{f.stem}_w_{w_idx}",
                        user_id=eff_uid,
                        trajectory_id=traj_id,
                        segment_id=f"{traj_id}_s0",
                        window_index=int(w_idx),
                        is_full_window=True,
                        lats=w_df["latitude"].to_numpy(dtype=np.float64),
                        lons=w_df["longitude"].to_numpy(dtype=np.float64),
                        ts=w_df["timestamp"].to_numpy(),
                        dts=time_diffs,
                        config=feat_cfg,
                    )
                    all_windows.append(w_feats)
                    file_had_window = True

            if file_had_window:
                valid_trip_count += 1
                lats = df_pts["latitude"].to_numpy(dtype=np.float64)
                lons = df_pts["longitude"].to_numpy(dtype=np.float64)
                if len(lats) > 1:
                    dists = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
                    total_path_m = float(np.sum(dists))
                else:
                    total_path_m = 0.0

                all_endpoints.append({
                    "user_id": eff_uid,
                    "trajectory_id": traj_id,
                    "latitude": float(lats[0]),
                    "longitude": float(lons[0]),
                    "timestamp": df_pts["timestamp"].iloc[0],
                    "endpoint_type": "start",
                    "path_distance_m": total_path_m,
                })
                all_endpoints.append({
                    "user_id": eff_uid,
                    "trajectory_id": traj_id,
                    "latitude": float(lats[-1]),
                    "longitude": float(lons[-1]),
                    "timestamp": df_pts["timestamp"].iloc[-1],
                    "endpoint_type": "end",
                    "path_distance_m": total_path_m,
                })
        except Exception as err:
            logger.warning(f"Skipping corrupted or unreadable historical file {f.name}: {err}")
            continue

    if not all_windows or valid_trip_count == 0:
        logger.warning("No evaluable windows extracted from historical files.")
        return BaselineBuildResult(None, 0, "COLD_START", anchor_clusters=[])

    if valid_trip_count < min_trips:
        logger.warning(
            f"History contains only {valid_trip_count} trips (< {min_trips}). "
            "Profile mode remains COLD_START."
        )
        return BaselineBuildResult(None, valid_trip_count, "COLD_START", anchor_clusters=[])

    hist_df = pd.DataFrame(all_windows)
    baseline = compute_empirical_baseline_from_windows(hist_df, features=cfg.target_features)

    # Discover spatial anchors using existing frozen DBSCAN methodology
    p_cfg = profile_cfg or ProfileConfig()
    if all_endpoints:
        endpoints_df = pd.DataFrame(all_endpoints)
        learned_anchors = cluster_spatial_anchors(endpoints_df, config=p_cfg)
        anchor_dicts = [a.to_dict() for a in learned_anchors]
    else:
        anchor_dicts = []

    return BaselineBuildResult(baseline, valid_trip_count, "PERSONALIZED", anchor_clusters=anchor_dicts)


def run_manual_inference(
    input_file: str,
    history_path: Optional[str] = None,
    horizon: str = "120",
    output_dir: str = "ml/data/inference",
    require_personalized: bool = False,
    user_id: Optional[str] = None,
    save_profile_to_store: bool = True,
    safe_area: Optional[Any] = None,
) -> Dict[str, Any]:
    """Perform safe, standalone manual inference on a trajectory file.

    Supports two explicit inference modes:
    1. NEW USER / COLD START:
       - No history provided (or invalid/empty history)
       - Uses frozen population prior baseline
       - Profile mode explicitly marked as COLD_START
       - Never fabricates personal history (trip_count=0)
    2. EXISTING USER / PERSONALIZED:
       - Historical trajectory folder provided (or pre-stored user profile loaded)
       - Strict isolation: current trajectory excluded from historical profile
       - Transitions to PERSONALIZED if >= 7 historical trips
       - Persists profile and learned spatial anchors to stored user profile repository
    """
    engine = RiskInferenceEngine()
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input trajectory file does not exist: {input_file}")

    df_pts_curr, _ = parse_plt_file(input_path)
    if df_pts_curr is None or df_pts_curr.empty:
        raise ValueError(f"Could not parse valid trajectory points from {input_file}")

    u_id = user_id or (
        input_path.parent.parent.name
        if input_path.parent.name.lower() == "trajectory"
        else input_path.parent.name
    )
    t_start_current = pd.to_datetime(df_pts_curr["timestamp"]).min()

    baseline = None
    profile_mode = "COLD_START"
    cold_start_status = "COLD_START"
    trip_count = 0

    if history_path:
        logger.info(f"Processing historical trajectory data from: {history_path}")
        hist_res = build_personalized_baseline_from_history(
            history_path,
            cfg=engine.risk_cfg,
            feat_cfg=engine.feature_cfg,
            min_trips=engine.config.min_warmup_trips,
            current_trajectory_file=input_path,
            as_of_time=t_start_current,
            user_id=u_id,
        )
        hist_baseline, n_trips, mode = hist_res
        hist_anchors = getattr(hist_res, "anchor_clusters", [])
        trip_count = n_trips

        if mode == "PERSONALIZED" and hist_baseline is not None:
            profile_mode = "PERSONALIZED"
            cold_start_status = "ML_DRIVEN"
            baseline = hist_baseline
            logger.info(
                f"Personalized baseline constructed from {trip_count} historical trips "
                f"({len(hist_anchors)} spatial anchors discovered)."
            )

            prof_dict = {
                "user_id": u_id,
                "profile_mode": "PERSONALIZED",
                "cold_start_status": "ML_DRIVEN",
                "trip_count": trip_count,
                "baseline_distribution": hist_baseline,
                "anchor_clusters": hist_anchors,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            if save_profile_to_store:
                engine.save_user_profile(u_id, prof_dict)
                logger.info(f"Persisted personalized profile for user {u_id} to {engine.profiles_dir}")
            else:
                engine.user_profiles[u_id] = prof_dict
                if "baseline_distribution" in prof_dict:
                    engine.user_baselines[u_id] = prof_dict["baseline_distribution"]
        else:
            if require_personalized:
                raise ValueError(
                    f"Insufficient history: {n_trips} trips provided, but "
                    f"minimum {engine.config.min_warmup_trips} required for personalized mode."
                )
            profile_mode = "COLD_START"
            cold_start_status = "RULE_BASED_FALLBACK" if n_trips > 0 else "COLD_START"
            baseline = engine.population_baseline
            logger.info(
                f"History insufficient ({n_trips} trips). Using population baseline with {profile_mode} mode."
            )
    else:
        # Check stored user profile in repository
        stored_prof = engine.load_user_profile(u_id)
        if stored_prof and stored_prof.get("trip_count", 0) >= engine.config.min_warmup_trips:
            profile_mode = "PERSONALIZED"
            cold_start_status = stored_prof.get("cold_start_status", "ML_DRIVEN")
            trip_count = stored_prof.get("trip_count", 7)
            baseline = stored_prof.get("baseline_distribution", engine.population_baseline)
            logger.info(f"Loaded existing stored user profile for {u_id} ({trip_count} trips).")
        else:
            if require_personalized:
                raise ValueError(f"No stored personalized profile found for user {u_id} and no history provided.")
            profile_mode = "COLD_START"
            cold_start_status = "COLD_START"
            trip_count = 0  # Never fabricate personal history
            baseline = engine.population_baseline
            logger.info(f"No history provided or stored. Running NEW USER / COLD START mode for {u_id}.")

    # Determine horizons to evaluate
    if horizon.lower() == "all":
        horizons = list(engine.config.supported_horizons)
    else:
        horizons = [int(horizon)]

    logger.info(f"Parsing and evaluating input trajectory: {input_file}")
    results_by_horizon: Dict[str, List[Dict[str, Any]]] = {}

    u_ctx = {
        "user_id": u_id,
        "trip_count": trip_count,
        "profile_mode": profile_mode,
        "cold_start_status": cold_start_status,
    }

    for h in horizons:
        preds = engine.predict_plt_trajectory(
            plt_file_path=input_path,
            user_id=u_id,
            user_baseline=baseline,
            user_context=u_ctx,
            horizon_sec=h,
            safe_area=safe_area,
        )
        for p in preds:
            p["metadata"]["profile_mode"] = profile_mode
            p["metadata"]["cold_start_status"] = cold_start_status
            p["metadata"]["trip_count"] = trip_count
            p["metadata"]["baseline_type"] = "PERSONALIZED" if profile_mode == "PERSONALIZED" else "POPULATION"

        results_by_horizon[f"{h}s"] = preds

    # Summary
    first_h_key = f"{horizons[0]}s"
    n_windows = len(results_by_horizon[first_h_key])
    alerts_fired = {
        h_k: sum(1 for w in wins if w["trigger_state"]["binary_alert"])
        for h_k, wins in results_by_horizon.items()
    }
    max_scores = {
        h_k: max(w["risk_score"] for w in wins)
        for h_k, wins in results_by_horizon.items()
    }

    last_win_geo = (
        results_by_horizon[first_h_key][-1].get("geospatial_context", {})
        if results_by_horizon.get(first_h_key)
        else {}
    )
    safe_area_state = last_win_geo.get("safe_area_state", "SAFE_AREA_UNAVAILABLE")
    familiarity_state = last_win_geo.get("familiarity_state", "NO_HISTORY")

    last_win = (
        results_by_horizon[first_h_key][-1]
        if results_by_horizon.get(first_h_key)
        else {}
    )
    decision_state = last_win.get("decision_state", "SAFE / NORMAL")
    human_reason = last_win.get("human_readable_reason", "")
    decision_dict = last_win.get("decision", {})

    persisted_anchors_count = 0
    active_profile = engine.load_user_profile(u_id)
    if active_profile and "anchor_clusters" in active_profile:
        persisted_anchors_count = len(active_profile["anchor_clusters"])

    curr_score = float(last_win.get("risk_score", 0.0))
    peak_score = float(max_scores[first_h_key])

    summary = {
        "input_file": str(input_path.resolve()),
        "user_id": u_id,
        "n_evaluated_windows": n_windows,
        "profile_mode": profile_mode,
        "cold_start_status": cold_start_status,
        "trip_count": trip_count,
        "baseline_type": "PERSONALIZED" if profile_mode == "PERSONALIZED" else "POPULATION",
        "persisted_anchor_clusters_count": persisted_anchors_count,
        "decision_state": decision_state,
        "human_readable_reason": human_reason,
        "decision": decision_dict,
        "safe_area_state": safe_area_state,
        "familiarity_state": familiarity_state,
        "nearest_anchor_id": last_win_geo.get("nearest_anchor_id"),
        "distance_to_nearest_anchor_m": last_win_geo.get("distance_to_nearest_anchor_m"),
        "nearest_anchor_status": last_win_geo.get("nearest_anchor_status"),
        "is_caregiver_confirmed": last_win_geo.get("is_caregiver_confirmed", False),
        "current_window_risk_score": curr_score,
        "peak_trajectory_risk_score": peak_score,
        "evaluated_horizons": horizons,
        "alerts_fired_per_horizon": alerts_fired,
        "max_risk_scores": max_scores,
        "last_geospatial_context": last_win_geo,
        "window_predictions": results_by_horizon,
    }

    # Save output to separate inference directory (do NOT touch canonical artifacts)
    stem = input_path.stem
    out_json = os.path.join(output_dir, f"inference_{stem}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Manual inference complete! Saved results to: {out_json}")
    print("\n" + "=" * 60)
    print("MANUAL INFERENCE SUMMARY RESULT")
    print("=" * 60)
    print(f"File: {input_file}")
    print(f"User: {summary['user_id']}")
    print(f"Evaluated Windows (120s each): {n_windows}")
    print(f"Profile Mode: {profile_mode} (Status: {cold_start_status}, Trips: {trip_count})")
    print(f"Baseline Type: {summary['baseline_type']}")
    print(f"Persisted Anchors: {persisted_anchors_count}")
    for h in horizons:
        h_k = f"{h}s"
        thresh = engine.thresholds[h]
        print(f"  Horizon {h}s (Threshold={thresh:.2f}):")
        print(f"    Alerts: {alerts_fired[h_k]}/{n_windows}")
        print(f"    Peak Trajectory Risk Score: {max_scores[h_k]:.2f}/100.0")

    print("\n" + "=" * 60)
    print("HUMAN-READABLE DECISION OUTPUT (ML Interpretation Layer)")
    print("=" * 60)
    print(f"  Decision State:             [{decision_state}]")
    print(f"  Headline:                   {decision_dict.get('headline', '')}")
    print(f"  Explanation:                {human_reason}")
    b_tier = decision_dict.get('behavioral_tier', 'QUIESCENT')
    b_pat = decision_dict.get('behavioral_pattern', 'NORMAL')
    print(f"  Behavioral Risk Tier:       {b_tier}")
    print(f"  Geometric Movement Pattern: {b_pat}")
    print(f"  Current Window Risk Score:  {curr_score:.2f}/100")
    print(f"  Peak Trajectory Risk:       {peak_score:.2f}/100")
    print(f"  Safe-Area State:            {safe_area_state}")
    print(f"  Familiarity State:          {familiarity_state}")
    if last_win_geo.get("nearest_anchor_id"):
        print(f"  Nearest Anchor:             {last_win_geo['nearest_anchor_id']}")
        print(f"  Anchor Distance:            {last_win_geo['distance_to_nearest_anchor_m']}m")
        print(f"  Anchor Status:              {last_win_geo.get('nearest_anchor_status')}")
    print(f"  Caregiver Confirmed:        {last_win_geo.get('is_caregiver_confirmed', False)}")
    print(f"  Profile Mode:               {profile_mode}")
    print("  Notice: Factual non-clinical interpretation. No dementia/wandering claim.")
    print("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Manual Inference CLI for Mobility Guardian")
    parser.add_argument("--input", required=True, help="Path to input .plt trajectory file")
    parser.add_argument("--history", default=None, help="Path to directory of historical .plt files")
    parser.add_argument("--horizon", default="120", help="Horizon in seconds (120, 360, 600, 840, or 'all')")
    parser.add_argument("--output-dir", default="ml/data/inference", help="Directory for inference output files")
    parser.add_argument("--require-personalized", action="store_true", help="Fail if history is < 7 trips")
    parser.add_argument("--user-id", default=None, help="User ID override")
    parser.add_argument("--safe-area", default=None, help="Configured safe area (JSON string, file, or lat,lon,radius)")
    args = parser.parse_args()

    run_manual_inference(
        input_file=args.input,
        history_path=args.history,
        horizon=args.horizon,
        output_dir=args.output_dir,
        require_personalized=args.require_personalized,
        user_id=args.user_id,
        safe_area=args.safe_area,
    )


if __name__ == "__main__":
    main()

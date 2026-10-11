"""Production Inference & Deployment Bridge Module.

Provides end-to-end inference from serialized frozen XGBoost models:
1. Model Deserialization: Loads native XGBoost models, calibrators, thresholds, and metadata.
2. Feature Reconstruction: Matches exact 49-feature schema with identical column ordering.
3. Baseline Adaptation: Supports personalized historical profiles or frozen population priors.
4. Telemetry Streaming: Buffers TelemetryPayload readings into 120s analysis windows.
5. Behavioral Preservation: Computes geometric classifications (PACING, LAPPING, RANDOM_DRIFT)
   independently of the future excursion prediction target.
6. Signal Degradation & PDR Fallback: Freezes ML scoring during GPS blind spots.
7. Schema Conformance: Returns RiskScoreOutput strictly conforming to shared/schema.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.src.behavior import BehaviorConfig, classify_behavior_dataframe
from ml.src.data import parse_plt_file
from ml.src.features import FeatureConfig, compute_window_features_dict
from ml.src.decision import (
    DecisionConfig,
    interpret_decision,
)
from ml.src.geofence import (
    GeospatialConfig,
    SafeAreaBoundary,
    evaluate_geospatial_context,
)
from ml.src.risk import RiskConfig, extract_prediction_features

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class InferenceConfig:
    """Centralized inference configuration."""

    model_dir: str = "ml/models/xgboost"
    profiles_dir: str = "ml/models/profiles"
    metadata_filename: str = "model_metadata.json"
    population_baseline_filename: str = "population_baseline.json"

    # Supported evaluation horizons (seconds)
    supported_horizons: Tuple[int, ...] = (120, 360, 600, 840)
    default_horizon_sec: int = 120

    # Window parameters
    window_duration_sec: float = 120.0
    min_points_per_window: int = 3

    # Personalization thresholds
    min_warmup_trips: int = 7
    min_baseline_samples: int = 10

    # Operational Risk Tier Boundaries (0-100 continuous score)
    # §2.6 Battery & Polling Table mapping:
    # 0 <= Score < 35  -> QUIESCENT (Tier 0)
    # 35 <= Score < threshold -> NORMAL_TRANSIT (Tier 1)
    # threshold <= Score < 70 -> SUSPICIOUS (Tier 2)
    # Score >= 70      -> CRITICAL (Tier 3)
    quiescent_max_score: float = 35.0
    critical_min_score: float = 70.0

    # Signal degradation
    pdr_max_blind_spot_sec: float = 900.0  # 15 minutes max blind spot


# ==============================================================================
# INFERENCE ENGINE
# ==============================================================================


class RiskInferenceEngine:
    """Production inference engine evaluating real-time or offline trajectories."""

    def __init__(self, config: InferenceConfig = InferenceConfig()):
        self.config = config
        self.risk_cfg = RiskConfig()
        self.feature_cfg = FeatureConfig()
        self.behavior_cfg = BehaviorConfig()

        self.models: Dict[int, xgb.Booster] = {}
        self.thresholds: Dict[int, float] = {}
        self.calibrators: Dict[int, Dict[str, Any]] = {}
        self.feature_names: List[str] = []
        self.population_baseline: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

        # Streaming session buffers per user: user_id -> list of raw point dicts
        self.point_buffers: Dict[str, List[Dict[str, Any]]] = {}
        # Completed window histories per user: user_id -> list of window feature dicts
        self.window_histories: Dict[str, List[Dict[str, Any]]] = {}
        # Last known risk score output per user
        self.last_risk_outputs: Dict[str, Dict[str, Any]] = {}
        # Cached user baseline profiles: user_id -> baseline dict
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        # Cached full user profiles: user_id -> profile dict
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.geospatial_cfg = GeospatialConfig()
        self.decision_cfg = DecisionConfig()
        self.user_safe_areas: Dict[str, Any] = {}
        self.profiles_dir: Path = Path(self.config.profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._load_artifacts()

    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load stored user profile from memory or disk repository."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        prof_file = self.profiles_dir / f"{user_id}.json"
        if prof_file.exists():
            try:
                with open(prof_file, "r") as f:
                    prof_data = json.load(f)
                self.user_profiles[user_id] = prof_data
                if "baseline_distribution" in prof_data:
                    self.user_baselines[user_id] = prof_data["baseline_distribution"]
                if "safe_area" in prof_data or "safe_areas" in prof_data:
                    self.set_user_safe_area(user_id, prof_data.get("safe_area") or prof_data.get("safe_areas"))
                return prof_data
            except Exception as e:
                logger.warning(f"Failed to load user profile for {user_id}: {e}")
        return None

    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Path:
        """Persist user profile to repository and update memory caches."""
        prof_file = self.profiles_dir / f"{user_id}.json"
        with open(prof_file, "w") as f:
            json.dump(profile_data, f, indent=2)
        self.user_profiles[user_id] = profile_data
        if "baseline_distribution" in profile_data:
            self.user_baselines[user_id] = profile_data["baseline_distribution"]
        if "safe_area" in profile_data or "safe_areas" in profile_data:
            self.set_user_safe_area(user_id, profile_data.get("safe_area") or profile_data.get("safe_areas"))
        return prof_file

    def delete_user_profile(self, user_id: str) -> bool:
        """Remove user profile from repository and memory."""
        self.user_profiles.pop(user_id, None)
        self.user_baselines.pop(user_id, None)
        self.user_safe_areas.pop(user_id, None)
        prof_file = self.profiles_dir / f"{user_id}.json"
        if prof_file.exists():
            prof_file.unlink()
            return True
        return False

    def set_user_safe_area(self, user_id: str, safe_area: Any) -> Optional[SafeAreaBoundary]:
        """Configure or update external safe area boundary for a user."""
        boundary = SafeAreaBoundary.from_spec(safe_area)
        if boundary:
            self.user_safe_areas[user_id] = boundary
        else:
            self.user_safe_areas.pop(user_id, None)
        return boundary

    def get_user_safe_area(self, user_id: str) -> Optional[SafeAreaBoundary]:
        """Retrieve configured safe area boundary for a user."""
        return self.user_safe_areas.get(user_id)

    def clear_user_safe_area(self, user_id: str) -> bool:
        """Clear configured safe area boundary for a user."""
        return self.user_safe_areas.pop(user_id, None) is not None

    def evaluate_geospatial_context(
        self,
        lat: Optional[float],
        lon: Optional[float],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        safe_area: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Evaluate safe-area state and location familiarity for a coordinate."""
        prof = user_profile
        if prof is None and user_id:
            prof = self.load_user_profile(user_id)
        effective_safe = (
            safe_area if safe_area is not None
            else (self.user_safe_areas.get(user_id) if user_id else None)
        )
        res = evaluate_geospatial_context(
            latitude=lat,
            longitude=lon,
            safe_area=effective_safe,
            user_profile=prof,
            config=self.geospatial_cfg,
        )
        return res.to_dict()

    def _load_artifacts(self) -> None:
        """Load serialized XGBoost models, metadata, thresholds, and population baseline."""
        meta_path = os.path.join(self.config.model_dir, self.config.metadata_filename)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Model metadata not found at {meta_path}. Run model serialization first."
            )

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        if not self.feature_names:
            raise ValueError("Model metadata does not contain feature_names.")

        # Load models and thresholds per horizon
        horizons_meta = self.metadata.get("horizons", {})
        for h in self.config.supported_horizons:
            h_key = f"{h}s"
            if h_key not in horizons_meta:
                continue

            h_info = horizons_meta[h_key]
            model_file = h_info.get("model_file", f"xgb_horizon_{h}s.json")
            model_path = os.path.join(self.config.model_dir, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model artifact not found for horizon {h}s at {model_path}")

            booster = xgb.Booster()
            booster.load_model(model_path)
            self.models[h] = booster
            self.thresholds[h] = float(h_info.get("optimal_threshold", 0.5))
            self.calibrators[h] = h_info.get("calibrator", {})

        # Load population baseline
        pop_path = os.path.join(self.config.model_dir, self.config.population_baseline_filename)
        if os.path.exists(pop_path):
            with open(pop_path, "r") as f:
                self.population_baseline = json.load(f)

        logger.info(
            f"RiskInferenceEngine loaded {len(self.models)} models for horizons: "
            f"{list(self.models.keys())} with {len(self.feature_names)} features."
        )

    def compute_calibrated_probability(self, raw_prob: float, horizon_sec: int) -> float:
        """Apply validation-fitted logistic calibrator: p_cal = 1 / (1 + exp(-(w * p + b)))."""
        cal_info = self.calibrators.get(horizon_sec, {})
        coef = cal_info.get("coef")
        intercept = cal_info.get("intercept")

        if coef is not None and intercept is not None:
            w = float(coef[0][0]) if isinstance(coef[0], list) else float(coef[0])
            b = float(intercept[0]) if isinstance(intercept, list) else float(intercept)
            logit = w * float(raw_prob) + b
            # Numerically stable sigmoid
            if logit >= 0:
                p_cal = 1.0 / (1.0 + math.exp(-logit))
            else:
                exp_z = math.exp(logit)
                p_cal = exp_z / (1.0 + exp_z)
            return float(p_cal)

        return float(raw_prob)

    def map_score_to_tiers(
        self,
        calibrated_prob: float,
        horizon_sec: int,
    ) -> Tuple[float, str, int]:
        """Map calibrated probability to continuous risk score [0, 100], risk tier, and polling tier."""
        threshold = self.thresholds.get(horizon_sec, 0.35)
        risk_score = round(float(calibrated_prob) * 100.0, 2)

        # Map to operational tiers
        if risk_score < self.config.quiescent_max_score:
            risk_tier = "QUIESCENT"
            polling_tier = 0
        elif calibrated_prob < threshold:
            risk_tier = "NORMAL_TRANSIT"
            polling_tier = 1
        elif risk_score < self.config.critical_min_score:
            risk_tier = "SUSPICIOUS"
            polling_tier = 2
        else:
            risk_tier = "CRITICAL"
            polling_tier = 3

        return risk_score, risk_tier, polling_tier

    def predict_window(
        self,
        curr_window: Union[pd.Series, Dict[str, Any]],
        prev_window: Optional[Union[pd.Series, Dict[str, Any]]] = None,
        user_baseline: Optional[Dict[str, Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        horizon_sec: int = 120,
        safe_area: Optional[Any] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform manual or automated prediction on a single 120-second analysis window."""
        if horizon_sec not in self.models:
            raise ValueError(
                f"Unsupported horizon: {horizon_sec}s. Available: {list(self.models.keys())}"
            )

        u_ctx = user_context.copy() if user_context else {}
        user_id = str(curr_window.get("user_id", u_ctx.get("user_id", "default_user")))
        stored_profile = (
            self.load_user_profile(user_id)
            if user_id not in ("default_user", "unknown_user", "")
            else None
        )

        # Determine effective baseline and profile mode
        if user_baseline is not None:
            baseline = user_baseline
            is_custom_baseline = (baseline != self.population_baseline)
        elif stored_profile and stored_profile.get("trip_count", 0) >= self.config.min_warmup_trips:
            baseline = stored_profile.get("baseline_distribution", self.population_baseline)
            is_custom_baseline = True
        else:
            baseline = self.population_baseline
            is_custom_baseline = False

        if not baseline:
            raise ValueError("No valid user baseline or population baseline available for prediction.")

        # Determine profile_mode and cold_start_status
        if "profile_mode" in u_ctx:
            profile_mode = u_ctx["profile_mode"]
        elif is_custom_baseline:
            profile_mode = "PERSONALIZED"
        else:
            profile_mode = "COLD_START"

        if "cold_start_status" in u_ctx:
            cold_start_status = u_ctx["cold_start_status"]
        elif profile_mode == "PERSONALIZED":
            cold_start_status = stored_profile.get("cold_start_status", "ML_DRIVEN") if stored_profile else "ML_DRIVEN"
        else:
            cold_start_status = "COLD_START"

        trip_count = u_ctx.get(
            "trip_count",
            stored_profile.get("trip_count", 0) if stored_profile else (0 if profile_mode == "COLD_START" else 7),
        )
        baseline_type = "PERSONALIZED" if profile_mode == "PERSONALIZED" else "POPULATION"

        u_ctx["profile_mode"] = profile_mode
        u_ctx["cold_start_status"] = cold_start_status
        u_ctx["trip_count"] = trip_count

        # 1. Extract exact 49 features matching training pipeline
        feats = extract_prediction_features(
            curr_window=curr_window,
            prev_window=prev_window,
            baselines=baseline,
            user_context=u_ctx,
            config=self.risk_cfg,
        )

        # 2. Construct DMatrix with exact feature ordering
        feature_vector = [feats.get(c, np.nan) for c in self.feature_names]
        feat_df = pd.DataFrame([feature_vector], columns=self.feature_names)
        dmatrix = xgb.DMatrix(feat_df)

        # 3. Model inference
        booster = self.models[horizon_sec]
        raw_prob = float(booster.predict(dmatrix)[0])
        cal_prob = self.compute_calibrated_probability(raw_prob, horizon_sec)
        threshold = self.thresholds[horizon_sec]
        binary_pred = 1 if cal_prob >= threshold else 0

        # 4. Map to continuous score and tiers
        risk_score, risk_tier, polling_tier = self.map_score_to_tiers(cal_prob, horizon_sec)

        # 5. Behavioral movement pattern classification (independent geometric assessment)
        if isinstance(curr_window, (pd.Series, dict)):
            curr_df = pd.DataFrame([dict(curr_window)])
        else:
            curr_df = curr_window.copy()
        behavior_req_cols = {
            "window_id": str(curr_window.get("window_id", "w_0")),
            "user_id": str(curr_window.get("user_id", "u_0")),
            "trajectory_id": str(curr_window.get("trajectory_id", "t_0")),
            "segment_id": str(curr_window.get("segment_id", "s_0")),
            "window_index": int(curr_window.get("window_index", 0)),
            "is_full_window": bool(curr_window.get("is_full_window", True)),
            "path_distance_m": 0.0,
            "bbox_diagonal_m": 0.0,
            "bbox_width_m": 0.0,
            "bbox_height_m": 0.0,
            "is_kinematically_evaluable": True,
            "has_extreme_kinematic_transition": False,
            "path_closure_ratio": 0.0,
            "pacing_tendency": 0.0,
            "backtracking_tendency": 0.0,
            "loop_metric": 0.0,
            "entropy_directional": 0.0,
            "heading_variability": 0.0,
            "turn_frequency": 0.0,
            "heading_change_mean": 0.0,
            "straight_line_displacement_m": 0.0,
        }
        for col, default_val in behavior_req_cols.items():
            if col not in curr_df.columns:
                curr_df[col] = default_val
        behavior_df = classify_behavior_dataframe(curr_df, self.behavior_cfg)
        behavior_pattern = "NORMAL"
        behavior_conf = 1.0
        if not behavior_df.empty and "assigned_behavior_pattern" in behavior_df.columns:
            behavior_pattern = str(behavior_df.iloc[0]["assigned_behavior_pattern"])
            behavior_conf = float(behavior_df.iloc[0].get("behavior_confidence", 1.0))

        # 6. Extract top contributing features for explainability
        top_features = []
        for feat in ["max_mad_z_score", "p95_exceedance_count", "mean_speed_mps", "tortuosity_index"]:
            if feat in feats and pd.notna(feats[feat]):
                top_features.append({"feature": feat, "value": round(float(feats[feat]), 3)})

        # 7. Assemble RiskScoreOutput conforming to shared/schema.json
        window_start = str(curr_window.get("start_time", ""))
        window_end = str(curr_window.get("end_time", ""))
        user_id = str(curr_window.get("user_id", u_ctx.get("user_id", "unknown_user")))

        output = {
            "user_id": user_id,
            "timestamp": window_end or datetime.now(timezone.utc).isoformat(),
            "risk_tier": risk_tier,
            "risk_score": risk_score,
            "polling_tier": polling_tier,
            "predicted_lead_time_sec": float(horizon_sec) if binary_pred == 1 else None,
            "battery_override_active": False,
            "kinematic_features": {
                "mean_speed_mps": round(float(curr_window.get("mean_speed_mps", 0.0)), 2),
                "tortuosity_index": round(float(curr_window.get("tortuosity_index", 1.0)), 2),
                "entropy_directional": round(float(curr_window.get("entropy_directional", 0.0)), 2),
                "straight_line_displacement_m": round(
                    float(curr_window.get("straight_line_displacement_m", 0.0)), 1
                ),
                "path_distance_m": round(float(curr_window.get("path_distance_m", 0.0)), 1),
                "max_mad_z_score": round(float(feats.get("max_mad_z_score", 0.0)), 2),
                "behavioral_indicator": behavior_pattern,
                "behavioral_confidence": round(behavior_conf, 3),
            },
            "trigger_state": {
                "window_seconds": int(self.config.window_duration_sec),
                "horizon_sec": horizon_sec,
                "decision_threshold": threshold,
                "calibrated_probability": round(cal_prob, 4),
                "raw_probability": round(raw_prob, 4),
                "binary_alert": bool(binary_pred),
            },
            "explainability": {
                "top_features": top_features,
                "behavior_classification": behavior_pattern,
                "disclaimer": (
                    "Associational kinematic excursion prediction relative to learned baseline. "
                    "Reflects mathematical outlier trajectory probability, NOT clinical dementia or wandering."
                ),
            },
            "metadata": {
                "window_id": str(curr_window.get("window_id", "")),
                "start_time": window_start,
                "end_time": window_end,
                "profile_mode": profile_mode,
                "cold_start_status": cold_start_status,
                "trip_count": trip_count,
                "baseline_type": baseline_type,
            },
        }

        # 7. Evaluate orthogonal geospatial and familiarity context
        lat_val = (
            latitude if latitude is not None
            else curr_window.get("latitude", curr_window.get("lat"))
        )
        lon_val = (
            longitude if longitude is not None
            else curr_window.get("longitude", curr_window.get("lon", curr_window.get("lng")))
        )
        effective_lat = float(lat_val) if lat_val is not None and pd.notna(lat_val) else None
        effective_lon = float(lon_val) if lon_val is not None and pd.notna(lon_val) else None
        effective_safe = safe_area if safe_area is not None else self.get_user_safe_area(user_id)

        geo_ctx = self.evaluate_geospatial_context(
            lat=effective_lat,
            lon=effective_lon,
            user_id=user_id,
            user_profile=stored_profile,
            safe_area=effective_safe,
        )

        output["safe_area_state"] = geo_ctx["safe_area_state"]
        output["familiarity_state"] = geo_ctx["familiarity_state"]
        output["geospatial_context"] = geo_ctx
        output["metadata"]["safe_area_state"] = geo_ctx["safe_area_state"]
        output["metadata"]["familiarity_state"] = geo_ctx["familiarity_state"]

        # 8. Deterministic decision interpretation combining ML risk, safe-area, and familiarity
        dec = interpret_decision(output, self.decision_cfg)
        output["decision_state"] = dec.decision_state
        output["display_state"] = dec.display_state
        output["human_readable_reason"] = dec.reason
        output["decision"] = dec.to_dict()
        output["metadata"]["decision_state"] = dec.decision_state
        output["metadata"]["headline"] = dec.headline

        return output

    def predict_telemetry_reading(
        self, payload: Dict[str, Any], safe_area: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Ingest a single TelemetryPayload reading and evaluate risk if window is ready."""
        user_id = str(payload.get("user_id", "default_user"))
        sig_status = payload.get("signal_status", {})
        sig_state = sig_status.get("state", "VALID")
        timestamp_str = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
        loc = payload.get("location", {})
        lat_val = float(loc.get("lat")) if "lat" in loc and loc.get("lat") is not None else None
        lon_val = float(loc.get("lng")) if "lng" in loc and loc.get("lng") is not None else None
        effective_safe = safe_area if safe_area is not None else self.get_user_safe_area(user_id)

        # 1. Check Signal Degradation Contract
        if sig_state == "DEGRADED_SIGNAL":
            last_out = self.last_risk_outputs.get(user_id)
            pdr_tier = 1
            imu_metrics = payload.get("imu_metrics")
            if imu_metrics:
                disp = float(imu_metrics.get("net_displacement_m", 0.0))
                if disp > 50.0:
                    pdr_tier = 3
                elif disp > 25.0:
                    pdr_tier = 2

            # Frozen risk score with PDR escalation
            base_score = last_out.get("risk_score", 20.0) if last_out else 20.0
            escalated_score = max(base_score, 75.0 if pdr_tier >= 3 else (40.0 if pdr_tier == 2 else base_score))
            risk_tier = "CRITICAL" if pdr_tier >= 3 else ("SUSPICIOUS" if pdr_tier == 2 else "QUIESCENT")

            geo_ctx = self.evaluate_geospatial_context(
                lat=lat_val,
                lon=lon_val,
                user_id=user_id,
                safe_area=effective_safe,
            )

            degraded_out = {
                "user_id": user_id,
                "timestamp": timestamp_str,
                "risk_tier": risk_tier,
                "risk_score": round(escalated_score, 2),
                "polling_tier": pdr_tier,
                "predicted_lead_time_sec": None,
                "battery_override_active": False,
                "kinematic_features": {"signal_state": "DEGRADED_SIGNAL", "pdr_tier": pdr_tier},
                "trigger_state": {"frozen_reason": "DEGRADED_SIGNAL_PDR_FALLBACK"},
                "explainability": {"status": "GPS degraded; score frozen, tracking via PDR net displacement."},
                "safe_area_state": geo_ctx["safe_area_state"],
                "familiarity_state": geo_ctx["familiarity_state"],
                "geospatial_context": geo_ctx,
                "metadata": {
                    "profile_mode": "DEGRADED_SIGNAL",
                    "safe_area_state": geo_ctx["safe_area_state"],
                    "familiarity_state": geo_ctx["familiarity_state"],
                },
            }
            dec = interpret_decision(degraded_out, self.decision_cfg)
            degraded_out["decision_state"] = dec.decision_state
            degraded_out["display_state"] = dec.display_state
            degraded_out["human_readable_reason"] = dec.reason
            degraded_out["decision"] = dec.to_dict()
            return degraded_out

        # 2. Valid Signal: Buffer point
        sensor = payload.get("sensor_metrics", {})
        point_record = {
            "latitude": float(loc.get("lat", 0.0)),
            "longitude": float(loc.get("lng", 0.0)),
            "altitude_m": loc.get("altitude_m", 0.0),
            "timestamp": pd.to_datetime(timestamp_str),
            "speed_mps": float(sensor.get("speed_mps", 0.0)),
            "heading_deg": float(sensor.get("heading_deg", 0.0)),
        }

        if user_id not in self.point_buffers:
            self.point_buffers[user_id] = []
        buf = self.point_buffers[user_id]
        buf.append(point_record)

        # Check if accumulated buffer duration >= window_duration_sec
        t_start = buf[0]["timestamp"]
        t_curr = buf[-1]["timestamp"]
        duration = (t_curr - t_start).total_seconds()

        if duration >= self.config.window_duration_sec and len(buf) >= self.config.min_points_per_window:
            pts_df = pd.DataFrame(buf)
            win_features = compute_window_features_dict(pts_df, self.feature_cfg)
            win_features["window_id"] = f"{user_id}_w_{len(self.window_histories.get(user_id, []))}"
            win_features["user_id"] = user_id
            win_features["start_time"] = t_start.isoformat()
            win_features["end_time"] = t_curr.isoformat()
            win_features["latitude"] = float(point_record["latitude"])
            win_features["longitude"] = float(point_record["longitude"])

            # Previous window for lag features
            user_hist = self.window_histories.setdefault(user_id, [])
            prev_win = user_hist[-1] if user_hist else None

            # User baseline: check stored profile or fallback to population baseline
            stored_prof = self.load_user_profile(user_id)
            if stored_prof and stored_prof.get("trip_count", 0) >= self.config.min_warmup_trips:
                user_base = stored_prof.get("baseline_distribution", self.population_baseline)
                prof_mode = "PERSONALIZED"
                cs_status = stored_prof.get("cold_start_status", "ML_DRIVEN")
                t_count = stored_prof.get("trip_count", 7)
            else:
                user_base = self.population_baseline
                prof_mode = "COLD_START"
                cs_status = "COLD_START"
                t_count = 0

            u_ctx = {
                "user_id": user_id,
                "trip_count": t_count,
                "profile_mode": prof_mode,
                "cold_start_status": cs_status,
            }

            risk_out = self.predict_window(
                curr_window=win_features,
                prev_window=prev_win,
                user_baseline=user_base,
                user_context=u_ctx,
                horizon_sec=self.config.default_horizon_sec,
                safe_area=effective_safe,
                latitude=win_features["latitude"],
                longitude=win_features["longitude"],
            )

            # Update histories and reset point buffer
            user_hist.append(win_features)
            self.last_risk_outputs[user_id] = risk_out
            self.point_buffers[user_id] = [buf[-1]]
            return risk_out

        # Intermediate reading while window is accumulating
        geo_ctx = self.evaluate_geospatial_context(
            lat=lat_val,
            lon=lon_val,
            user_id=user_id,
            safe_area=effective_safe,
        )

        last_out = self.last_risk_outputs.get(user_id)
        if last_out:
            updated = {
                **last_out,
                "timestamp": timestamp_str,
                "safe_area_state": geo_ctx["safe_area_state"],
                "familiarity_state": geo_ctx["familiarity_state"],
                "geospatial_context": geo_ctx,
            }
            dec = interpret_decision(updated, self.decision_cfg)
            updated["decision_state"] = dec.decision_state
            updated["display_state"] = dec.display_state
            updated["human_readable_reason"] = dec.reason
            updated["decision"] = dec.to_dict()
            return updated

        accum_out = {
            "user_id": user_id,
            "timestamp": timestamp_str,
            "risk_tier": "QUIESCENT",
            "risk_score": 0.0,
            "polling_tier": 0,
            "predicted_lead_time_sec": None,
            "battery_override_active": False,
            "kinematic_features": {"points_accumulated": len(buf), "buffer_duration_sec": round(duration, 1)},
            "trigger_state": {"status": "ACCUMULATING_WINDOW"},
            "explainability": {"status": "Accumulating 120s window telemetry."},
            "safe_area_state": geo_ctx["safe_area_state"],
            "familiarity_state": geo_ctx["familiarity_state"],
            "geospatial_context": geo_ctx,
            "metadata": {
                "profile_mode": "COLD_START",
                "cold_start_status": "COLD_START",
                "trip_count": 0,
                "baseline_type": "POPULATION",
                "safe_area_state": geo_ctx["safe_area_state"],
                "familiarity_state": geo_ctx["familiarity_state"],
            },
        }
        dec = interpret_decision(accum_out, self.decision_cfg)
        accum_out["decision_state"] = dec.decision_state
        accum_out["display_state"] = dec.display_state
        accum_out["human_readable_reason"] = dec.reason
        accum_out["decision"] = dec.to_dict()
        return accum_out

    def predict_plt_trajectory(
        self,
        plt_file_path: Union[str, Path],
        user_id: Optional[str] = None,
        user_baseline: Optional[Dict[str, Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        horizon_sec: int = 120,
        safe_area: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Parse raw GeoLife .plt file, segment into 120s windows, and run model inference."""
        df_pts, diag = parse_plt_file(plt_file_path, user_id=user_id)
        if df_pts is None or df_pts.empty:
            raise ValueError(f"Failed to parse PLT file {plt_file_path}: {diag.get('error')}")

        u_id = user_id or str(df_pts["user_id"].iloc[0])
        df_pts["timestamp"] = pd.to_datetime(df_pts["timestamp"])
        df_pts = df_pts.sort_values("timestamp").reset_index(drop=True)

        # Segment points into 120-second non-overlapping windows
        t0 = df_pts["timestamp"].iloc[0]
        elapsed_sec = (df_pts["timestamp"] - t0).dt.total_seconds()
        df_pts["window_index"] = (elapsed_sec // self.config.window_duration_sec).astype(int)

        traj_id = Path(plt_file_path).stem
        windows: List[Dict[str, Any]] = []
        for w_idx, w_df in df_pts.groupby("window_index"):
            if len(w_df) < self.config.min_points_per_window:
                continue
            dt_series = (w_df["timestamp"] - w_df["timestamp"].shift(1)).dt.total_seconds().fillna(0.0)
            time_diffs = dt_series.to_numpy(dtype=np.float64)
            w_feats = compute_window_features_dict(
                window_id=f"{u_id}_w_{w_idx}",
                user_id=u_id,
                trajectory_id=traj_id,
                segment_id=f"{traj_id}_s0",
                window_index=int(w_idx),
                is_full_window=True,
                lats=w_df["latitude"].to_numpy(dtype=np.float64),
                lons=w_df["longitude"].to_numpy(dtype=np.float64),
                ts=w_df["timestamp"].to_numpy(),
                dts=time_diffs,
                config=self.feature_cfg,
            )
            w_feats["latitude"] = float(w_df["latitude"].iloc[-1])
            w_feats["longitude"] = float(w_df["longitude"].iloc[-1])
            windows.append(w_feats)

        if not windows:
            raise ValueError(f"No valid 120-second windows extracted from {plt_file_path}")

        # Resolve user baseline and context
        u_ctx = user_context.copy() if user_context else {}
        stored_prof = self.load_user_profile(u_id)

        if user_baseline is not None:
            baseline = user_baseline
            is_custom = (baseline != self.population_baseline)
        elif stored_prof and stored_prof.get("trip_count", 0) >= self.config.min_warmup_trips:
            baseline = stored_prof.get("baseline_distribution", self.population_baseline)
            is_custom = True
            if "trip_count" not in u_ctx:
                u_ctx["trip_count"] = stored_prof.get("trip_count", 7)
            if "cold_start_status" not in u_ctx:
                u_ctx["cold_start_status"] = stored_prof.get("cold_start_status", "ML_DRIVEN")
        else:
            baseline = self.population_baseline
            is_custom = False

        if "profile_mode" not in u_ctx:
            u_ctx["profile_mode"] = "PERSONALIZED" if is_custom else "COLD_START"
        if "cold_start_status" not in u_ctx:
            u_ctx["cold_start_status"] = "ML_DRIVEN" if u_ctx["profile_mode"] == "PERSONALIZED" else "COLD_START"
        if "trip_count" not in u_ctx:
            u_ctx["trip_count"] = 7 if u_ctx["profile_mode"] == "PERSONALIZED" else 0

        effective_safe = safe_area if safe_area is not None else self.get_user_safe_area(u_id)

        predictions: List[Dict[str, Any]] = []
        for i, curr_win in enumerate(windows):
            prev_win = windows[i - 1] if i > 0 else None
            pred = self.predict_window(
                curr_window=curr_win,
                prev_window=prev_win,
                user_baseline=baseline,
                user_context=u_ctx,
                horizon_sec=horizon_sec,
                safe_area=effective_safe,
                latitude=curr_win.get("latitude"),
                longitude=curr_win.get("longitude"),
            )
            predictions.append(pred)

        return predictions

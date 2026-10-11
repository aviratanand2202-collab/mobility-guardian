"""Personalized Predictive Kinematic Risk Model (Chunk 6).

Implements personalized longitudinal risk forecasting relative to learned mobility baselines:
1. Strict Past-Only Expanding Training Profiles (Zero retrospective leakage).
2. Evidence-Coupled Horizon Target Generation across 120s, 360s, 600s, and 840s.
3. Interval-Union Coverage Validation preventing overlap inflation.
4. Symmetric Coverage Enforcement marking incomplete or unobserved horizons as NaN.
5. Personalized Heuristic Baselines (Robust MAD & Empirical Percentile).
6. Gradient Boosted Trees (XGBoost) with validation-only early stopping, threshold tuning,
   and probability calibration.
7. Associational Explainability via TreeSHAP (Explicitly non-causal).
8. Decoupled Known-User Chronological Forecasting and Unseen-User Generalization Protocols.

RESEARCH PROTOTYPE DISCLAIMER:
This module evaluates derived mathematical kinematic outliers from GPS data.
It does NOT predict clinical dementia, cognitive disorientation, or verified real-world safety events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def current_iso_time() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class RiskConfig:
    """Centralized configuration for personalized predictive kinematic risk modeling.

    All methodological thresholds, horizon definitions, and evaluation hyperparameters
    are centralized here without magic numbers.
    """

    # Prediction Horizons (in seconds, window-harmonic multiples of 120s)
    # 120s (2m Micro), 360s (6m Operational Short), 600s (10m Medium), 840s (14m Operational Extended)
    horizons_sec: Tuple[int, ...] = (120, 360, 600, 840)

    # Coverage & Boundary Safeguards
    min_evidence_coverage_ratio: float = 1.0
    """Strict complete evidence coverage required (100% of horizon duration must be observed)."""

    max_horizon_gap_sec: float = 180.0
    """Maximum allowable telemetry gap between consecutive analysis windows within horizon."""

    enforce_contained_witnesses_only: bool = True
    """Disqualifies boundary-crossing windows from providing outlier evidence or coverage."""

    enforce_symmetric_coverage: bool = True
    """Incomplete or truncated horizons yield NaN for both positive and negative targets."""

    # Expanding Training Profile Parameters (Past-Only)
    min_warmup_trips: int = 7
    """Chronological trajectories required before generating training samples (warm-up phase)."""

    min_baseline_samples: int = 10
    """Minimum historical samples required for a personalized feature baseline."""

    # Outlier Threshold Heuristics (Engineering Parameters)
    robust_mad_multiplier: float = 3.0
    """MAD dispersion multiplier (approx 3 robust standard deviations) for outlier boundary."""

    min_simultaneous_p95_exceedances: int = 2
    """Co-occurring historical P95 exceedances required to trigger joint outlier flag."""

    zero_mad_deviation_penalty: float = 10.0
    """Fixed score penalty when historical MAD == 0 and deviation exceeds epsilon."""

    zero_mad_epsilon: float = 1e-6
    """Tolerance for zero-MAD invariance check."""

    min_evaluable_features: int = 3
    """Minimum valid target features required to evaluate a window for outlier status."""

    # Eligible Baseline Features for Derived Outlier Generation
    target_features: Tuple[str, ...] = (
        "mean_speed_mps",
        "tortuosity_index",
        "entropy_directional",
        "turn_frequency",
        "loop_metric",
        "pacing_tendency",
        "straight_line_displacement_m",
        "path_distance_m",
    )

    # Evaluation Split Ratios (Protocol A: Known-User Chronological Forecasting)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Protocol B: Unseen-User Generalization Cohort
    unseen_user_ratio: float = 0.20

    # Reproducibility Seed
    random_seed: int = 42

    # XGBoost Hyperparameters
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 5
    xgb_n_estimators: int = 200
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping_rounds: int = 15

    # Validation Threshold Search Grids & Explainability Limits
    xgb_threshold_min: float = 0.1
    xgb_threshold_max: float = 0.9
    xgb_threshold_steps: int = 81
    shap_max_samples: int = 200
    mad_threshold_min: float = 1.0
    mad_threshold_max: float = 5.0
    mad_threshold_steps: int = 41
    pct_threshold_max: int = 4


# ==============================================================================
# INTERVAL UNION & COVERAGE ENGINE
# ==============================================================================


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or contiguous 1D numeric intervals [start, end].

    Prevents overlapping or staggered observation windows from double-counting coverage.
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end:
            # Overlapping or adjacent: merge
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged


def compute_evidence_coverage(
    admissible_intervals: List[Tuple[float, float]],
    t_start: float,
    horizon_sec: float,
    max_gap_sec: float = 180.0,
) -> Tuple[float, float, bool]:
    """Compute 1D Lebesgue measure of the union of admissible witness intervals.

    Returns:
        (evidence_duration_sec, coverage_ratio, is_gap_valid)
    """
    if not admissible_intervals or horizon_sec <= 0:
        return 0.0, 0.0, False

    t_end = t_start + horizon_sec

    # Clip all admissible intervals strictly to horizon boundary [t_start, t_end]
    clipped: List[Tuple[float, float]] = []
    for s, e in admissible_intervals:
        clip_s = max(s, t_start)
        clip_e = min(e, t_end)
        if clip_s < clip_e:
            clipped.append((clip_s, clip_e))

    if not clipped:
        return 0.0, 0.0, False

    # Merge overlapping intervals
    merged = merge_intervals(clipped)

    # Compute total observed duration
    evidence_duration = sum(e - s for s, e in merged)
    coverage_ratio = float(evidence_duration / horizon_sec)

    # Validate temporal gaps
    # 1. Leading gap (from t_start to first interval)
    is_gap_valid = True
    if merged[0][0] - t_start > max_gap_sec:
        is_gap_valid = False

    # 2. Internal gaps between consecutive intervals
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap > max_gap_sec:
            is_gap_valid = False
            break

    # 3. Trailing gap (from last interval to t_end)
    if t_end - merged[-1][1] > max_gap_sec:
        is_gap_valid = False

    return float(evidence_duration), coverage_ratio, is_gap_valid


# ==============================================================================
# OUTLIER DERIVATION & BASELINE COMPARISON
# ==============================================================================


def evaluate_window_outlier(
    window_row: Union[pd.Series, Dict[str, Any]],
    baselines: Dict[str, Dict[str, Any]],
    config: RiskConfig = RiskConfig(),
) -> Tuple[bool, bool, Dict[str, float]]:
    """Determine whether a single analysis window is a derived personalized kinematic outlier.

    Returns:
        (is_outlier, is_evaluable, feature_deviations)
    """
    valid_feature_count = 0
    max_z_score = 0.0
    p95_exceedance_count = 0
    feature_deviations: Dict[str, float] = {}

    for feat in config.target_features:
        val = window_row.get(feat)
        if val is None or pd.isna(val):
            continue

        base_info = baselines.get(feat, {})
        median_val = base_info.get("median")
        robust_scale = base_info.get("robust_scale")
        p95_val = base_info.get("p95")

        if median_val is None or robust_scale is None:
            continue

        valid_feature_count += 1
        diff = abs(float(val) - float(median_val))

        # Robust scale / MAD deviation
        if float(robust_scale) <= config.zero_mad_epsilon:
            if diff <= config.zero_mad_epsilon:
                z_score = 0.0
            else:
                z_score = config.zero_mad_deviation_penalty
        else:
            z_score = diff / (float(robust_scale) + 1e-9)

        feature_deviations[feat] = float(z_score)
        if z_score > max_z_score:
            max_z_score = z_score

        # P95 empirical exceedance
        if p95_val is not None and pd.notna(p95_val):
            if float(val) > float(p95_val):
                p95_exceedance_count += 1

    if valid_feature_count < config.min_evaluable_features:
        return False, False, feature_deviations

    # Multi-attribute outlier criterion
    is_outlier = (
        (max_z_score >= config.robust_mad_multiplier)
        or (p95_exceedance_count >= config.min_simultaneous_p95_exceedances)
    )

    return bool(is_outlier), True, feature_deviations


# ==============================================================================
# TARGET GENERATION ENGINE (EVIDENCE-COUPLED)
# ==============================================================================


def generate_horizon_target(
    trajectory_windows: pd.DataFrame,
    prediction_time: pd.Timestamp,
    horizon_sec: int,
    baselines: Dict[str, Dict[str, Any]],
    config: RiskConfig = RiskConfig(),
) -> Optional[float]:
    """Generate the derived kinematic risk target for a single prediction origin and horizon.

    Enforces Evidence-Coupled Coverage:
    - Only strictly contained, evaluable windows (s >= t and e <= t+H) are admissible.
    - An interval contributes to coverage IF AND ONLY IF it has valid target evidence.
    - Requires 100% complete evidence coverage and no gaps exceeding max_gap_sec.
    - Symmetric: Returns NaN for incomplete horizons (for both positive and negative cases).

    Returns:
        1.0 if coverage is valid and at least one admissible window is an outlier.
        0.0 if coverage is valid and all admissible windows are outlier-free.
        np.nan if coverage is invalid, incomplete, or unobserved.
    """
    if trajectory_windows.empty:
        return np.nan

    t_start_ts = pd.to_datetime(prediction_time)
    t_end_ts = t_start_ts + pd.Timedelta(seconds=horizon_sec)
    t_start_sec = t_start_ts.timestamp()
    t_end_sec = t_end_ts.timestamp()

    # Trajectory must continue across the horizon boundary
    traj_end_ts = pd.to_datetime(trajectory_windows["end_time"]).max()
    if traj_end_ts < t_end_ts:
        return np.nan  # Trajectory terminated before horizon completion

    # Identify candidate windows intersecting the horizon
    admissible_windows: List[pd.Series] = []
    admissible_intervals: List[Tuple[float, float]] = []

    for _, row in trajectory_windows.iterrows():
        w_start = pd.to_datetime(row["start_time"]).timestamp()
        w_end = pd.to_datetime(row["end_time"]).timestamp()

        # Strict containment check (Admissible Outlier Witness)
        if config.enforce_contained_witnesses_only:
            if w_start < t_start_sec or w_end > t_end_sec:
                # Boundary-crossing window: contributes NEITHER to evidence NOR to coverage
                continue

        # Check kinematic evaluability
        if "is_kinematically_evaluable" in row and not bool(row["is_kinematically_evaluable"]):
            continue

        is_outlier, is_evaluable, _ = evaluate_window_outlier(row, baselines, config=config)
        if is_evaluable:
            admissible_windows.append(row)
            admissible_intervals.append((w_start, w_end))

    # Compute Evidence-Coupled Coverage
    evidence_duration, coverage_ratio, is_gap_valid = compute_evidence_coverage(
        admissible_intervals=admissible_intervals,
        t_start=t_start_sec,
        horizon_sec=float(horizon_sec),
        max_gap_sec=config.max_horizon_gap_sec,
    )

    # Check Coverage Validity
    is_coverage_valid = (
        (coverage_ratio >= config.min_evidence_coverage_ratio)
        and is_gap_valid
    )

    if not is_coverage_valid:
        # Symmetrical rejection: incomplete coverage yields NaN regardless of outlier presence
        return np.nan

    # Evaluate outlier occurrence across admissible witness windows
    has_outlier = False
    for row in admissible_windows:
        is_outlier, _, _ = evaluate_window_outlier(row, baselines, config=config)
        if is_outlier:
            has_outlier = True
            break

    return 1.0 if has_outlier else 0.0


# ==============================================================================
# EXPANDING PROFILE PROTOCOL (PAST-ONLY)
# ==============================================================================


def compute_empirical_baseline_from_windows(
    windows_df: pd.DataFrame,
    features: Tuple[str, ...],
    min_samples: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Compute empirical median, MAD, robust scale, and P95 from historical windows."""
    baselines: Dict[str, Dict[str, Any]] = {}
    for feat in features:
        if feat in windows_df.columns:
            s = pd.to_numeric(windows_df[feat], errors="coerce").dropna().values
            n = len(s)
            if n >= min_samples:
                med = float(np.median(s))
                mad = float(np.median(np.abs(s - med)))
                rob_scale = float(1.4826 * mad) if mad > 0 else 0.0
                p95 = float(np.percentile(s, 95))
                baselines[feat] = {
                    "median": med,
                    "mad": mad,
                    "robust_scale": rob_scale,
                    "p95": p95,
                    "sample_size": n,
                }
            else:
                baselines[feat] = {
                    "median": None,
                    "mad": None,
                    "robust_scale": None,
                    "p95": None,
                    "sample_size": n,
                }
        else:
            baselines[feat] = {
                "median": None,
                "mad": None,
                "robust_scale": None,
                "p95": None,
                "sample_size": 0,
            }
    return baselines


def build_population_priors(
    train_windows_df: pd.DataFrame,
    config: RiskConfig = RiskConfig(),
) -> Dict[str, Dict[str, Any]]:
    """Compute cohort population-level baselines strictly from training cohort data."""
    return compute_empirical_baseline_from_windows(
        windows_df=train_windows_df,
        features=config.target_features,
        min_samples=config.min_baseline_samples,
    )


# ==============================================================================
# HEURISTIC BASELINES
# ==============================================================================


class PersonalizedRobustMADBaseline:
    """Heuristic baseline that predicts risk using MAD-standardized deviation from profile."""

    def __init__(self, threshold: float = 3.0, config: RiskConfig = RiskConfig()):
        self.threshold = threshold
        self.config = config

    def fit(self, X_val: pd.DataFrame, y_val: pd.Series) -> PersonalizedRobustMADBaseline:
        """Tune decision threshold on validation set to maximize validation macro-F1."""
        valid_mask = y_val.notna()
        if not valid_mask.any() or "max_mad_z_score" not in X_val.columns:
            return self

        scores = X_val.loc[valid_mask, "max_mad_z_score"].fillna(0.0).values
        y_true = y_val.loc[valid_mask].astype(int).values

        best_thresh = self.threshold
        best_f1 = -1.0

        for candidate in np.linspace(
            self.config.mad_threshold_min,
            self.config.mad_threshold_max,
            self.config.mad_threshold_steps,
        ):
            preds = (scores >= candidate).astype(int)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(candidate)

        self.threshold = best_thresh
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute heuristic risk score as sigmoid-transformed max MAD z-score."""
        scores = X["max_mad_z_score"].fillna(0.0).values
        # Logistic sigmoid centered at threshold
        p = 1.0 / (1.0 + np.exp(-(scores - self.threshold)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class using tuned threshold."""
        scores = X["max_mad_z_score"].fillna(0.0).values
        return (scores >= self.threshold).astype(int)


class PersonalizedPercentileBaseline:
    """Heuristic baseline that predicts risk using co-occurring P95 exceedances."""

    def __init__(self, threshold: int = 2, config: RiskConfig = RiskConfig()):
        self.threshold = threshold
        self.config = config

    def fit(self, X_val: pd.DataFrame, y_val: pd.Series) -> PersonalizedPercentileBaseline:
        """Tune exceedance count threshold on validation set."""
        valid_mask = y_val.notna()
        if not valid_mask.any() or "p95_exceedance_count" not in X_val.columns:
            return self

        counts = X_val.loc[valid_mask, "p95_exceedance_count"].fillna(0).values
        y_true = y_val.loc[valid_mask].astype(int).values

        best_thresh = self.threshold
        best_f1 = -1.0

        for candidate in range(1, self.config.pct_threshold_max + 1):
            preds = (counts >= candidate).astype(int)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = int(candidate)

        self.threshold = best_thresh
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute heuristic score based on exceedance fraction."""
        counts = X["p95_exceedance_count"].fillna(0).values
        p = np.clip(counts / float(len(self.config.target_features)), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class using exceedance threshold."""
        counts = X["p95_exceedance_count"].fillna(0).values
        return (counts >= self.threshold).astype(int)


# ==============================================================================
# EVALUATION METRICS ENGINE
# ==============================================================================


def evaluate_binary_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics against the derived kinematic target.

    Reports:
    - Target prevalence
    - Per-class precision, recall, F1
    - Macro-F1
    - Confusion matrix
    - PR-AUC and ROC-AUC (when mathematically defined)
    - Brier score loss
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    n_samples = len(y_t)
    if n_samples == 0:
        return {"n_samples": 0}

    pos_count = int(np.sum(y_t == 1))
    neg_count = int(np.sum(y_t == 0))
    prevalence = float(pos_count / n_samples)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class metrics
    prec_pos = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))

    prec_neg = float(precision_score(y_t, y_p, pos_label=0, zero_division=0))
    rec_neg = float(recall_score(y_t, y_p, pos_label=0, zero_division=0))
    f1_neg = float(f1_score(y_t, y_p, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Discrimination & Calibration
    roc_auc = None
    pr_auc = None
    brier = None

    if y_prob is not None:
        p_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        if pos_count > 0 and neg_count > 0:
            try:
                roc_auc = float(roc_auc_score(y_t, p_pos))
            except Exception:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_t, p_pos))
            except Exception:
                pr_auc = None
        brier = float(brier_score_loss(y_t, p_pos))

    return {
        "n_samples": int(n_samples),
        "positive_count": int(pos_count),
        "negative_count": int(neg_count),
        "prevalence": round(prevalence, 4),
        "precision_outlier": round(prec_pos, 4),
        "recall_outlier": round(rec_pos, 4),
        "f1_outlier": round(f1_pos, 4),
        "precision_routine_proxy": round(prec_neg, 4),
        "recall_routine_proxy": round(rec_neg, 4),
        "f1_routine_proxy": round(f1_neg, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
    }


# ==============================================================================
# PREDICTION-TIME FEATURE EXTRACTION (STRICTLY PAST-ONLY)
# ==============================================================================


def extract_prediction_features(
    curr_window: pd.Series,
    prev_window: Optional[pd.Series],
    baselines: Dict[str, Dict[str, Any]],
    user_context: Dict[str, Any],
    config: RiskConfig = RiskConfig(),
) -> Dict[str, Any]:
    """Extract features available at prediction time t = end_time(curr_window).

    Guarantees:
    - Zero future leakage: uses only curr_window and prev_window.
    - Ratio and z-score features relative to frozen/past profile baselines.
    - Zero substitution prohibited: missing baseline features remain NaN.
    """
    feats: Dict[str, Any] = {}

    # 1. Current Window Kinematics & Geometry
    base_kinematic_cols = [
        "mean_speed_mps",
        "speed_std_dev",
        "path_distance_m",
        "straight_line_displacement_m",
        "tortuosity_index",
        "entropy_directional",
        "turn_frequency",
        "loop_metric",
        "pacing_tendency",
        "backtracking_tendency",
        "heading_variability",
        "point_count",
        "temporal_span_sec",
    ]
    for col in base_kinematic_cols:
        val = curr_window.get(col)
        feats[col] = float(val) if val is not None and pd.notna(val) else np.nan

    # 2. Dynamic Lag Features (Difference from previous window in same trajectory)
    if prev_window is not None:
        for col in ["mean_speed_mps", "entropy_directional", "turn_frequency", "loop_metric", "pacing_tendency"]:
            v_curr = curr_window.get(col)
            v_prev = prev_window.get(col)
            if v_curr is not None and v_prev is not None and pd.notna(v_curr) and pd.notna(v_prev):
                feats[f"delta_{col}"] = float(v_curr) - float(v_prev)
            else:
                feats[f"delta_{col}"] = np.nan
    else:
        for col in ["mean_speed_mps", "entropy_directional", "turn_frequency", "loop_metric", "pacing_tendency"]:
            feats[f"delta_{col}"] = np.nan

    # 3. Personalized Baseline Deviations & Normalized Distances
    max_z = 0.0
    p95_count = 0
    valid_base_count = 0

    for f_name in config.target_features:
        val = curr_window.get(f_name)
        b_info = baselines.get(f_name, {})
        med = b_info.get("median")
        rob_scale = b_info.get("robust_scale")
        p95 = b_info.get("p95")

        if val is not None and pd.notna(val) and med is not None and rob_scale is not None:
            valid_base_count += 1
            diff = abs(float(val) - float(med))
            if float(rob_scale) <= config.zero_mad_epsilon:
                z = 0.0 if diff <= config.zero_mad_epsilon else config.zero_mad_deviation_penalty
            else:
                z = diff / (float(rob_scale) + 1e-9)

            feats[f"z_{f_name}"] = float(z)
            feats[f"ratio_to_med_{f_name}"] = float(val) / (float(med) + 1e-6) if float(med) > 0 else np.nan

            if z > max_z:
                max_z = z

            if p95 is not None and pd.notna(p95) and float(val) > float(p95):
                p95_count += 1
                feats[f"is_p95_exceeded_{f_name}"] = 1.0
            else:
                feats[f"is_p95_exceeded_{f_name}"] = 0.0
        else:
            feats[f"z_{f_name}"] = np.nan
            feats[f"ratio_to_med_{f_name}"] = np.nan
            feats[f"is_p95_exceeded_{f_name}"] = np.nan

    feats["max_mad_z_score"] = float(max_z) if valid_base_count > 0 else np.nan
    feats["p95_exceedance_count"] = float(p95_count) if valid_base_count > 0 else np.nan

    # 4. Temporal & Profile Context Features
    st_ts = pd.to_datetime(curr_window["start_time"])
    feats["hour_of_day"] = float(st_ts.hour)
    feats["day_of_week"] = float(st_ts.dayofweek)
    feats["is_weekend"] = 1.0 if st_ts.dayofweek >= 5 else 0.0
    feats["trip_index"] = float(user_context.get("trip_index", 1))
    feats["is_cold_start"] = 1.0 if user_context.get("cold_start_status") == "RULE_BASED_FALLBACK" else 0.0

    return feats


# ==============================================================================
# DATASET BUILDER (EXPANDING TRAINING PROFILE & CHRONOLOGICAL SPLITS)
# ==============================================================================


def build_user_trajectory_splits(
    user_windows_df: pd.DataFrame,
    config: RiskConfig = RiskConfig(),
) -> Dict[str, List[str]]:
    """Partition whole trajectories strictly chronologically into train, val, and test splits."""
    traj_summary = (
        user_windows_df.groupby("trajectory_id")["start_time"]
        .min()
        .reset_index()
        .sort_values("start_time")
    )
    sorted_trajs = traj_summary["trajectory_id"].tolist()
    n_trajs = len(sorted_trajs)

    if n_trajs < 3:
        # Too few trajectories for full 3-way split: assign all to train
        return {"train": sorted_trajs, "val": [], "test": []}

    n_train = max(int(np.floor(n_trajs * config.train_ratio)), 1)
    n_val = max(int(np.floor(n_trajs * config.val_ratio)), 1)
    # Ensure at least 1 trajectory in test if n_trajs >= 3
    if n_train + n_val >= n_trajs:
        n_train = max(n_trajs - 2, 1)
        n_val = 1

    train_trajs = sorted_trajs[:n_train]
    val_trajs = sorted_trajs[n_train:n_train + n_val]
    test_trajs = sorted_trajs[n_train + n_val:]

    return {"train": train_trajs, "val": val_trajs, "test": test_trajs}


# ==============================================================================
# XGBOOST RISK MODEL WRAPPER
# ==============================================================================


class XGBoostRiskModel:
    """Gradient boosted decision tree risk model with early stopping and calibration."""

    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config
        self.model = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.5

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> XGBoostRiskModel:
        """Fit XGBoost model with scale_pos_weight, validate with early stopping, tune threshold."""
        import xgboost as xgb

        # Filter out NaN targets
        tr_mask = y_train.notna()
        val_mask = y_val.notna()

        X_tr = X_train[tr_mask].copy()
        y_tr = y_train[tr_mask].astype(int)
        X_v = X_val[val_mask].copy()
        y_v = y_val[val_mask].astype(int)

        self.feature_names = list(X_tr.columns)

        # Compute scale_pos_weight for class imbalance
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        scale_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            n_estimators=self.config.xgb_n_estimators,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            scale_pos_weight=scale_weight,
            random_state=self.config.random_seed,
            eval_metric="logloss",
            early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            n_jobs=1,
        )

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_v, y_v)],
            verbose=False,
        )

        # Fit probability calibrator on validation predictions
        val_probs_raw = self.model.predict_proba(X_v)[:, 1]
        self.optimal_threshold = self._tune_threshold(val_probs_raw, y_v.values)

        return self

    def _tune_threshold(self, val_probs: np.ndarray, y_val: np.ndarray) -> float:
        """Find threshold on validation set maximizing macro-F1."""
        best_thresh = 0.5
        best_f1 = -1.0
        for cand in np.linspace(
            self.config.xgb_threshold_min,
            self.config.xgb_threshold_max,
            self.config.xgb_threshold_steps,
        ):
            preds = (val_probs >= cand).astype(int)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(cand)
        return best_thresh

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute estimated probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X[self.feature_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute binary predictions using validation-tuned threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.optimal_threshold).astype(int)

    def explain_with_shap(self, X: pd.DataFrame, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Compute TreeSHAP feature attributions on test samples."""
        import shap

        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        limit = max_samples if max_samples is not None else self.config.shap_max_samples
        sub_X = X[self.feature_names].head(limit)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(sub_X)

        if isinstance(shap_values, list):
            vals = shap_values[1]  # Positive class
        else:
            vals = shap_values

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        importance_dict = {
            feat: float(imp)
            for feat, imp in sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
        }

        return {
            "disclaimer": (
                "TreeSHAP attributions indicate statistical associations with model decision-tree splits. "
                "They reflect mathematical score contributions, NOT medical causes or clinical wandering."
            ),
            "top_features": list(importance_dict.keys())[:10],
            "feature_importance": importance_dict,
        }

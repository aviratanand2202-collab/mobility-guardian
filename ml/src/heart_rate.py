"""Heart Rate Biometric Evidence & Personal Baseline Evaluator.

Provides physiological context alongside kinematic mobility indicators:
1. Baseline Integrity: Calculates personal baselines (mean, std) exclusively from
   >= 10 valid on-wrist readings. Does NOT substitute arbitrary population defaults.
2. Signal Validity & Freshness: Enforces strict wear-state validation (ON_WRIST)
   and timestamp freshness (<= 60s). Stale, missing, or off-wrist readings are explicitly
   flagged as UNAVAILABLE/INSUFFICIENT and NEVER treated as normal or safe.
3. Exertion Concordance: Compares heart rate with concurrent movement speed. Distinguishes
   expected cardiovascular response during physical exertion (>= 1.2 m/s) from anomalous
   resting elevation (< 0.8 m/s with z >= 2.0).
4. Non-Suppression Guarantee: Normal or missing heart rate must NEVER suppress or lower
   an urgent geofence breach, persistent loop, or kinematic anomaly.

Scientific & Clinical Disclaimer:
Heart rate analysis provides associational physiological context only. It does NOT constitute
clinical diagnosis, cardiac telemetry, panic detection, dementia evaluation, or proof of distress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS & CONSTANTS
# ==============================================================================


class HeartRateEvidenceStatus(str, Enum):
    """Integrity and availability status of the wearable reading."""

    VALID = "VALID"
    INSUFFICIENT_BASELINE = "INSUFFICIENT_BASELINE"
    STALE_READING = "STALE_READING"
    OFF_WRIST = "OFF_WRIST"
    INVALID_READING = "INVALID_READING"
    MISSING_DATA = "MISSING_DATA"


class HeartRateResponsePattern(str, Enum):
    """Categorical physiological response pattern relative to baseline and activity."""

    ELEVATED_UNUSUAL_RESTING = "ELEVATED_UNUSUAL_RESTING"
    EXERTION_CONCORDANT = "EXERTION_CONCORDANT"
    WITHIN_NORMAL_BASELINE = "WITHIN_NORMAL_BASELINE"
    UNAVAILABLE = "UNAVAILABLE"


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class HeartRateConfig:
    """Configurable thresholds for heart rate evidence evaluation."""

    freshness_timeout_sec: float = 60.0
    """Maximum age in seconds before a wearable reading is considered stale."""

    min_baseline_observations: int = 10
    """Minimum valid on-wrist observations required to establish a personal baseline."""

    unusual_elevation_z_score: float = 2.0
    """Z-score threshold above personal baseline mean indicating significant elevation."""

    resting_speed_threshold_mps: float = 0.8
    """Movement speed (m/s) below which unexpected elevated HR suggests resting arousal."""

    exertion_speed_threshold_mps: float = 1.2
    """Movement speed (m/s) above which elevated HR is concordant with physical exertion."""

    min_valid_bpm: int = 30
    """Physiological minimum valid heart rate in beats per minute."""

    max_valid_bpm: int = 250
    """Physiological maximum valid heart rate in beats per minute."""

    min_baseline_std: float = 3.0
    """Floor for baseline standard deviation to prevent division-by-zero or hypersensitivity."""

    disclaimer: str = (
        "Associational wearable heart-rate metric. Reflects physiological deviation "
        "relative to observed personal baseline and movement speed. Does NOT constitute "
        "medical monitoring, clinical panic detection, or ground-truth safety guarantees."
    )


DEFAULT_HR_CONFIG = HeartRateConfig()


# ==============================================================================
# STRUCTURED EVIDENCE OUTPUT
# ==============================================================================


@dataclass
class HeartRateEvidence:
    """Interpretable heart-rate evidence object for the decision layer."""

    status: str  # One of HeartRateEvidenceStatus
    response_pattern: str  # One of HeartRateResponsePattern
    current_hr_bpm: Optional[int]
    baseline_mean_bpm: Optional[float]
    baseline_std_bpm: Optional[float]
    z_score: Optional[float]
    is_corroborating_risk: bool  # True ONLY if VALID and ELEVATED_UNUSUAL_RESTING
    reading_age_sec: Optional[float]
    wear_state: Optional[str]
    baseline_sample_count: int
    reason: str
    disclaimer: str = DEFAULT_HR_CONFIG.disclaimer

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to a clean dictionary."""
        return asdict(self)


# ==============================================================================
# BASELINE ESTIMATION
# ==============================================================================


def compute_baseline_from_observations(
    records: List[Dict[str, Any]],
    config: HeartRateConfig = DEFAULT_HR_CONFIG,
) -> Optional[Tuple[float, float, int]]:
    """Compute personal baseline (mean_bpm, std_bpm, sample_count) from historical records.

    Filters strictly for:
    - wear_state == 'ON_WRIST'
    - min_valid_bpm <= heart_rate_bpm <= max_valid_bpm

    Returns None if valid on-wrist observations < min_baseline_observations.
    """
    valid_bpms: List[float] = []

    for r in records:
        w_state = str(r.get("wear_state", "")).upper()
        if w_state != "ON_WRIST":
            continue

        hr = r.get("heart_rate_bpm")
        if hr is None:
            continue

        try:
            hr_val = float(hr)
            if config.min_valid_bpm <= hr_val <= config.max_valid_bpm:
                valid_bpms.append(hr_val)
        except (ValueError, TypeError):
            continue

    if len(valid_bpms) < config.min_baseline_observations:
        return None

    arr = np.array(valid_bpms, dtype=np.float64)
    mean_val = float(np.mean(arr))
    raw_std = float(np.std(arr))
    std_val = max(raw_std, config.min_baseline_std)

    return (round(mean_val, 1), round(std_val, 2), len(valid_bpms))


def load_user_baseline_from_db(
    user_id: str,
    db_path: Optional[str] = None,
    config: HeartRateConfig = DEFAULT_HR_CONFIG,
) -> Optional[Tuple[float, float, int]]:
    """Load and compute personal baseline synchronously from SQLite wearable_readings table.

    Safely handles missing DB, missing tables, or insufficient historical observations.
    """
    if db_path is None:
        db_env = os.getenv("DATABASE_PATH")
        if db_env:
            target_path = Path(db_env)
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
            target_path = repo_root / "backend" / "app" / "data" / "app.db"
    else:
        target_path = Path(db_path)

    if not target_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(target_path), timeout=2.0)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT heart_rate_bpm, wear_state
            FROM wearable_readings
            WHERE user_id = ? AND UPPER(wear_state) = 'ON_WRIST'
            ORDER BY timestamp DESC
            LIMIT 500
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        records = [{"heart_rate_bpm": r[0], "wear_state": r[1]} for r in rows]
        return compute_baseline_from_observations(records, config=config)
    except Exception as exc:
        logger.debug(f"Could not load personal HR baseline from DB for user {user_id}: {exc}")
        return None


# ==============================================================================
# HEART RATE EVALUATION ENGINE
# ==============================================================================


def evaluate_heart_rate(
    reading: Optional[Union[Dict[str, Any], Any]],
    current_speed_mps: float = 0.0,
    baseline: Optional[Tuple[float, float, int]] = None,
    current_timestamp: Optional[Union[datetime, str, float]] = None,
    config: HeartRateConfig = DEFAULT_HR_CONFIG,
) -> HeartRateEvidence:
    """Evaluate wearable heart rate reading against personal baseline and concurrent movement intensity.

    Guarantees:
    - Missing reading -> MISSING_DATA (UNAVAILABLE, is_corroborating_risk=False).
    - Off-wrist reading -> OFF_WRIST (UNAVAILABLE, is_corroborating_risk=False).
    - Stale reading (> 60s) -> STALE_READING (UNAVAILABLE, is_corroborating_risk=False).
    - Insufficient baseline (< 10 points) -> INSUFFICIENT_BASELINE (UNAVAILABLE, is_corroborating_risk=False).
    - Elevated HR + Walking/Running (>= 1.2 m/s) -> EXERTION_CONCORDANT (is_corroborating_risk=False).
    - Elevated HR + Sedentary (< 0.8 m/s) with z >= 2.0 -> ELEVATED_UNUSUAL_RESTING (is_corroborating_risk=True).
    - Normal HR (z < 2.0) -> WITHIN_NORMAL_BASELINE (is_corroborating_risk=False).
    """
    # 1. Missing reading check
    if reading is None:
        return HeartRateEvidence(
            status=HeartRateEvidenceStatus.MISSING_DATA.value,
            response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
            current_hr_bpm=None,
            baseline_mean_bpm=baseline[0] if baseline else None,
            baseline_std_bpm=baseline[1] if baseline else None,
            z_score=None,
            is_corroborating_risk=False,
            reading_age_sec=None,
            wear_state=None,
            baseline_sample_count=baseline[2] if baseline else 0,
            reason="No wearable reading provided.",
            disclaimer=config.disclaimer,
        )

    # Normalize dict access for Pydantic models or dicts
    if hasattr(reading, "model_dump"):
        r_dict = reading.model_dump(mode="python")
    elif isinstance(reading, dict):
        r_dict = reading
    else:
        r_dict = {
            "heart_rate_bpm": getattr(reading, "heart_rate_bpm", None),
            "wear_state": getattr(reading, "wear_state", None),
            "timestamp": getattr(reading, "timestamp", None),
        }

    # 2. Extract values
    raw_hr = r_dict.get("heart_rate_bpm")
    raw_wear = r_dict.get("wear_state")
    wear_state_str = str(getattr(raw_wear, "value", raw_wear) or "UNKNOWN").upper()
    reading_ts = r_dict.get("timestamp")

    # 3. Off-wrist check
    if wear_state_str != "ON_WRIST":
        return HeartRateEvidence(
            status=HeartRateEvidenceStatus.OFF_WRIST.value,
            response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
            current_hr_bpm=int(raw_hr) if raw_hr is not None else None,
            baseline_mean_bpm=baseline[0] if baseline else None,
            baseline_std_bpm=baseline[1] if baseline else None,
            z_score=None,
            is_corroborating_risk=False,
            reading_age_sec=None,
            wear_state=wear_state_str,
            baseline_sample_count=baseline[2] if baseline else 0,
            reason=f"Wearable sensor is {wear_state_str}; physiological evidence unavailable.",
            disclaimer=config.disclaimer,
        )

    # 4. Valid BPM range check
    try:
        bpm = int(raw_hr)
    except (ValueError, TypeError):
        return HeartRateEvidence(
            status=HeartRateEvidenceStatus.INVALID_READING.value,
            response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
            current_hr_bpm=None,
            baseline_mean_bpm=baseline[0] if baseline else None,
            baseline_std_bpm=baseline[1] if baseline else None,
            z_score=None,
            is_corroborating_risk=False,
            reading_age_sec=None,
            wear_state=wear_state_str,
            baseline_sample_count=baseline[2] if baseline else 0,
            reason=f"Invalid heart rate value: {raw_hr}.",
            disclaimer=config.disclaimer,
        )

    if bpm < config.min_valid_bpm or bpm > config.max_valid_bpm:
        return HeartRateEvidence(
            status=HeartRateEvidenceStatus.INVALID_READING.value,
            response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
            current_hr_bpm=bpm,
            baseline_mean_bpm=baseline[0] if baseline else None,
            baseline_std_bpm=baseline[1] if baseline else None,
            z_score=None,
            is_corroborating_risk=False,
            reading_age_sec=None,
            wear_state=wear_state_str,
            baseline_sample_count=baseline[2] if baseline else 0,
            reason=f"Heart rate {bpm} bpm falls outside valid range [{config.min_valid_bpm}, {config.max_valid_bpm}].",
            disclaimer=config.disclaimer,
        )

    # 5. Timestamp freshness check
    age_sec: Optional[float] = None
    if reading_ts is not None:
        try:
            # Parse reading timestamp
            if isinstance(reading_ts, (int, float)):
                r_epoch = float(reading_ts)
            elif hasattr(reading_ts, "timestamp"):
                r_epoch = float(reading_ts.timestamp())
            else:
                import pandas as pd
                r_epoch = float(pd.to_datetime(reading_ts).timestamp())

            # Parse current timestamp
            if current_timestamp is None:
                c_epoch = datetime.now(timezone.utc).timestamp()
            elif isinstance(current_timestamp, (int, float)):
                c_epoch = float(current_timestamp)
            elif hasattr(current_timestamp, "timestamp"):
                c_epoch = float(current_timestamp.timestamp())
            else:
                import pandas as pd
                c_epoch = float(pd.to_datetime(current_timestamp).timestamp())

            age_sec = round(max(0.0, c_epoch - r_epoch), 1)

            # Check if reading is stale
            if age_sec > config.freshness_timeout_sec:
                return HeartRateEvidence(
                    status=HeartRateEvidenceStatus.STALE_READING.value,
                    response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
                    current_hr_bpm=bpm,
                    baseline_mean_bpm=baseline[0] if baseline else None,
                    baseline_std_bpm=baseline[1] if baseline else None,
                    z_score=None,
                    is_corroborating_risk=False,
                    reading_age_sec=age_sec,
                    wear_state=wear_state_str,
                    baseline_sample_count=baseline[2] if baseline else 0,
                    reason=(
                        f"Wearable reading is stale ({age_sec:.1f}s old > {config.freshness_timeout_sec}s limit). "
                        "Expired telemetry cannot serve as physiological evidence."
                    ),
                    disclaimer=config.disclaimer,
                )
        except Exception as exc:
            logger.debug(f"Timestamp parsing error in heart rate evaluation: {exc}")

    # 6. Personal Baseline Check
    if baseline is None or baseline[2] < config.min_baseline_observations:
        sample_cnt = baseline[2] if baseline else 0
        return HeartRateEvidence(
            status=HeartRateEvidenceStatus.INSUFFICIENT_BASELINE.value,
            response_pattern=HeartRateResponsePattern.UNAVAILABLE.value,
            current_hr_bpm=bpm,
            baseline_mean_bpm=None,
            baseline_std_bpm=None,
            z_score=None,
            is_corroborating_risk=False,
            reading_age_sec=age_sec,
            wear_state=wear_state_str,
            baseline_sample_count=sample_cnt,
            reason=(
                f"Insufficient personal baseline observations ({sample_cnt} valid on-wrist samples < "
                f"{config.min_baseline_observations} required). Physiological arousal cannot be assessed."
            ),
            disclaimer=config.disclaimer,
        )

    # 7. Valid personal baseline comparison
    mean_bpm, std_bpm, sample_cnt = baseline
    z_score = round(float((bpm - mean_bpm) / std_bpm), 2)
    speed = float(max(0.0, current_speed_mps))

    # Evaluate arousal vs exertion
    if z_score >= config.unusual_elevation_z_score:
        if speed < config.resting_speed_threshold_mps:
            pattern = HeartRateResponsePattern.ELEVATED_UNUSUAL_RESTING.value
            is_corroborating = True
            reason = (
                f"Significantly elevated heart rate ({bpm} bpm, z={z_score:+.2f}) observed during "
                f"low/sedentary movement ({speed:.2f} m/s < {config.resting_speed_threshold_mps} m/s). "
                f"Corroborates physiological arousal relative to personal baseline ({mean_bpm:.1f} ± {std_bpm:.1f} bpm)."
            )
        else:
            pattern = HeartRateResponsePattern.EXERTION_CONCORDANT.value
            is_corroborating = False
            reason = (
                f"Elevated heart rate ({bpm} bpm, z={z_score:+.2f}) is concordant with concurrent "
                f"physical movement speed ({speed:.2f} m/s). Expected cardiovascular exertion response."
            )
    else:
        pattern = HeartRateResponsePattern.WITHIN_NORMAL_BASELINE.value
        is_corroborating = False
        reason = (
            f"Heart rate ({bpm} bpm, z={z_score:+.2f}) is within expected personal baseline range "
            f"({mean_bpm:.1f} ± {std_bpm:.1f} bpm)."
        )

    return HeartRateEvidence(
        status=HeartRateEvidenceStatus.VALID.value,
        response_pattern=pattern,
        current_hr_bpm=bpm,
        baseline_mean_bpm=mean_bpm,
        baseline_std_bpm=std_bpm,
        z_score=z_score,
        is_corroborating_risk=is_corroborating,
        reading_age_sec=age_sec,
        wear_state=wear_state_str,
        baseline_sample_count=sample_cnt,
        reason=reason,
        disclaimer=config.disclaimer,
    )

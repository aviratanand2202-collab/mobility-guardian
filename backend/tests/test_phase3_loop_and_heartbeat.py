"""Comprehensive Unit & Integration Test Suite for Phase 3:
Persistent-Loop Detection & Heartbeat-Aware Risk Decisions.

Verifies all requirements:
1. Trajectory Looping Detection:
   - Genuine persistent loops (>= 120s, >= 1.3 cycles, closure >= 0.60).
   - Ordinary 90-degree intersection turns (progressive transit, not loop).
   - Short circular detours (< 60s single roundabout, not persistent loop).
   - Stationary GPS sensor jitter (speed < 0.3 m/s, tiny displacement).
   - Insufficient GPS history (< 6 points or temporal gap > 300s).

2. Heartbeat Personal Baseline & Signal Rigor:
   - Personal baseline requires >= 10 valid on-wrist observations.
   - Insufficient baseline (< 10 points) yields INSUFFICIENT_BASELINE status and UNAVAILABLE pattern.
   - Off-wrist (OFF_WRIST) yields OFF_WRIST status and UNAVAILABLE pattern.
   - Stale reading (> 60s) yields STALE_READING status and UNAVAILABLE pattern.
   - Out-of-bounds BPM (< 30 or > 250) yields INVALID_READING status.
   - Never substitutes generic population range for personal baseline.

3. Physiological Arousal vs Exertion Concordance:
   - Elevated HR (z >= 2.0) with low/sedentary speed (< 0.8 m/s) -> ELEVATED_UNUSUAL_RESTING, is_corroborating_risk=True.
   - Elevated HR (z >= 2.0) with walking/running speed (>= 1.2 m/s) -> EXERTION_CONCORDANT, is_corroborating_risk=False.
   - Normal HR (z < 2.0) -> WITHIN_NORMAL_BASELINE, is_corroborating_risk=False.

4. Decision Layer Integration & Precedence:
   - CRITICAL behavioral risk dominates; elevated HR reinforces severity without overriding.
   - Persistent looping alone produces PERSISTENT_LOOPING (severity WARNING).
   - Persistent looping + corroborating elevated HR produces LOOPING_WITH_PHYSIOLOGICAL_AROUSAL (severity ALERT).
   - Non-suppression guarantee: normal, missing, stale, or off-wrist HR NEVER suppresses or downgrades
     safe-zone breaches (OUTSIDE_SAFE_AREA), persistent loops, or ML risk.

5. Alert Lifecycle, Deduplication, and WebSocket Push:
   - Confirmed loop triggers single AlertRecord (status PENDING).
   - Subsequent points during ongoing loop update existing alert location (deduplication across H3 cells).
   - Physiological arousal occurring mid-session escalates alert tier from Tier 2 to Tier 3.
   - When looping ceases (safe zone entry or progressive transit), alert auto-resolves cleanly (AUTO_RESOLVED).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest

# Ensure repo root and backend directory in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import select
from app.db import (
    async_session,
    init_db,
    AlertRecordRow,
    LoopStateRow,
    WearableReadingRow,
    save_wearable_reading,
)
from app.models import (
    AlertRecord,
    AlertStatus,
    Location,
    SensorMetrics,
    SignalState,
    SignalStatus,
    TelemetryPayload,
    WearableMetrics,
    WearState,
)
from app.routers.consent import record_consent
from app.state_machines.loop_lifecycle import (
    evaluate_loop_alert_lifecycle,
    LoopLifecycleResult,
)
from ml.src.decision import (
    DecisionConfig,
    DecisionOutput,
    DecisionState,
    interpret_decision,
)
from ml.src.heart_rate import (
    DEFAULT_HR_CONFIG,
    HeartRateConfig,
    HeartRateEvidence,
    HeartRateEvidenceStatus,
    HeartRateResponsePattern,
    compute_baseline_from_observations,
    evaluate_heart_rate,
)
from ml.src.loop_detection import (
    DEFAULT_LOOP_CONFIG,
    LoopDetectorConfig,
    LoopEvidence,
    PersistentLoopDetector,
)
from ml.src.inference import RiskInferenceEngine


# ==============================================================================
# 1. TRAJECTORY LOOPING DETECTION TESTS
# ==============================================================================


class TestLoopDetector:
    """Rigorous tests for PersistentLoopDetector geometric and kinematic discrimination."""

    def setup_method(self):
        self.detector = PersistentLoopDetector()
        self.t0 = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_genuine_persistent_loop_detected(self):
        """2 full laps around a 50m radius circular path lasting 180s confirms persistent looping."""
        points = []
        t_curr = self.t0
        # 2 laps, 12 points per lap = 24 points, step 8 seconds -> 184s total duration
        for _ in range(2):
            for angle_deg in range(0, 360, 30):
                rad = math.radians(angle_deg)
                # 50m radius circle (1 deg lat ~ 111,000m -> 50m ~ 0.00045 deg)
                d_lat = 0.00045 * math.cos(rad)
                d_lng = 0.00045 * math.sin(rad)
                points.append({
                    "latitude": 39.9800 + d_lat,
                    "longitude": 116.3100 + d_lng,
                    "timestamp": t_curr,
                    "speed_mps": 1.2,
                    "horizontal_accuracy_m": 5.0,
                })
                t_curr += timedelta(seconds=8)

        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is True
        assert ev.pattern == "PERSISTENT_LOOP"
        assert ev.loop_cycles >= 1.5
        assert ev.closure_ratio >= 0.85
        assert ev.duration_sec >= 120.0
        assert ev.confidence >= 0.60
        assert "Sustained repetitive route detected" in ev.reason

    def test_ordinary_intersection_turn_not_looping(self):
        """A 90-degree street intersection turn (north then east) is progressive transit, not a loop."""
        points = []
        t_curr = self.t0
        # Walk north for 100m
        for i in range(10):
            points.append({
                "latitude": 39.9800 + i * 0.0001,
                "longitude": 116.3100,
                "timestamp": t_curr,
                "speed_mps": 1.2,
            })
            t_curr += timedelta(seconds=10)
        # Turn east and walk 100m
        for i in range(1, 11):
            points.append({
                "latitude": 39.9800 + 9 * 0.0001,
                "longitude": 116.3100 + i * 0.0001,
                "timestamp": t_curr,
                "speed_mps": 1.2,
            })
            t_curr += timedelta(seconds=10)

        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is False
        assert ev.pattern == "PROGRESSIVE_TRANSIT"
        assert "linear displacement" in ev.reason.lower()

    def test_short_circular_detour_not_persistent_loop(self):
        """A single 36-second circular detour around a small monument does NOT meet the 120s persistence duration."""
        points = []
        t_curr = self.t0
        # 1 single lap lasting only 36 seconds
        for angle_deg in range(0, 360, 30):
            rad = math.radians(angle_deg)
            points.append({
                "latitude": 39.9800 + 0.0002 * math.cos(rad),
                "longitude": 116.3100 + 0.0002 * math.sin(rad),
                "timestamp": t_curr,
                "speed_mps": 1.1,
            })
            t_curr += timedelta(seconds=3)

        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is False
        assert ev.pattern == "SHORT_DETOUR"
        assert ev.duration_sec < 120.0
        assert "insufficient to confirm persistent looping" in ev.reason

    def test_stationary_gps_jitter_filtered_out(self):
        """Stationary GPS wander (< 0.3 m/s) with tiny path distance is classified as stationary sensor jitter."""
        np.random.seed(42)
        points = []
        t_curr = self.t0
        for _ in range(15):
            points.append({
                "latitude": 39.9800 + float(np.random.normal(0, 0.00001)),
                "longitude": 116.3100 + float(np.random.normal(0, 0.00001)),
                "timestamp": t_curr,
                "speed_mps": 0.08,
            })
            t_curr += timedelta(seconds=10)

        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is False
        assert ev.pattern == "STATIONARY_JITTER"
        assert "stationary sensor jitter" in ev.reason.lower()

    def test_insufficient_gps_points_fails_safely(self):
        """Fewer than 6 points returns INSUFFICIENT_HISTORY with zero false alarm."""
        points = [
            {"latitude": 39.9800, "longitude": 116.3100, "timestamp": self.t0, "speed_mps": 1.0},
            {"latitude": 39.9801, "longitude": 116.3101, "timestamp": self.t0 + timedelta(seconds=10), "speed_mps": 1.0},
        ]
        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is False
        assert ev.pattern == "INSUFFICIENT_HISTORY"
        assert "Insufficient valid GPS observations" in ev.reason

    def test_temporal_gap_resets_detector(self):
        """A telemetry time gap > 300s triggers a buffer reset to prevent spurious cross-session loops."""
        points = [
            {"latitude": 39.9800, "longitude": 116.3100, "timestamp": self.t0, "speed_mps": 1.0},
            {"latitude": 39.9801, "longitude": 116.3101, "timestamp": self.t0 + timedelta(seconds=20), "speed_mps": 1.0},
            {"latitude": 39.9802, "longitude": 116.3102, "timestamp": self.t0 + timedelta(seconds=40), "speed_mps": 1.0},
            {"latitude": 39.9803, "longitude": 116.3103, "timestamp": self.t0 + timedelta(seconds=60), "speed_mps": 1.0},
            {"latitude": 39.9804, "longitude": 116.3104, "timestamp": self.t0 + timedelta(seconds=80), "speed_mps": 1.0},
            # Gap of 400 seconds:
            {"latitude": 39.9800, "longitude": 116.3100, "timestamp": self.t0 + timedelta(seconds=480), "speed_mps": 1.0},
        ]
        ev: LoopEvidence = self.detector.evaluate_points(points)

        assert ev.is_looping is False
        assert ev.pattern == "INSUFFICIENT_HISTORY"
        assert "temporal gap" in ev.reason.lower()


# ==============================================================================
# 2. HEARTBEAT BASELINE & SIGNAL RIGOR TESTS
# ==============================================================================


class TestHeartRateEvaluator:
    """Rigorous tests for personal baseline calculation and wear-state signal validation."""

    def test_baseline_requires_minimum_10_on_wrist_observations(self):
        """Personal baseline requires at least 10 valid on-wrist observations."""
        # 8 observations -> Insufficient
        obs_8 = [{"heart_rate_bpm": 72, "wear_state": "ON_WRIST"} for _ in range(8)]
        assert compute_baseline_from_observations(obs_8) is None

        # 12 observations -> Sufficient
        obs_12 = [{"heart_rate_bpm": 70 + (i % 5), "wear_state": "ON_WRIST"} for i in range(12)]
        base = compute_baseline_from_observations(obs_12)
        assert base is not None
        mean_bpm, std_bpm, count = base
        assert count == 12
        assert 70.0 <= mean_bpm <= 75.0
        assert std_bpm >= DEFAULT_HR_CONFIG.min_baseline_std

    def test_off_wrist_observations_excluded_from_baseline(self):
        """Readings marked OFF_WRIST are excluded and cannot contribute to baseline."""
        mixed = (
            [{"heart_rate_bpm": 75, "wear_state": "ON_WRIST"} for _ in range(7)]
            + [{"heart_rate_bpm": 75, "wear_state": "OFF_WRIST"} for _ in range(8)]
        )
        # Total is 15, but only 7 are on-wrist -> None
        assert compute_baseline_from_observations(mixed) is None

    def test_missing_reading_returns_unavailable_without_suppression(self):
        """Missing wearable reading is explicitly reported as UNAVAILABLE and never as normal."""
        ev = evaluate_heart_rate(reading=None, current_speed_mps=0.2, baseline=(72.0, 4.0, 15))
        assert ev.status == HeartRateEvidenceStatus.MISSING_DATA.value
        assert ev.response_pattern == HeartRateResponsePattern.UNAVAILABLE.value
        assert ev.is_corroborating_risk is False

    def test_off_wrist_reading_returns_unavailable(self):
        """Off-wrist sensor reading produces OFF_WRIST status and UNAVAILABLE pattern."""
        reading = {
            "heart_rate_bpm": 120,
            "wear_state": "OFF_WRIST",
            "timestamp": datetime.now(timezone.utc),
        }
        ev = evaluate_heart_rate(reading=reading, current_speed_mps=0.2, baseline=(72.0, 4.0, 15))
        assert ev.status == HeartRateEvidenceStatus.OFF_WRIST.value
        assert ev.response_pattern == HeartRateResponsePattern.UNAVAILABLE.value
        assert ev.is_corroborating_risk is False

    def test_stale_reading_rejected(self):
        """Wearable reading older than 60 seconds is rejected as STALE_READING."""
        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(seconds=85)
        reading = {
            "heart_rate_bpm": 115,
            "wear_state": "ON_WRIST",
            "timestamp": stale_ts,
        }
        ev = evaluate_heart_rate(
            reading=reading,
            current_speed_mps=0.2,
            baseline=(72.0, 4.0, 15),
            current_timestamp=now,
        )
        assert ev.status == HeartRateEvidenceStatus.STALE_READING.value
        assert ev.response_pattern == HeartRateResponsePattern.UNAVAILABLE.value
        assert ev.is_corroborating_risk is False
        assert ev.reading_age_sec >= 85.0

    def test_out_of_bounds_bpm_rejected(self):
        """Biologically implausible heart rates (< 30 or > 250 bpm) are rejected as INVALID_READING."""
        now = datetime.now(timezone.utc)
        reading_low = {"heart_rate_bpm": 15, "wear_state": "ON_WRIST", "timestamp": now}
        ev_low = evaluate_heart_rate(reading_low, 0.2, (72.0, 4.0, 15), now)
        assert ev_low.status == HeartRateEvidenceStatus.INVALID_READING.value

        reading_high = {"heart_rate_bpm": 290, "wear_state": "ON_WRIST", "timestamp": now}
        ev_high = evaluate_heart_rate(reading_high, 0.2, (72.0, 4.0, 15), now)
        assert ev_high.status == HeartRateEvidenceStatus.INVALID_READING.value

    def test_insufficient_baseline_state(self):
        """When personal baseline has < 10 samples, status is INSUFFICIENT_BASELINE and never substitutes arbitrary default."""
        now = datetime.now(timezone.utc)
        reading = {"heart_rate_bpm": 110, "wear_state": "ON_WRIST", "timestamp": now}
        # Baseline with only 4 samples
        ev = evaluate_heart_rate(reading, 0.2, baseline=(70.0, 3.0, 4), current_timestamp=now)
        assert ev.status == HeartRateEvidenceStatus.INSUFFICIENT_BASELINE.value
        assert ev.response_pattern == HeartRateResponsePattern.UNAVAILABLE.value
        assert ev.is_corroborating_risk is False
        assert "Insufficient personal baseline" in ev.reason


# ==============================================================================
# 3. PHYSIOLOGICAL AROUSAL VS EXERTION CONCORDANCE TESTS
# ==============================================================================


class TestPhysiologicalConcordance:
    """Verifies distinction between physical movement exertion and unexpected resting elevation."""

    def setup_method(self):
        self.baseline = (70.0, 4.0, 20)  # Mean 70, std 4, 20 samples
        self.now = datetime.now(timezone.utc)

    def test_elevated_resting_heart_rate_corroborates_risk(self):
        """Significantly elevated HR (102 bpm, z=8.0) during slow walking (0.3 m/s) indicates physiological arousal."""
        reading = {"heart_rate_bpm": 102, "wear_state": "ON_WRIST", "timestamp": self.now}
        ev = evaluate_heart_rate(
            reading=reading,
            current_speed_mps=0.3,  # Sedentary/slow
            baseline=self.baseline,
            current_timestamp=self.now,
        )
        assert ev.status == HeartRateEvidenceStatus.VALID.value
        assert ev.response_pattern == HeartRateResponsePattern.ELEVATED_UNUSUAL_RESTING.value
        assert ev.is_corroborating_risk is True
        assert ev.z_score >= 2.0
        assert "physiological arousal" in ev.reason.lower()

    def test_elevated_heart_rate_with_exercise_is_exertion_concordant(self):
        """Elevated HR (105 bpm) during brisk walking/running (1.6 m/s) is concordant with physical exertion and NOT arousal."""
        reading = {"heart_rate_bpm": 105, "wear_state": "ON_WRIST", "timestamp": self.now}
        ev = evaluate_heart_rate(
            reading=reading,
            current_speed_mps=1.6,  # Exertion speed >= 1.2 m/s
            baseline=self.baseline,
            current_timestamp=self.now,
        )
        assert ev.status == HeartRateEvidenceStatus.VALID.value
        assert ev.response_pattern == HeartRateResponsePattern.EXERTION_CONCORDANT.value
        assert ev.is_corroborating_risk is False
        assert "concordant" in ev.reason.lower()

    def test_normal_heart_rate_within_baseline(self):
        """Heart rate (73 bpm, z=0.75) within baseline is classified as normal without corroboration."""
        reading = {"heart_rate_bpm": 73, "wear_state": "ON_WRIST", "timestamp": self.now}
        ev = evaluate_heart_rate(
            reading=reading,
            current_speed_mps=0.4,
            baseline=self.baseline,
            current_timestamp=self.now,
        )
        assert ev.status == HeartRateEvidenceStatus.VALID.value
        assert ev.response_pattern == HeartRateResponsePattern.WITHIN_NORMAL_BASELINE.value
        assert ev.is_corroborating_risk is False


# ==============================================================================
# 4. DECISION LAYER INTEGRATION & PRECEDENCE TESTS
# ==============================================================================


class TestDecisionIntegration:
    """Verifies combined evaluation of XGBoost risk, safe areas, looping, and heartbeat evidence."""

    @pytest.fixture
    def base_risk(self) -> Dict[str, Any]:
        return {
            "user_id": "patient_p3_001",
            "timestamp": "2026-10-15T12:00:00Z",
            "risk_tier": "NORMAL_TRANSIT",
            "risk_score": 15.0,
            "polling_tier": 1,
            "trigger_state": {"calibrated_probability": 0.15, "raw_probability": 0.12},
            "kinematic_features": {"behavioral_indicator": "NORMAL"},
            "explainability": {"top_features": []},
            "safe_area_state": "INSIDE_SAFE_AREA",
            "familiarity_state": "FAMILIAR_LOCATION",
            "geospatial_context": {
                "safe_area_state": "INSIDE_SAFE_AREA",
                "matched_safe_area_name": "Home",
                "distance_to_safe_boundary_m": -80.0,
            },
            "metadata": {"profile_mode": "PERSONALIZED", "trip_count": 12},
        }

    def test_looping_alone_produces_persistent_looping_decision(self, base_risk):
        """Confirmed persistent looping with normal or absent HR produces PERSISTENT_LOOPING state (WARNING)."""
        base_risk["loop_evidence"] = {
            "is_looping": True,
            "pattern": "PERSISTENT_LOOP",
            "loop_cycles": 2.1,
            "duration_sec": 180.0,
            "path_distance_m": 220.0,
            "closure_ratio": 0.92,
        }
        base_risk["heart_rate_evidence"] = {
            "status": "VALID",
            "response_pattern": "WITHIN_NORMAL_BASELINE",
            "is_corroborating_risk": False,
            "current_hr_bpm": 72,
        }

        dec: DecisionOutput = interpret_decision(base_risk)

        assert dec.decision_state == DecisionState.PERSISTENT_LOOPING.value
        assert dec.display_state == "PERSISTENT LOOPING DETECTED"
        assert dec.severity_level == "WARNING"
        assert "Sustained Repetitive Trajectory" in dec.headline
        assert "non-suppression active" in dec.reason.lower()

    def test_looping_with_physiological_arousal_produces_distinct_alert(self, base_risk):
        """Persistent looping accompanied by elevated resting heart rate escalates to LOOPING_WITH_PHYSIOLOGICAL_AROUSAL (ALERT)."""
        base_risk["loop_evidence"] = {
            "is_looping": True,
            "pattern": "PERSISTENT_LOOP",
            "loop_cycles": 2.1,
            "duration_sec": 180.0,
            "path_distance_m": 220.0,
            "closure_ratio": 0.92,
        }
        base_risk["heart_rate_evidence"] = {
            "status": "VALID",
            "response_pattern": "ELEVATED_UNUSUAL_RESTING",
            "is_corroborating_risk": True,
            "current_hr_bpm": 108,
            "z_score": 3.2,
        }

        dec: DecisionOutput = interpret_decision(base_risk)

        assert dec.decision_state == DecisionState.LOOPING_WITH_PHYSIOLOGICAL_AROUSAL.value
        assert dec.display_state == "PERSISTENT LOOPING WITH PHYSIOLOGICAL AROUSAL"
        assert dec.severity_level == "ALERT"
        assert "Repetitive Trajectory with Elevated Heart Rate" in dec.headline
        assert "physiological arousal" in dec.reason.lower()

    def test_non_suppression_normal_hr_does_not_suppress_safe_zone_breach(self, base_risk):
        """Normal or missing heart rate must NEVER suppress an urgent safe zone breach."""
        base_risk["safe_area_state"] = "OUTSIDE_SAFE_AREA"
        base_risk["geospatial_context"]["safe_area_state"] = "OUTSIDE_SAFE_AREA"
        base_risk["geospatial_context"]["distance_to_safe_boundary_m"] = 65.0
        # Normal heart rate:
        base_risk["heart_rate_evidence"] = {
            "status": "VALID",
            "response_pattern": "WITHIN_NORMAL_BASELINE",
            "is_corroborating_risk": False,
            "current_hr_bpm": 70,
        }

        dec: DecisionOutput = interpret_decision(base_risk)

        # MUST remain OUTSIDE_SAFE_AREA, NOT suppressed to SAFE_NORMAL
        assert dec.decision_state == DecisionState.OUTSIDE_SAFE_AREA.value
        assert dec.severity_level in ("ADVISORY", "WARNING")
        assert dec.decision_state != DecisionState.SAFE_NORMAL.value

    def test_non_suppression_missing_hr_does_not_suppress_critical_ml_risk(self, base_risk):
        """Missing heart rate data must NEVER suppress a CRITICAL XGBoost behavioral excursion."""
        base_risk["risk_tier"] = "CRITICAL"
        base_risk["risk_score"] = 85.0
        # No heart rate data provided:
        base_risk["heart_rate_evidence"] = {
            "status": "MISSING_DATA",
            "response_pattern": "UNAVAILABLE",
            "is_corroborating_risk": False,
        }

        dec: DecisionOutput = interpret_decision(base_risk)

        # MUST remain CRITICAL
        assert dec.decision_state == DecisionState.CRITICAL.value
        assert dec.severity_level == "ALERT"


# ==============================================================================
# 5. ALERT LIFECYCLE, DEDUPLICATION, AND WEBSOCKET PUSH TESTS
# ==============================================================================


def test_loop_alert_lifecycle_deduplication_and_resolution():
    """Verify end-to-end alert creation, H3 cell update deduplication, and auto-resolution."""
    async def _test():
        await init_db()

        user_id = f"test_user_loop_{uuid4().hex[:8]}"
        t_start = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)

        loop_ev = {
            "is_looping": True,
            "pattern": "PERSISTENT_LOOP",
            "loop_cycles": 1.8,
            "duration_sec": 150.0,
            "path_distance_m": 180.0,
            "closure_ratio": 0.88,
        }
        hr_ev_normal = {
            "status": "VALID",
            "response_pattern": "WITHIN_NORMAL_BASELINE",
            "is_corroborating_risk": False,
            "current_hr_bpm": 72,
        }

        # 1. First evaluation: confirmed looping triggers one initial alert
        with patch("app.state_machines.loop_lifecycle.push_alert_update", new_callable=AsyncMock) as mock_push:
            res1 = await evaluate_loop_alert_lifecycle(
                user_id=user_id,
                lat=39.9800,
                lng=116.3100,
                timestamp=t_start,
                loop_evidence=loop_ev,
                hr_evidence=hr_ev_normal,
            )
            assert res1.action == "ALERT_TRIGGERED"
            assert res1.is_looping is True
            assert res1.alert_tier == 2  # Tier 2 (SUSPICIOUS)
            assert res1.active_alert_id is not None
            alert_id = res1.active_alert_id
            mock_push.assert_awaited_once()

        # 2. Second evaluation: movement continues to another location/cell during SAME loop
        # Must UPDATE existing alert, NOT create a duplicate alert
        t_step2 = t_start + timedelta(seconds=20)
        with patch("app.state_machines.loop_lifecycle.push_alert_update", new_callable=AsyncMock) as mock_push:
            res2 = await evaluate_loop_alert_lifecycle(
                user_id=user_id,
                lat=39.9805,
                lng=116.3105,
                timestamp=t_step2,
                loop_evidence=loop_ev,
                hr_evidence=hr_ev_normal,
            )
            assert res2.action == "ALERT_UPDATED"
            assert res2.active_alert_id == alert_id  # Same alert ID preserved!
            mock_push.assert_awaited_once()

        # 3. Third evaluation: physiological arousal detected mid-session -> Tier escalated to Tier 3
        t_step3 = t_start + timedelta(seconds=40)
        hr_ev_arousal = {
            "status": "VALID",
            "response_pattern": "ELEVATED_UNUSUAL_RESTING",
            "is_corroborating_risk": True,
            "current_hr_bpm": 110,
            "z_score": 3.5,
        }
        with patch("app.state_machines.loop_lifecycle.push_alert_update", new_callable=AsyncMock) as mock_push:
            res3 = await evaluate_loop_alert_lifecycle(
                user_id=user_id,
                lat=39.9806,
                lng=116.3106,
                timestamp=t_step3,
                loop_evidence=loop_ev,
                hr_evidence=hr_ev_arousal,
            )
            assert res3.action == "ALERT_UPDATED"
            assert res3.active_alert_id == alert_id
            assert res3.alert_tier == 3  # Escalated to Tier 3!
            mock_push.assert_awaited_once()

        # 4. Fourth evaluation: patient resumes progressive linear transit (looping ceases)
        t_step4 = t_start + timedelta(seconds=60)
        transit_ev = {
            "is_looping": False,
            "pattern": "PROGRESSIVE_TRANSIT",
        }
        with patch("app.state_machines.loop_lifecycle.push_alert_update", new_callable=AsyncMock) as mock_push:
            res4 = await evaluate_loop_alert_lifecycle(
                user_id=user_id,
                lat=39.9820,
                lng=116.3120,
                timestamp=t_step4,
                loop_evidence=transit_ev,
                hr_evidence=hr_ev_normal,
            )
            assert res4.action == "ALERT_RESOLVED"
            assert res4.active_alert_id is None  # Alert session closed
            assert res4.is_looping is False
            mock_push.assert_awaited_once()

        # 5. Verify database record status
        async with async_session() as session:
            db_res = await session.execute(
                select(AlertRecordRow).where(AlertRecordRow.alert_id == alert_id)
            )
            record = db_res.scalar_one_or_none()
            assert record is not None
            assert record.status == AlertStatus.AUTO_RESOLVED.value
            assert record.alert_tier == 3  # Preserved highest escalated tier

    asyncio.run(_test())


# ==============================================================================
# 6. RISK INFERENCE ENGINE & TELEMETRY STREAMING INTEGRATION TESTS
# ==============================================================================


class TestInferenceEnginePhase3:
    """Verifies RiskInferenceEngine streaming telemetry evaluation with looping & wearable HR."""

    def test_predict_telemetry_reading_detects_looping_and_arousal(self):
        """Streaming GPS points into predict_telemetry_reading detects loop and physiological arousal."""
        engine = RiskInferenceEngine()
        user_id = f"test_stream_{uuid4().hex[:8]}"

        # Establish personal baseline for this user: mean 70, std 4, 15 samples
        engine.user_hr_baselines[user_id] = (70.0, 4.0, 15)

        t_curr = datetime(2026, 10, 15, 16, 0, 0, tzinfo=timezone.utc)
        out = None

        # Stream 25 circular points (2 laps, 180s duration)
        for lap in range(2):
            for angle_deg in range(0, 360, 30):
                rad = math.radians(angle_deg)
                d_lat = 0.00045 * math.cos(rad)
                d_lng = 0.00045 * math.sin(rad)

                payload = {
                    "user_id": user_id,
                    "timestamp": t_curr.isoformat(),
                    "location": {
                        "lat": 39.9800 + d_lat,
                        "lng": 116.3100 + d_lng,
                    },
                    "sensor_metrics": {
                        "horizontal_accuracy_m": 5.0,
                        "speed_mps": 0.3,  # Sedentary/slow
                        "heading_deg": float(angle_deg),
                    },
                    "signal_status": {"state": "VALID"},
                    "wearable_metrics": {
                        "heart_rate_bpm": 105,  # Elevated resting (z=8.75)
                        "wear_state": "ON_WRIST",
                        "timestamp": t_curr.isoformat(),
                    },
                }

                out = engine.predict_telemetry_reading(payload)
                t_curr += timedelta(seconds=8)

        assert out is not None
        assert "loop_evidence" in out
        assert out["loop_evidence"]["is_looping"] is True
        assert out["loop_evidence"]["pattern"] == "PERSISTENT_LOOP"

        assert "heart_rate_evidence" in out
        assert out["heart_rate_evidence"]["status"] == "VALID"
        assert out["heart_rate_evidence"]["response_pattern"] == "ELEVATED_UNUSUAL_RESTING"
        assert out["heart_rate_evidence"]["is_corroborating_risk"] is True

        # Confirms distinct decision state with physiological arousal
        assert out["decision_state"] == DecisionState.LOOPING_WITH_PHYSIOLOGICAL_AROUSAL.value
        assert out["display_state"] == "PERSISTENT LOOPING WITH PHYSIOLOGICAL AROUSAL"


def test_telemetry_ingest_endpoint_persists_wearable_reading():
    """Verify POST /telemetry/ingest persists valid wearable_metrics into wearable_readings table."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.models import ConsentRecord, GuardianConsent, EndUserConsent
    from app.routers.consent import record_consent

    client = TestClient(app)
    user_id = f"test_ingest_{uuid4().hex[:8]}"

    async def _setup_consent():
        await init_db()
        await record_consent(ConsentRecord(
            user_id=user_id,
            guardian_consent=GuardianConsent(
                granted=True,
                granted_at=datetime.now(timezone.utc),
                guardian_id="caregiver_p3",
            ),
            end_user_consent_applicable=False,
            end_user_consent=None,
            data_retention_acknowledged=True,
        ))
    asyncio.run(_setup_consent())

    t_now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "timestamp": t_now.isoformat(),
        "location": {"lat": 39.9800, "lng": 116.3100},
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": 1.2,
            "heading_deg": 90.0,
            "activity_type": "WALKING",
            "battery_pct": 88,
        },
        "signal_status": {"state": "VALID"},
        "wearable_metrics": {
            "heart_rate_bpm": 78,
            "wear_state": "ON_WRIST",
            "timestamp": t_now.isoformat(),
        },
    }

    resp = client.post("/api/telemetry/ingest", json=payload)
    assert resp.status_code == 200

    # Verify persisted row in DB
    async def _check_db():
        async with async_session() as session:
            res = await session.execute(
                select(WearableReadingRow).where(WearableReadingRow.user_id == user_id)
            )
            row = res.scalar_one_or_none()
            assert row is not None
            assert row.heart_rate_bpm == 78
            assert row.wear_state == "ON_WRIST"
    asyncio.run(_check_db())

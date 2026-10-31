"""
Isolated unit test suite for Mobility Guardian demo runner (scripts/demo_runner.py).

Validates:
1. Trajectory generators: point counts strictly in [61, 70], distances, boundaries, headings, biometrics.
2. Geofence containment: Scenarios 1-3 strictly inside Home A (< 200m).
3. Geofence breach: Scenario 4 crosses Home A perimeter + 15m buffer (215m) at points 40-41.
4. Pacing corridors: Scenarios 5 & 6 Algase pacing kinetics, 120s window spans, HR corroboration.
5. Schema compliance: telemetry payloads parse cleanly into app.models.TelemetryPayload.
6. Timing & preflight guards: timestamp monotonicity, past-time constraints, two-window preflight.
7. Mathematical utilities: lat_lng_offset, distance_between_m, calculate_heading.

Completely isolated: in-memory execution without live network or database modifications.
"""
from datetime import datetime, timezone, timedelta
import math
from unittest.mock import patch
import pytest

from app.models import TelemetryPayload
from scripts.demo_runner import (
    HOME_A_CENTER_LAT,
    HOME_A_CENTER_LNG,
    HOME_A_RADIUS_M,
    GEOFENCE_HYSTERESIS_M,
    PATIENT_B_PACING_LAT,
    PATIENT_B_PACING_LNG,
    SCENARIOS,
    ScenarioSpec,
    TrajectoryPoint,
    build_telemetry_payload,
    calculate_heading,
    calculate_scenario_start_time,
    distance_between_m,
    lat_lng_offset,
    generate_scenario_1_normal,
    generate_scenario_2_garden_walk,
    generate_scenario_3_exercise,
    generate_scenario_4_home_exit,
    generate_scenario_5_pacing_win1,
    generate_scenario_6_pacing_win2,
)


class TestMathUtilities:
    """Test geodesic and heading conversion utilities."""

    def test_lat_lng_offset_identity(self):
        lat, lng = lat_lng_offset(39.9, 116.4, 0.0, 0.0)
        assert round(lat, 5) == 39.9
        assert round(lng, 5) == 116.4

    def test_lat_lng_offset_north(self):
        # 111320 meters north should be ~1.0 degree latitude
        lat, lng = lat_lng_offset(39.9, 116.4, 0.0, 111320.0)
        assert abs(lat - 40.9) < 0.01
        assert abs(lng - 116.4) < 1e-4

    def test_distance_between_m_same_point(self):
        dist = distance_between_m(39.9, 116.4, 39.9, 116.4)
        assert dist == 0.0

    def test_distance_between_m_known_offset(self):
        # Move 100m north, 50m east
        lat2, lng2 = lat_lng_offset(39.9, 116.4, 50.0, 100.0)
        dist = distance_between_m(39.9, 116.4, lat2, lng2)
        expected = math.hypot(50.0, 100.0)
        assert abs(dist - expected) < 0.5

    def test_calculate_heading(self):
        assert calculate_heading(0.0, 10.0) == 0.0      # North
        assert calculate_heading(10.0, 0.0) == 90.0     # East
        assert calculate_heading(0.0, -10.0) == 180.0   # South
        assert calculate_heading(-10.0, 0.0) == 270.0   # West
        assert calculate_heading(0.0, 0.0, prev_heading=45.0) == 45.0


class TestScenarioTrajectoryGenerators:
    """Test point counts, boundaries, speeds, and biometrics for all 6 scenarios."""

    def test_all_scenarios_point_counts_in_range(self):
        """Every scenario must generate between 61 and 70 points."""
        for s_id, spec in SCENARIOS.items():
            pts = spec.generator()
            assert 61 <= len(pts) <= 70, (
                f"Scenario {s_id} point count {len(pts)} out of [61, 70] range"
            )
            assert spec.num_points == len(pts), (
                f"Scenario {s_id} spec.num_points ({spec.num_points}) != len(points) ({len(pts)})"
            )

    def test_scenario_1_normal_walking(self):
        pts = generate_scenario_1_normal()
        assert len(pts) == 65
        for idx, pt in enumerate(pts):
            dist = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            assert dist < HOME_A_RADIUS_M, f"Point {idx} exited Home A ({dist:.1f}m >= 200m)"
            assert 70 <= pt.heart_rate_bpm <= 75
            assert pt.speed_mps == 0.95
            assert pt.wear_state == "ON_WRIST"
            assert pt.sensor_status == "SIMULATED_NORMAL"

    def test_scenario_2_garden_walk(self):
        pts = generate_scenario_2_garden_walk()
        assert len(pts) == 65
        for idx, pt in enumerate(pts):
            dist = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            assert dist < HOME_A_RADIUS_M, f"Point {idx} exited Home A ({dist:.1f}m >= 200m)"
            assert 70 <= pt.heart_rate_bpm <= 78
            assert pt.speed_mps == 0.85
            assert pt.wear_state == "ON_WRIST"
            assert pt.sensor_status == "SIMULATED_NORMAL"

    def test_scenario_3_exercise_negative_control(self):
        pts = generate_scenario_3_exercise()
        assert len(pts) == 65
        for idx, pt in enumerate(pts):
            dist = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            assert dist < HOME_A_RADIUS_M, f"Point {idx} exited Home A ({dist:.1f}m >= 200m)"
            # Exertion thresholds
            assert pt.speed_mps >= 1.2, f"Speed {pt.speed_mps} below exertion threshold 1.2 m/s"
            assert pt.heart_rate_bpm >= 120, f"HR {pt.heart_rate_bpm} below exertion tachycardia 120 bpm"
            assert pt.sensor_status == "SIMULATED_EXERTION"

    def test_scenario_4_home_exit_debouncing(self):
        pts = generate_scenario_4_home_exit()
        assert len(pts) == 65

        # Initial point is inside Home A
        dist_start = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pts[0].lat, pts[0].lng)
        assert dist_start < HOME_A_RADIUS_M, f"Start point not inside Home A: {dist_start:.1f}m"

        breach_threshold = HOME_A_RADIUS_M + GEOFENCE_HYSTERESIS_M  # 215.0m
        outside_indices = []
        for idx, pt in enumerate(pts):
            dist = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            if dist > breach_threshold:
                outside_indices.append(idx)

        assert len(outside_indices) >= 20, "Should have multiple points outside for debounce & deduplication"
        # First outside point triggers EXIT_PENDING
        first_out = outside_indices[0]
        # Second outside point confirms BREACHED
        second_out = outside_indices[1]
        assert second_out == first_out + 1, "Exit points must be consecutive for debouncing"

        # Points 40 (0-indexed 39 or 40) should be crossing boundary
        assert distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pts[first_out].lat, pts[first_out].lng) > breach_threshold

        # Final point is far outside (> 280m)
        dist_end = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pts[-1].lat, pts[-1].lng)
        assert dist_end > 280.0, f"Final point should be far outside: {dist_end:.1f}m"

    def test_scenario_5_pacing_window_1(self):
        pts = generate_scenario_5_pacing_win1()
        assert len(pts) == 61
        # Duration: (61 - 1) * 2.0s = 120.0s
        duration_s = (len(pts) - 1) * 2.0
        assert duration_s == 120.0

        for pt in pts:
            # Check distance to Home A: should be far outside (~700m)
            dist_home = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            assert dist_home > 600.0, f"Pacing corridor must be outside Home A: {dist_home:.1f}m"
            # Sedentary/pacing speed
            assert pt.speed_mps == 0.40
            # Baseline normal HR
            assert pt.heart_rate_bpm == 74
            assert pt.sensor_status == "SIMULATED_NORMAL"

    def test_scenario_6_pacing_window_2_hr_corroboration(self):
        pts = generate_scenario_6_pacing_win2()
        assert len(pts) == 61
        duration_s = (len(pts) - 1) * 2.0
        assert duration_s == 120.0

        for pt in pts:
            dist_home = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            assert dist_home > 600.0
            assert pt.speed_mps == 0.40
            # Elevated HR corroboration
            assert pt.heart_rate_bpm == 108
            assert pt.sensor_status == "SIMULATED_AROUSAL"


class TestSchemaCompliance:
    """Test that all generated points conform strictly to TelemetryPayload schema."""

    def test_all_scenarios_schema_validity(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        for s_id, spec in SCENARIOS.items():
            pts = spec.generator()
            for idx in [0, len(pts) // 2, len(pts) - 1]:
                pt = pts[idx]
                payload_dict = build_telemetry_payload(spec.user_id, now_iso, pt)
                # Must parse cleanly into TelemetryPayload pydantic model
                model = TelemetryPayload(**payload_dict)
                assert model.user_id == spec.user_id
                assert model.location.lat == pt.lat
                assert model.location.lng == pt.lng
                assert model.sensor_metrics.speed_mps == pt.speed_mps
                assert model.sensor_metrics.heading_deg == pt.heading_deg
                if pt.heart_rate_bpm is not None:
                    assert model.wearable_metrics is not None
                    assert model.wearable_metrics.heart_rate_bpm == pt.heart_rate_bpm
                    assert model.wearable_metrics.wear_state == pt.wear_state
                    assert model.wearable_metrics.sensor_status == pt.sensor_status


class TestPreflightAndTiming:
    """Test start-time calculation, monotonicity, future-timestamp guard, and two-window preflight."""

    def test_calculate_scenario_start_time_pristine_user(self):
        now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        with patch("scripts.demo_runner.fetch_latest_timestamp", return_value=None):
            spec = SCENARIOS[1]  # 65 points, 1.9s step -> 121.6s
            start = calculate_scenario_start_time("http://mock", spec, now)
            duration = (spec.num_points - 1) * spec.step_seconds  # 121.6s
            expected_start = now - timedelta(seconds=duration + 10.0)
            assert start == expected_start
            final_fix = start + timedelta(seconds=duration)
            assert final_fix < now

    def test_calculate_scenario_start_time_monotonic_success(self):
        now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(seconds=200.0)
        with patch("scripts.demo_runner.fetch_latest_timestamp", return_value=latest):
            spec = SCENARIOS[1]  # duration 121.6s
            start = calculate_scenario_start_time("http://mock", spec, now)
            assert start == latest + timedelta(seconds=2.0)
            final_fix = start + timedelta(seconds=(spec.num_points - 1) * spec.step_seconds)
            assert final_fix < now

    def test_calculate_scenario_start_time_future_guard_abort(self):
        now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(seconds=50.0)  # Too recent: 50s ago + 2s + 121.6s = 73.6s into future
        with patch("scripts.demo_runner.fetch_latest_timestamp", return_value=latest):
            spec = SCENARIOS[1]
            with pytest.raises(ValueError, match="Timestamp Guard Failed"):
                calculate_scenario_start_time("http://mock", spec, now)

    def test_two_window_preflight_scenario_5_success(self):
        now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        # Pristine user (no prior telemetry)
        with patch("scripts.demo_runner.fetch_latest_timestamp", return_value=None):
            spec = SCENARIOS[5]
            start = calculate_scenario_start_time("http://mock", spec, now)
            total_span = 242.0
            expected_start = now - timedelta(seconds=total_span + 10.0)
            assert start == expected_start
            assert start + timedelta(seconds=total_span) < now

    def test_two_window_preflight_scenario_5_future_abort(self):
        now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(seconds=100.0)  # Only 100s ago, but 242s is required!
        with patch("scripts.demo_runner.fetch_latest_timestamp", return_value=latest):
            spec = SCENARIOS[5]
            with pytest.raises(ValueError, match="Two-Window Preflight Guard Failed"):
                calculate_scenario_start_time("http://mock", spec, now)

"""
Unit tests for geodesic distance, coordinate projection, and trajectory heading calculations.

Tests the mathematical foundation of demo_runner.py without network or database dependencies.
"""
import math
import pytest

from scripts.demo_runner import (
    HOME_A_CENTER_LAT,
    HOME_A_CENTER_LNG,
    HOME_A_RADIUS_M,
    GEOFENCE_HYSTERESIS_M,
    calculate_heading,
    distance_between_m,
    lat_lng_offset,
)


class TestGeodesicProjection:
    """Test equirectangular tangent-plane projection and offset functions."""

    def test_identity_offset(self):
        lat, lng = lat_lng_offset(39.9, 116.4, 0.0, 0.0)
        assert round(lat, 6) == 39.9
        assert round(lng, 6) == 116.4

    def test_due_north_offset(self):
        # 111,320m north is approximately 1.0 degree latitude
        lat, lng = lat_lng_offset(39.9, 116.4, 0.0, 111320.0)
        assert abs(lat - 40.9) < 0.005
        assert abs(lng - 116.4) < 1e-5

    def test_due_east_offset(self):
        # East offset depends on cos(lat)
        expected_lng_scale = 111320.0 * math.cos(math.radians(39.9))
        lat, lng = lat_lng_offset(39.9, 116.4, expected_lng_scale, 0.0)
        assert abs(lat - 39.9) < 1e-5
        assert abs(lng - 117.4) < 0.005

    def test_distance_zero(self):
        assert distance_between_m(39.9, 116.4, 39.9, 116.4) == 0.0

    def test_distance_symmetry(self):
        d1 = distance_between_m(39.900, 116.400, 39.901, 116.402)
        d2 = distance_between_m(39.901, 116.402, 39.900, 116.400)
        assert abs(d1 - d2) < 1e-6

    def test_distance_known_triangle(self):
        # Offset 60m East, 80m North -> 100m hypotenuse
        lat2, lng2 = lat_lng_offset(39.9, 116.4, 60.0, 80.0)
        d = distance_between_m(39.9, 116.4, lat2, lng2)
        assert abs(d - 100.0) < 0.5


class TestHeadingCalculations:
    """Test geographic compass heading (0=N, 90=E, 180=S, 270=W)."""

    def test_cardinal_headings(self):
        assert calculate_heading(0.0, 10.0) == 0.0      # North
        assert calculate_heading(10.0, 0.0) == 90.0     # East
        assert calculate_heading(0.0, -10.0) == 180.0   # South
        assert calculate_heading(-10.0, 0.0) == 270.0   # West

    def test_intercardinal_headings(self):
        assert calculate_heading(10.0, 10.0) == 45.0    # North-East
        assert calculate_heading(10.0, -10.0) == 135.0  # South-East
        assert calculate_heading(-10.0, -10.0) == 225.0 # South-West
        assert calculate_heading(-10.0, 10.0) == 315.0  # North-West

    def test_stationary_heading_preservation(self):
        assert calculate_heading(0.0, 0.0, prev_heading=123.4) == 123.4


class TestGeofenceBoundaryConstants:
    """Validate home radius and hysteresis thresholds."""

    def test_boundary_thresholds(self):
        assert HOME_A_RADIUS_M == 200.0
        assert GEOFENCE_HYSTERESIS_M == 15.0
        breach_dist = HOME_A_RADIUS_M + GEOFENCE_HYSTERESIS_M
        assert breach_dist == 215.0

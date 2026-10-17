"""Unit tests for GeoLife Replay CLI utility (backend/scripts/geolife_replay.py)."""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.models import TelemetryPayload
from scripts.geolife_replay import (
    calculate_bearing,
    derive_kinematic_features,
    haversine_distance,
    parse_plt_file,
    build_telemetry_payload,
    GeoLifePoint,
)


def _write_plt(path: Path, rows: list[str]) -> Path:
    """Helper to write GeoLife .plt file with standard 6-line header."""
    header = [
        "Geolife trajectory\n",
        "WGS 84\n",
        "Altitude is in Feet\n",
        "Reserved 3\n",
        "0,2,255,My Track,0,0,2,8421376\n",
        "0\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(header)
        for r in rows:
            f.write(r + "\n")
    return path


def test_parse_plt_valid_records(tmp_path: Path):
    """Verify standard GeoLife .plt file parses valid rows and converts altitude correctly."""
    rows = [
        "39.984702,116.318417,0,492,39744.12018,2008-10-23,02:53:04",
        "39.984683,116.318450,0,492,39744.12025,2008-10-23,02:53:10",
        "39.984686,116.318417,0,-777,39744.12031,2008-10-23,02:53:15",
    ]
    plt_file = _write_plt(tmp_path / "valid.plt", rows)

    points, diag = parse_plt_file(plt_file)
    assert diag["valid_count"] == 3
    assert diag["skipped_count"] == 0
    assert len(points) == 3

    p0 = points[0]
    assert p0.latitude == pytest.approx(39.984702)
    assert p0.longitude == pytest.approx(116.318417)
    assert p0.altitude_m == pytest.approx(492 * 0.3048, abs=0.1)
    assert p0.timestamp == datetime(2008, 10, 23, 2, 53, 4, tzinfo=timezone.utc)

    # -777 indicates uncalibrated altitude, should map to None
    p2 = points[2]
    assert p2.altitude_m is None


def test_parse_plt_skips_malformed_and_out_of_bounds(tmp_path: Path):
    """Verify malformed lines (short fields, bad dates, out-of-bounds coordinates) are safely skipped."""
    rows = [
        "39.984702,116.318417,0,492,39744.12018,2008-10-23,02:53:04",  # valid
        "95.000000,116.318417,0,492,39744.12025,2008-10-23,02:53:10",  # invalid latitude > 90
        "39.984683,200.000000,0,492,39744.12025,2008-10-23,02:53:10",  # invalid longitude > 180
        "short,row,only,four",                                          # invalid column count
        "39.984686,116.318417,0,492,39744.12031,not-a-date,02:53:15",   # invalid date string
        "39.984690,116.318420,0,492,39744.12037,2008-10-23,02:53:20",  # valid
    ]
    plt_file = _write_plt(tmp_path / "mixed.plt", rows)

    points, diag = parse_plt_file(plt_file)
    assert diag["valid_count"] == 2
    assert diag["skipped_count"] == 4
    assert len(points) == 2


def test_parse_plt_chronological_ordering(tmp_path: Path):
    """Verify points out of order in the file are sorted chronologically."""
    rows = [
        "39.980,116.310,0,100,39744.0,2008-10-23,02:53:20",
        "39.980,116.310,0,100,39744.0,2008-10-23,02:53:00",
        "39.980,116.310,0,100,39744.0,2008-10-23,02:53:10",
    ]
    plt_file = _write_plt(tmp_path / "unsorted.plt", rows)

    points, _ = parse_plt_file(plt_file)
    assert points[0].timestamp < points[1].timestamp < points[2].timestamp
    assert points[0].timestamp.second == 0
    assert points[1].timestamp.second == 10
    assert points[2].timestamp.second == 20


def test_haversine_and_bearing():
    """Verify mathematical distance and azimuth calculation."""
    # Beijing roughly 111km per degree latitude
    dist = haversine_distance(39.9800, 116.3100, 39.9890, 116.3100)
    assert 990.0 < dist < 1010.0  # ~1000m north

    bearing_north = calculate_bearing(39.9800, 116.3100, 39.9890, 116.3100)
    assert bearing_north == 0.0 or bearing_north == 360.0

    bearing_east = calculate_bearing(39.9800, 116.3100, 39.9800, 116.3200)
    assert 89.0 <= bearing_east <= 91.0


def test_derive_kinematic_features_live_and_original():
    """Verify speed calculation and timestamp shifting under original and live modes."""
    t0 = datetime(2008, 10, 23, 2, 53, 0, tzinfo=timezone.utc)
    raw = [
        GeoLifePoint(39.9800, 116.3100, 50.0, t0, 7),
        GeoLifePoint(39.9809, 116.3100, 50.0, t0 + timedelta(seconds=100), 8),  # ~100m in 100s -> ~1.0 m/s
    ]

    # Original mode
    orig = derive_kinematic_features(raw, time_mode="original")
    assert orig[0].timestamp == t0
    assert orig[1].timestamp == t0 + timedelta(seconds=100)
    assert orig[0].speed_mps == 0.0
    assert 0.9 <= orig[1].speed_mps <= 1.1

    # Live mode
    live_start = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)
    live = derive_kinematic_features(raw, time_mode="live", target_start_time=live_start)
    assert live[0].timestamp == live_start
    assert live[1].timestamp == live_start + timedelta(seconds=100)
    # Speed and kinematics remain completely identical
    assert live[1].speed_mps == orig[1].speed_mps
    assert live[1].heading_deg == orig[1].heading_deg


def test_build_telemetry_payload_schema_valid():
    """Verify constructed payload strictly satisfies TelemetryPayload Pydantic validation."""
    raw = [
        GeoLifePoint(39.9800, 116.3100, 45.0, datetime.now(timezone.utc), 7),
    ]
    enriched = derive_kinematic_features(raw)[0]
    payload_dict = build_telemetry_payload(enriched, user_id="test_schema_user", battery_pct=77)

    # Validate with Pydantic
    validated = TelemetryPayload.model_validate(payload_dict)
    assert validated.user_id == "test_schema_user"
    assert validated.location.lat == 39.98
    assert validated.location.lng == 116.31
    assert validated.location.altitude_m == 45.0
    assert validated.sensor_metrics.battery_pct == 77
    assert validated.signal_status.state.value == "VALID"

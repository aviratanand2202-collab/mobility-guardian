"""Unit tests for GeoLife Ingestion, Validation & Minimal Cleaning (Chunk 1).

These tests use small fixtures and do NOT depend on the full raw GeoLife dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src.data import (
    IngestionConfig,
    calculate_distribution_stats,
    discover_trajectory_files,
    parse_plt_file,
    process_geolife_dataset,
    validate_and_clean_trajectory,
)


def _create_sample_plt(
    file_path: Path,
    rows: list[tuple[float, float, int, float, float, str, str]],
) -> Path:
    """Helper to write a valid GeoLife-formatted .plt file with headers."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Geolife trajectory\n",
        "WGS 84\n",
        "Altitude is in Feet\n",
        "Reserved 3\n",
        "0,2,255,My Track,0,0,2,8421376\n",
        "0\n",
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(header)
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")
    return file_path


# -----------------------------------------------------------------------------
# 1. Dataset & File Discovery Tests
# -----------------------------------------------------------------------------


def test_discover_trajectory_files_dynamic(tmp_path: Path):
    """Test dynamic discovery of user folders and .plt files without hardcoded names."""
    user_alpha = tmp_path / "user_alpha" / "Trajectory"
    user_beta = tmp_path / "user_beta"
    user_empty = tmp_path / "user_empty"
    user_empty.mkdir(parents=True)

    _create_sample_plt(user_alpha / "20081023010101.plt", [(39.9, 116.3, 0, 100, 39700.1, "2008-10-23", "01:01:01")])
    _create_sample_plt(user_alpha / "20081023020202.plt", [(39.9, 116.3, 0, 100, 39700.2, "2008-10-23", "02:02:02")])
    _create_sample_plt(user_beta / "20081023030303.plt", [(39.8, 116.2, 0, 100, 39700.3, "2008-10-23", "03:03:03")])

    (user_alpha / "labels.txt").write_text("dummy labels", encoding="utf-8")

    discovered = discover_trajectory_files(tmp_path)
    assert len(discovered) == 3

    users = [u for u, _ in discovered]
    assert users == ["user_alpha", "user_alpha", "user_beta"]


def test_discover_trajectory_files_nonexistent_dir():
    """Test discovery raises FileNotFoundError on nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        discover_trajectory_files(Path("nonexistent/dir/path"))


# -----------------------------------------------------------------------------
# 2. PLT Parsing & Header Handling Tests
# -----------------------------------------------------------------------------


def test_parse_plt_file_header_handling(tmp_path: Path):
    """Test skipping metadata headers and extracting normalized DataFrame with raw fields."""
    plt_file = tmp_path / "000" / "Trajectory" / "20081023025304.plt"
    rows = [
        (39.984702, 116.318417, 0, 492.0, 39744.120185, "2008-10-23", "02:53:04"),
        (39.984683, 116.318450, 0, 492.0, 39744.120255, "2008-10-23", "02:53:10"),
    ]
    _create_sample_plt(plt_file, rows)

    df, diag = parse_plt_file(plt_file)
    assert diag["success"] is True
    assert diag["raw_points"] == 2
    assert df is not None
    assert len(df) == 2

    expected_cols = {
        "user_id",
        "trajectory_id",
        "timestamp",
        "latitude",
        "longitude",
        "altitude",
        "raw_days",
        "date_str",
        "time_str",
    }
    assert expected_cols.issubset(set(df.columns))

    assert df["user_id"].iloc[0] == "000"
    assert df["trajectory_id"].iloc[0] == "000_20081023025304"
    assert df["latitude"].iloc[0] == pytest.approx(39.984702)
    assert df["longitude"].iloc[0] == pytest.approx(116.318417)
    assert df["altitude"].iloc[0] == pytest.approx(492.0)
    assert df["timestamp"].iloc[0] == pd.Timestamp("2008-10-23 02:53:04")


def test_parse_plt_file_corrupt(tmp_path: Path):
    """Test handling of corrupt or unreadable PLT files."""
    bad_file = tmp_path / "bad.plt"
    bad_file.write_text("", encoding="utf-8")

    df, diag = parse_plt_file(bad_file)
    assert df is None
    assert diag["success"] is False
    assert diag["error"] is not None


# -----------------------------------------------------------------------------
# 3. Coordinate Validation Tests
# -----------------------------------------------------------------------------


def test_coordinate_validation():
    """Test latitude in [-90, 90] and longitude in [-180, 180] physical bounds."""
    timestamps = pd.date_range("2008-10-23 10:00:00", periods=5, freq="5s")
    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": timestamps,
            "latitude": [39.9, -91.0, 40.0, 90.1, 0.0],
            "longitude": [116.3, 116.3, 181.0, 100.0, -180.1],
            "altitude": [100.0, 100.0, 100.0, 100.0, 100.0],
            "raw_days": [39700.0] * 5,
            "date_str": ["2008-10-23"] * 5,
            "time_str": ["10:00:00"] * 5,
        }
    )

    clean_df, diag, _ = validate_and_clean_trajectory(df)
    assert diag["invalid_coordinates"] == 4
    assert len(clean_df) == 1
    assert clean_df["latitude"].iloc[0] == pytest.approx(39.9)


# -----------------------------------------------------------------------------
# 4. Timestamp Parsing & Uninitialized RTC Filtering Tests
# -----------------------------------------------------------------------------


def test_timestamp_parsing_invalid_handling():
    """Test invalid or unparseable timestamps are detected and removed."""
    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": [
                pd.Timestamp("2008-10-23 10:00:00"),
                pd.NaT,
                pd.Timestamp("2008-10-23 10:00:10"),
            ],
            "latitude": [39.9, 39.9, 39.9],
            "longitude": [116.3, 116.3, 116.3],
            "altitude": [100.0, 100.0, 100.0],
            "raw_days": [39700.0] * 3,
            "date_str": ["2008-10-23"] * 3,
            "time_str": ["10:00:00"] * 3,
        }
    )

    clean_df, diag, _ = validate_and_clean_trajectory(df)
    assert diag["unparseable_timestamps"] == 1
    assert len(clean_df) == 2


def test_uninitialized_rtc_timestamp_filtering():
    """Test pre-2005 hardware clock reset artifacts (e.g. 2000-01-01) are dropped."""
    df = pd.DataFrame(
        {
            "user_id": "163",
            "trajectory_id": "163_20000101231219",
            "timestamp": [
                pd.Timestamp("2000-01-01 23:12:19"),
                pd.Timestamp("2000-01-01 23:13:21"),
                pd.Timestamp("2008-10-23 10:00:00"),
            ],
            "latitude": [39.988, 39.990, 39.992],
            "longitude": [116.327, 116.327, 116.328],
            "altitude": [128.0, 221.0, 217.0],
            "raw_days": [36526.9, 36526.9, 39700.0],
            "date_str": ["2000-01-01", "2000-01-01", "2008-10-23"],
            "time_str": ["23:12:19", "23:13:21", "10:00:00"],
        }
    )

    clean_df, diag, _ = validate_and_clean_trajectory(df)
    assert diag["uninitialized_rtc_timestamps"] == 2
    assert len(clean_df) == 1
    assert clean_df["timestamp"].iloc[0] == pd.Timestamp("2008-10-23 10:00:00")


# -----------------------------------------------------------------------------
# 5. Chronological Validation & Sorting Tests
# -----------------------------------------------------------------------------


def test_chronological_ordering_validation():
    """Test out-of-order timestamps are detected and sorted chronologically."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:00:10")
    t2 = pd.Timestamp("2008-10-23 10:00:05")

    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": [t0, t1, t2],
            "latitude": [39.90, 39.92, 39.91],
            "longitude": [116.30, 116.32, 116.31],
            "altitude": [100.0, 120.0, 110.0],
            "raw_days": [39700.0] * 3,
            "date_str": ["2008-10-23"] * 3,
            "time_str": ["10:00:00"] * 3,
        }
    )

    clean_df, diag, intervals = validate_and_clean_trajectory(df)
    assert diag["chronological_inversions"] == 1
    assert len(clean_df) == 3

    assert clean_df["timestamp"].tolist() == [t0, t2, t1]
    assert np.all(intervals > 0)


# -----------------------------------------------------------------------------
# 6. Duplicate Detection & Same-Second Observation Preservation Tests
# -----------------------------------------------------------------------------


def test_duplicate_detection_and_same_second_preservation():
    """Test exact duplicates are removed while distinct same-second spatial observations are preserved."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:00:05")

    df = pd.DataFrame(
        {
            "user_id": ["001", "001", "001", "001", "001"],
            "trajectory_id": ["001_t1"] * 5,
            # Row 0 and 1: exact duplicates at t0
            # Row 2, 3, 4: same second t1; row 2 and 3 differ in lat; row 4 has same coords as row 2 (diff alt)
            "timestamp": [t0, t0, t1, t1, t1],
            "latitude": [39.90, 39.90, 39.91, 39.92, 39.91],
            "longitude": [116.30, 116.30, 116.31, 116.31, 116.31],
            "altitude": [100.0, 100.0, 100.0, 100.0, 105.0],
            "raw_days": [39700.0] * 5,
            "date_str": ["2008-10-23"] * 5,
            "time_str": ["10:00:00", "10:00:00", "10:00:05", "10:00:05", "10:00:05"],
        }
    )

    clean_df, diag, intervals = validate_and_clean_trajectory(df)
    # Row 1 is exact duplicate -> dropped
    # Row 4 has same timestamp and same coords as row 2 -> dropped
    # Row 0 (t0), Row 2 (t1 fix 1), Row 3 (t1 fix 2) -> ALL PRESERVED!
    assert diag["exact_duplicates"] == 1
    assert diag["duplicate_same_coord_removed"] == 1
    assert len(clean_df) == 3

    # Check preserved same-second spatial observations
    t1_records = clean_df[clean_df["timestamp"] == t1]
    assert len(t1_records) == 2
    assert t1_records["subsecond_seq"].tolist() == [0, 1]
    assert t1_records["timestamp_collision"].tolist() == [True, True]
    assert clean_df["timestamp_collision"].iloc[0] is False or clean_df["timestamp_collision"].iloc[0] == 0

    # Interval checks
    assert diag["zero_dt_count"] == 1
    assert len(intervals) == 1
    assert intervals[0] == 5.0


# -----------------------------------------------------------------------------
# 7. Missing Value Detection Tests
# -----------------------------------------------------------------------------


def test_missing_value_detection():
    """Test missing values (NaN in lat or lon) are detected and dropped."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:00:05")
    t2 = pd.Timestamp("2008-10-23 10:00:10")

    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": [t0, t1, t2],
            "latitude": [39.9, np.nan, 39.9],
            "longitude": [116.3, 116.3, np.nan],
            "altitude": [100.0, 100.0, 100.0],
            "raw_days": [39700.0] * 3,
            "date_str": ["2008-10-23"] * 3,
            "time_str": ["10:00:00"] * 3,
        }
    )

    clean_df, diag, _ = validate_and_clean_trajectory(df)
    assert diag["missing_values"] == 2
    assert len(clean_df) == 1
    assert clean_df["timestamp"].iloc[0] == t0


# -----------------------------------------------------------------------------
# 8. Preservation Rules Tests (Conservative Cleaning)
# -----------------------------------------------------------------------------


def test_preservation_rules_stationary_and_speed_anomalies():
    """Ensure stationary points, rapid movements, and sensor -777 altitude are NOT removed."""
    timestamps = pd.date_range("2008-10-23 10:00:00", periods=4, freq="5s")
    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": timestamps,
            "latitude": [39.9000, 39.9000, 40.5000, 39.9000],
            "longitude": [116.3000, 116.3000, 117.0000, 116.3000],
            "altitude": [-777.0, -777.0, 500.0, -777.0],
            "raw_days": [39700.0] * 4,
            "date_str": ["2008-10-23"] * 4,
            "time_str": ["10:00:00"] * 4,
        }
    )

    clean_df, diag, intervals = validate_and_clean_trajectory(df)
    assert len(clean_df) == 4
    assert diag["removed_points"] == 0
    assert (clean_df["altitude"] == -777.0).sum() == 3
    assert len(intervals) == 3
    assert np.all(intervals == 5.0)


# -----------------------------------------------------------------------------
# 9. Sampling Interval & Duration Calculation Tests
# -----------------------------------------------------------------------------


def test_duration_and_sampling_interval_calculation():
    """Test duration and sampling interval statistics calculation."""
    times = [
        pd.Timestamp("2008-10-23 10:00:00"),
        pd.Timestamp("2008-10-23 10:00:02"),
        pd.Timestamp("2008-10-23 10:00:07"),
        pd.Timestamp("2008-10-23 10:00:17"),
    ]
    df = pd.DataFrame(
        {
            "user_id": "001",
            "trajectory_id": "001_t1",
            "timestamp": times,
            "latitude": [39.9] * 4,
            "longitude": [116.3] * 4,
            "altitude": [100.0] * 4,
            "raw_days": [39700.0] * 4,
            "date_str": ["2008-10-23"] * 4,
            "time_str": ["10:00:00"] * 4,
        }
    )

    clean_df, diag, intervals = validate_and_clean_trajectory(df)
    assert diag["duration_seconds"] == 17.0
    assert list(intervals) == [2.0, 5.0, 10.0]

    stats = calculate_distribution_stats(intervals)
    assert stats["count"] == 3
    assert stats["min"] == 2.0
    assert stats["max"] == 10.0
    assert stats["mean"] == pytest.approx(5.6666, rel=1e-3)
    assert stats["median"] == 5.0


def test_single_point_trajectory_graceful_handling():
    """Test 1-point trajectory does not fail and reports 0 duration."""
    df = pd.DataFrame(
        {
            "user_id": ["001"],
            "trajectory_id": ["001_t1"],
            "timestamp": [pd.Timestamp("2008-10-23 10:00:00")],
            "latitude": [39.9],
            "longitude": [116.3],
            "altitude": [100.0],
            "raw_days": [39700.0],
            "date_str": ["2008-10-23"],
            "time_str": ["10:00:00"],
        }
    )

    clean_df, diag, intervals = validate_and_clean_trajectory(df)
    assert len(clean_df) == 1
    assert diag["duration_seconds"] == 0.0
    assert len(intervals) == 0


# -----------------------------------------------------------------------------
# 10. End-to-End Pipeline Fixture Test
# -----------------------------------------------------------------------------


def test_end_to_end_pipeline_fixture(tmp_path: Path):
    """Test full processing pipeline on a mini fixture structure, verifying parquet & reports."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    u1_dir = raw_dir / "001" / "Trajectory"
    _create_sample_plt(
        u1_dir / "20081023010000.plt",
        [
            (39.9, 116.3, 0, 100, 39700.1, "2008-10-23", "01:00:00"),
            (39.9, 116.3, 0, 100, 39700.1, "2008-10-23", "01:00:05"),
        ],
    )
    _create_sample_plt(
        u1_dir / "20081023020000.plt",
        [
            (39.91, 116.31, 0, 110, 39700.2, "2008-10-23", "02:00:00"),
            (39.92, 116.32, 0, 120, 39700.2, "2008-10-23", "02:00:10"),
        ],
    )

    u2_dir = raw_dir / "002" / "Trajectory"
    _create_sample_plt(
        u2_dir / "20081023030000.plt",
        [
            (39.8, 116.2, 0, 50, 39700.3, "2008-10-23", "03:00:00"),
            (39.81, 116.21, 0, 60, 39700.3, "2008-10-23", "03:00:03"),
        ],
    )

    config = IngestionConfig(num_workers=2, batch_size=2)
    parquet_path, report = process_geolife_dataset(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        config=config,
    )

    assert parquet_path.exists()
    assert (processed_dir / "data_quality_report.json").exists()
    assert (processed_dir / "data_quality_report.md").exists()

    assert report["users"] == 2
    assert report["trajectories_processed"] == 3
    assert report["total_raw_points"] == 6
    assert report["total_clean_points"] == 6
    assert report["records_removed"]["total_removed"] == 0

    loaded_df = pd.read_parquet(parquet_path)
    assert len(loaded_df) == 6
    assert set(loaded_df["user_id"].unique()) == {"001", "002"}
    assert "subsecond_seq" in loaded_df.columns
    assert "timestamp_collision" in loaded_df.columns

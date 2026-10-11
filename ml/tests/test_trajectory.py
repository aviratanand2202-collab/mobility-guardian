"""Unit tests for Trajectory Continuous Segmentation & 120-Second Analysis Windowing (Chunk 2).

All tests use small synthetic fixtures and do NOT depend on the full GeoLife dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ml.src.trajectory import (
    SegmentationConfig,
    calculate_distribution_stats,
    process_dataset_segmentation,
    segment_and_window_dataframe,
    segment_trajectory,
)


def _create_synthetic_points(
    user_id: str,
    trajectory_id: str,
    timestamps: list[pd.Timestamp],
    subsecond_seqs: list[int] | None = None,
    latitudes: list[float] | None = None,
    longitudes: list[float] | None = None,
) -> pd.DataFrame:
    """Helper to build synthetic trajectory DataFrames with Chunk 1 schema."""
    n = len(timestamps)
    if subsecond_seqs is None:
        subsecond_seqs = [0] * n
    if latitudes is None:
        latitudes = [39.9 + 0.0001 * i for i in range(n)]
    if longitudes is None:
        longitudes = [116.3 + 0.0001 * i for i in range(n)]

    ts_series = pd.Series(timestamps)
    dup_ts = ts_series.duplicated(keep=False)

    return pd.DataFrame(
        {
            "user_id": [user_id] * n,
            "trajectory_id": [trajectory_id] * n,
            "timestamp": timestamps,
            "subsecond_seq": subsecond_seqs,
            "latitude": latitudes,
            "longitude": longitudes,
            "altitude": [100.0] * n,
            "raw_days": [39700.0] * n,
            "date_str": [t.strftime("%Y-%m-%d") for t in timestamps],
            "time_str": [t.strftime("%H:%M:%S") for t in timestamps],
            "timestamp_collision": dup_ts.values,
        }
    )


# -----------------------------------------------------------------------------
# 1. Chronological Ordering Test (timestamp + subsecond_seq)
# -----------------------------------------------------------------------------


def test_chronological_ordering_with_subsecond_seq():
    """Verify sorting strictly by timestamp, then subsecond_seq even if input is shuffled."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:00:05")

    # Deliberately out of order in time and subsecond_seq
    df = _create_synthetic_points(
        user_id="u1",
        trajectory_id="t1",
        timestamps=[t1, t0, t0],
        subsecond_seqs=[0, 1, 0],
    )

    segmented_df, _, _ = segment_trajectory(df)
    assert segmented_df["timestamp"].tolist() == [t0, t0, t1]
    assert segmented_df["subsecond_seq"].tolist() == [0, 1, 0]


# -----------------------------------------------------------------------------
# 2. Same-Second Observation (dt = 0) Does Not Split Segments
# -----------------------------------------------------------------------------


def test_same_second_dt_zero_does_not_create_new_segment():
    """Verify dt = 0 does not trigger a segment break and keeps identical segment_id."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:00:01")

    df = _create_synthetic_points(
        user_id="u1",
        trajectory_id="t1",
        timestamps=[t0, t0, t0, t1],
        subsecond_seqs=[0, 1, 2, 0],
    )

    segmented_df, seg_meta, _ = segment_trajectory(df)
    assert len(seg_meta) == 1
    assert segmented_df["segment_id"].nunique() == 1
    assert segmented_df["segment_index"].tolist() == [0, 0, 0, 0]
    assert segmented_df["dt"].iloc[1] == 0.0
    assert segmented_df["dt"].iloc[2] == 0.0
    assert segmented_df["dt"].iloc[3] == 1.0


# -----------------------------------------------------------------------------
# 3. Large Temporal Gap Creates A New Segment
# -----------------------------------------------------------------------------


def test_large_temporal_gap_creates_new_segment():
    """Verify gaps exceeding max_gap_seconds trigger a new continuous segment."""
    config = SegmentationConfig(max_gap_seconds=300.0)

    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:02:00")  # 120s gap -> same segment
    t2 = pd.Timestamp("2008-10-23 10:08:00")  # 360s gap -> new segment!
    t3 = pd.Timestamp("2008-10-23 10:08:05")  # 5s gap -> same segment

    df = _create_synthetic_points(
        user_id="u1",
        trajectory_id="t1",
        timestamps=[t0, t1, t2, t3],
    )

    segmented_df, seg_meta, gaps = segment_trajectory(df, config=config)
    assert len(seg_meta) == 2
    assert segmented_df["segment_index"].tolist() == [0, 0, 1, 1]
    assert segmented_df["segment_id"].iloc[0] == "t1_s000"
    assert segmented_df["segment_id"].iloc[2] == "t1_s001"
    assert gaps == [360.0]


# -----------------------------------------------------------------------------
# 4 & 5. Segment IDs Do Not Cross Users or Source Trajectories
# -----------------------------------------------------------------------------


def test_segment_ids_do_not_cross_users_or_trajectories():
    """Verify segment IDs are partitioned strictly within each user and trajectory."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")

    df_u1_t1 = _create_synthetic_points(user_id="u1", trajectory_id="u1_t1", timestamps=[t0])
    df_u1_t2 = _create_synthetic_points(user_id="u1", trajectory_id="u1_t2", timestamps=[t0])
    df_u2_t1 = _create_synthetic_points(user_id="u2", trajectory_id="u2_t1", timestamps=[t0])

    combined = pd.concat([df_u1_t1, df_u1_t2, df_u2_t1], ignore_index=True)
    res_df, seg_meta, _, _ = segment_and_window_dataframe(combined)

    seg_ids = res_df["segment_id"].tolist()
    assert seg_ids[0] == "u1_t1_s000"
    assert seg_ids[1] == "u1_t2_s000"
    assert seg_ids[2] == "u2_t1_s000"
    assert len(set(seg_ids)) == 3


# -----------------------------------------------------------------------------
# 6 & 7. Windows Are Exactly Non-Overlapping & No Point Belongs to > 1 Window
# -----------------------------------------------------------------------------


def test_windows_are_strictly_non_overlapping():
    """Verify non-overlapping 120s window bounds and single window membership per point."""
    config = SegmentationConfig(window_duration_seconds=120.0)

    # 10 points spanning 250 seconds
    base = pd.Timestamp("2008-10-23 10:00:00")
    timestamps = [base + pd.Timedelta(seconds=s) for s in [0, 30, 60, 119, 120, 150, 239, 240, 245, 250]]

    df = _create_synthetic_points(user_id="u1", trajectory_id="t1", timestamps=timestamps)
    res_df, _, win_meta, _ = segment_and_window_dataframe(df, config=config)

    # Windows:
    # w0: [0, 120) -> points at 0, 30, 60, 119
    # w1: [120, 240) -> points at 120, 150, 239
    # w2: [240, 360) -> points at 240, 245, 250
    assert len(win_meta) == 3
    assert res_df["window_index"].tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]

    # Every point belongs to exactly one window
    assert res_df["window_id"].notna().all()
    assert len(res_df) == 10

    # Window times do not overlap
    for i in range(len(win_meta) - 1):
        assert win_meta[i]["window_end"] == win_meta[i + 1]["window_start"]


# -----------------------------------------------------------------------------
# 8. Windows Never Cross Segment Boundaries
# -----------------------------------------------------------------------------


def test_windows_never_cross_segment_boundaries():
    """Verify windows are reset at segment boundaries and never span across segments."""
    config = SegmentationConfig(max_gap_seconds=300.0, window_duration_seconds=120.0)

    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = pd.Timestamp("2008-10-23 10:01:00")  # in s000
    t2 = pd.Timestamp("2008-10-23 10:10:00")  # gap of 540s -> s001 starts!
    t3 = pd.Timestamp("2008-10-23 10:11:00")  # in s001

    df = _create_synthetic_points(user_id="u1", trajectory_id="t1", timestamps=[t0, t1, t2, t3])
    res_df, _, win_meta, _ = segment_and_window_dataframe(df, config=config)

    # Segment s000 has window t1_s000_w0000
    # Segment s001 has window t1_s001_w0000
    assert res_df["segment_id"].iloc[0] == "t1_s000"
    assert res_df["segment_id"].iloc[2] == "t1_s001"

    assert res_df["window_id"].iloc[0] == "t1_s000_w0000"
    assert res_df["window_id"].iloc[1] == "t1_s000_w0000"
    assert res_df["window_id"].iloc[2] == "t1_s001_w0000"
    assert res_df["window_id"].iloc[3] == "t1_s001_w0000"


# -----------------------------------------------------------------------------
# 9 & 10. Same-Second Observations & Division-by-Zero Protection
# -----------------------------------------------------------------------------


def test_same_second_observations_in_windows_and_zero_dt():
    """Verify multiple same-second fixes map to the same window without division by zero."""
    t0 = pd.Timestamp("2008-10-23 10:00:00")

    df = _create_synthetic_points(
        user_id="u1",
        trajectory_id="t1",
        timestamps=[t0, t0, t0],
        subsecond_seqs=[0, 1, 2],
    )

    res_df, seg_meta, win_meta, _ = segment_and_window_dataframe(df)
    assert len(res_df) == 3
    assert res_df["subsecond_seq"].tolist() == [0, 1, 2]
    assert res_df["window_id"].nunique() == 1
    assert seg_meta[0]["zero_dt_count"] == 2
    assert seg_meta[0]["positive_dt_count"] == 0

    # Ensure dt=0 does not divide by zero in interval stats
    stats = calculate_distribution_stats(np.array([], dtype=np.float64))
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


# -----------------------------------------------------------------------------
# 11. Partial Trailing-Window Behavior & Configurable Rule
# -----------------------------------------------------------------------------


def test_partial_window_coverage_rule():
    """Verify full vs partial classification and configurable discarding threshold."""
    # 3 points: 0s, 60s, 140s (Segment duration 140s -> Window 0 is Full [0, 120), Window 1 is Partial [120, 240))
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = t0 + pd.Timedelta(seconds=60)
    t2 = t0 + pd.Timedelta(seconds=140)

    df = _create_synthetic_points(user_id="u1", trajectory_id="t1", timestamps=[t0, t1, t2])

    # Default rule: keep all partial windows (min_window_points=1, min_window_duration=0)
    cfg_keep = SegmentationConfig(min_window_points=1, min_window_duration_seconds=0.0)
    res_keep, _, win_keep, _ = segment_and_window_dataframe(df, config=cfg_keep)
    assert win_keep[0]["window_type"] == "FULL"
    assert win_keep[0]["is_full_window"] is True
    assert win_keep[1]["window_type"] == "PARTIAL"
    assert win_keep[1]["is_full_window"] is False
    assert res_keep["window_id"].notna().all()

    # Stricter rule: require at least 2 points in partial window
    cfg_strict = SegmentationConfig(min_window_points=2)
    res_strict, _, win_strict, _ = segment_and_window_dataframe(df, config=cfg_strict)
    # Window 0 has 2 points -> kept (FULL)
    # Window 1 has 1 point -> discarded
    assert win_strict[0]["window_type"] == "FULL"
    assert win_strict[1]["window_type"] == "DISCARDED"
    assert res_strict["window_id"].iloc[2] is None


# -----------------------------------------------------------------------------
# 12. Deterministic Results on Repeated Execution
# -----------------------------------------------------------------------------


def test_deterministic_results_repeated_execution():
    """Verify pipeline output is identical on repeated executions."""
    base = pd.Timestamp("2008-10-23 10:00:00")
    timestamps = [base + pd.Timedelta(seconds=s) for s in [0, 10, 20, 20, 150, 600]]
    subsecond = [0, 0, 0, 1, 0, 0]

    df = _create_synthetic_points(user_id="u1", trajectory_id="t1", timestamps=timestamps, subsecond_seqs=subsecond)

    res1, seg1, win1, _ = segment_and_window_dataframe(df)
    res2, seg2, win2, _ = segment_and_window_dataframe(df)

    pd.testing.assert_frame_equal(res1, res2)
    assert seg1 == seg2
    assert win1 == win2


# -----------------------------------------------------------------------------
# 13. End-to-End Pipeline Fixture Test
# -----------------------------------------------------------------------------


def test_end_to_end_segmentation_fixture(tmp_path: Path):
    """Verify end-to-end dataset segmentation and windowing with report generation."""
    input_parquet = tmp_path / "trajectories.parquet"
    out_dir = tmp_path / "processed"

    # Create dummy parquet with 2 trajectories
    t0 = pd.Timestamp("2008-10-23 10:00:00")
    t1 = t0 + pd.Timedelta(seconds=5)
    t2 = t0 + pd.Timedelta(seconds=500)  # gap

    df1 = _create_synthetic_points(user_id="001", trajectory_id="001_t1", timestamps=[t0, t1, t2])
    df2 = _create_synthetic_points(user_id="002", trajectory_id="002_t1", timestamps=[t0, t1])
    combined = pd.concat([df1, df2], ignore_index=True)
    combined.to_parquet(input_parquet)

    config = SegmentationConfig(max_gap_seconds=300.0)
    out_file, report = process_dataset_segmentation(input_parquet, out_dir, config=config)

    assert out_file.exists()
    assert (out_dir / "segmentation_window_report.json").exists()
    assert (out_dir / "segmentation_window_report.md").exists()

    assert report["users"] == 2
    assert report["source_trajectories"] == 2
    # df1 splits on 500s gap into 2 segments; df2 is 1 segment -> total 3 segments
    assert report["continuous_segments"] == 3
    assert report["total_points"] == 5
    assert report["points_assigned_to_windows"] == 5

    loaded = pd.read_parquet(out_file)
    assert len(loaded) == 5
    assert "segment_id" in loaded.columns
    assert "window_id" in loaded.columns
    assert "dt" in loaded.columns

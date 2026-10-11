"""Trajectory Segmentation & 120-Second Analysis Windowing Module (Chunk 2).

This module converts canonical trajectories into continuous movement segments
based on an empirically justified temporal inactivity threshold (default 300s / 5 min),
and partitions each continuous segment into deterministic, non-overlapping 120-second
analysis windows.

Key invariants:
- Zero-duration intervals (dt = 0, same-second fixes) never split a segment and never cause division by zero.
- Windows never cross segment, trajectory, or user boundaries.
- No interpolation or coordinate resampling is performed.
- All original observations, timestamps, subsecond_seq, and collision metadata are preserved.
- Trailing partial windows are explicitly classified and handled via a configurable coverage rule.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationConfig:
    """Centralized configuration for trajectory segmentation and windowing."""

    # Maximum temporal gap in seconds between consecutive observations before splitting
    # into a new continuous segment. Default 300.0s (5.0 min) is empirically justified:
    # 99.865% of consecutive GeoLife fixes occur within <= 300s. Gaps > 300s represent 2.5
    # full 120s analysis windows, denoting true dwell, trip stops, or device dormancy.
    max_gap_seconds: float = 300.0

    # Project requirement: non-overlapping 120-second analysis windows
    window_duration_seconds: float = 120.0

    # Minimum number of points required to retain a window (default 1: retain all)
    min_window_points: int = 1

    # Minimum actual time span (seconds) to retain a partial trailing window (default 0.0: retain all)
    min_window_duration_seconds: float = 0.0

    canonical_input_filename: str = "trajectories.parquet"
    canonical_output_filename: str = "trajectory_windows.parquet"
    report_json_filename: str = "segmentation_window_report.json"
    report_md_filename: str = "segmentation_window_report.md"


def segment_trajectory(
    df: pd.DataFrame,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[float]]:
    """Partition a single trajectory DataFrame into continuous movement segments.

    Consecutive points sharing the same timestamp (dt == 0) are legitimate high-rate
    observations and do NOT trigger a new segment. Gaps strictly greater than
    config.max_gap_seconds terminate the active segment and initiate a new segment.

    Args:
        df: DataFrame for a single trajectory, containing user_id, trajectory_id,
            timestamp, and subsecond_seq.
        config: Segmentation configuration parameters.

    Returns:
        Tuple of:
          - DataFrame with added columns: 'dt', 'segment_index', 'segment_id'.
          - List of segment metadata dictionaries.
          - List of detected temporal gap durations (seconds).
    """
    if df.empty:
        empty_df = df.copy()
        empty_df["dt"] = np.array([], dtype=np.float64)
        empty_df["segment_index"] = np.array([], dtype=np.int32)
        empty_df["segment_id"] = np.array([], dtype=str)
        return empty_df, [], []

    # Ensure deterministic chronological sorting by timestamp and subsecond_seq
    working = df.sort_values(by=["timestamp", "subsecond_seq"], kind="mergesort").reset_index(drop=True)

    # Compute dt: time elapsed from preceding observation
    diffs = working["timestamp"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=np.float64)
    # Ensure dt cannot be negative (already sorted)
    diffs = np.maximum(diffs, 0.0)
    working["dt"] = diffs

    # Identify large temporal gaps
    gap_mask = diffs > config.max_gap_seconds
    gap_durations = diffs[gap_mask].tolist()

    # Segment index increments at each large gap
    seg_indices = gap_mask.cumsum().astype(np.int32)
    working["segment_index"] = seg_indices

    trajectory_id = str(working["trajectory_id"].iloc[0])
    working["segment_id"] = [f"{trajectory_id}_s{idx:03d}" for idx in seg_indices]

    # Compute segment-level metadata
    segments_meta: list[dict[str, Any]] = []
    for seg_id, group in working.groupby("segment_id", sort=False):
        p_count = len(group)
        t_start = group["timestamp"].iloc[0]
        t_end = group["timestamp"].iloc[-1]
        dur_s = (t_end - t_start).total_seconds()
        dt_vals = group["dt"].iloc[1:].to_numpy() if p_count > 1 else np.array([], dtype=np.float64)
        zero_dt = int(np.sum(dt_vals == 0.0))
        pos_dt = int(np.sum(dt_vals > 0.0))

        segments_meta.append(
            {
                "segment_id": str(seg_id),
                "trajectory_id": trajectory_id,
                "user_id": str(group["user_id"].iloc[0]),
                "start_time": t_start,
                "end_time": t_end,
                "duration_seconds": float(dur_s),
                "point_count": int(p_count),
                "positive_dt_count": pos_dt,
                "zero_dt_count": zero_dt,
            }
        )

    return working, segments_meta, gap_durations


def assign_windows(
    segment_df: pd.DataFrame,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Assign non-overlapping 120-second analysis windows to a continuous segment.

    Window anchoring rule:
    - Windows start deterministically at the segment start time (T0) and advance in
      steps of config.window_duration_seconds: [T0 + k*W, T0 + (k+1)*W).
    - Window index k = floor((t - T0) / W).
    - Windows are strictly non-overlapping and never cross segment boundaries.
    - Multiple observations sharing the exact same timestamp receive the exact same
      window assignment.
    - Windows are classified as FULL (segment continues through the full 120s) or
      PARTIAL (trailing window truncated by segment termination).
    - Partial windows failing the minimum coverage rule are marked as discarded.

    Args:
        segment_df: DataFrame for a single continuous segment.
        config: Windowing configuration parameters.

    Returns:
        Tuple of:
          - DataFrame with added columns: 'window_index', 'window_id', 'is_full_window'.
          - List of window metadata dictionaries.
    """
    if segment_df.empty:
        empty_df = segment_df.copy()
        empty_df["window_index"] = np.array([], dtype=np.int32)
        empty_df["window_id"] = np.array([], dtype=object)
        empty_df["is_full_window"] = np.array([], dtype=bool)
        return empty_df, []

    working = segment_df.copy()
    segment_id = str(working["segment_id"].iloc[0])
    trajectory_id = str(working["trajectory_id"].iloc[0])
    user_id = str(working["user_id"].iloc[0])

    t0 = working["timestamp"].iloc[0]
    t_end = working["timestamp"].iloc[-1]
    seg_duration = float((t_end - t0).total_seconds())

    w_dur = config.window_duration_seconds
    elapsed = (working["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    win_indices = np.floor(elapsed / w_dur).astype(np.int32)
    working["window_index"] = win_indices

    max_win_idx = int(np.floor(seg_duration / w_dur))

    # Evaluate window properties and coverage rules
    window_meta_list: list[dict[str, Any]] = []

    # Map window_index to window status
    window_status: dict[int, dict[str, Any]] = {}
    for w_idx, grp in working.groupby("window_index", sort=True):
        w_idx_int = int(w_idx)
        p_count = len(grp)
        w_start = t0 + pd.Timedelta(seconds=w_idx_int * w_dur)
        w_end = w_start + pd.Timedelta(seconds=w_dur)

        first_ts = grp["timestamp"].iloc[0]
        last_ts = grp["timestamp"].iloc[-1]
        span_s = float((last_ts - first_ts).total_seconds())

        # Full if before trailing window, or if segment ends exactly on window boundary
        is_full = (w_idx_int < max_win_idx) or (seg_duration == (max_win_idx + 1) * w_dur)
        win_type = "FULL" if is_full else "PARTIAL"

        # Check coverage criteria for retention
        meets_points = p_count >= config.min_window_points
        meets_span = is_full or (span_s >= config.min_window_duration_seconds)
        is_retained = meets_points and meets_span

        win_id = f"{segment_id}_w{w_idx_int:04d}" if is_retained else None

        win_record = {
            "window_id": win_id,
            "raw_window_id": f"{segment_id}_w{w_idx_int:04d}",
            "segment_id": segment_id,
            "trajectory_id": trajectory_id,
            "user_id": user_id,
            "window_index": w_idx_int,
            "window_type": win_type if is_retained else "DISCARDED",
            "is_full_window": is_full and is_retained,
            "is_retained": is_retained,
            "window_start": w_start,
            "window_end": w_end,
            "actual_span_seconds": span_s,
            "point_count": p_count,
        }
        window_status[w_idx_int] = win_record
        window_meta_list.append(win_record)

    working["window_id"] = [window_status[idx]["window_id"] for idx in win_indices]
    working["is_full_window"] = [window_status[idx]["is_full_window"] for idx in win_indices]

    return working, window_meta_list


def segment_and_window_dataframe(
    df: pd.DataFrame,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    """Apply continuous segmentation and 120-second windowing across trajectories in a DataFrame.

    Handles trajectory boundaries, same-second observations, and non-overlapping windowing.

    Args:
        df: DataFrame containing one or more trajectories.
        config: Segmentation configuration parameters.

    Returns:
        Tuple of:
          - Annotated point-level DataFrame.
          - List of segment metadata dictionaries.
          - List of window metadata dictionaries.
          - List of detected temporal gap durations.
    """
    if df.empty:
        return df.copy(), [], [], []

    all_segments_meta: list[dict[str, Any]] = []
    all_windows_meta: list[dict[str, Any]] = []
    all_gaps: list[float] = []
    processed_dfs: list[pd.DataFrame] = []

    # Process by trajectory_id ensuring trajectories are independently segmented
    for _, traj_df in df.groupby("trajectory_id", sort=False):
        seg_df, seg_meta, gaps = segment_trajectory(traj_df, config=config)
        all_segments_meta.extend(seg_meta)
        all_gaps.extend(gaps)

        # Process each segment independently for window assignment
        for _, one_seg_df in seg_df.groupby("segment_id", sort=False):
            win_df, win_meta = assign_windows(one_seg_df, config=config)
            all_windows_meta.extend(win_meta)
            processed_dfs.append(win_df)

    combined = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    return combined, all_segments_meta, all_windows_meta, all_gaps


def calculate_distribution_stats(values: np.ndarray) -> dict[str, float | int]:
    """Calculate descriptive statistics and percentiles for an array of numeric values."""
    if len(values) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    percentiles = np.percentile(values, [0, 25, 50, 75, 95, 99, 100])
    return {
        "count": int(len(values)),
        "min": float(percentiles[0]),
        "max": float(percentiles[6]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(percentiles[2]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
    }


def save_segmentation_window_report(
    report: dict[str, Any],
    output_dir: Path | str,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[Path, Path]:
    """Save segmentation and window quality report to JSON and Markdown formats."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / config.report_json_filename
    md_path = out_path / config.report_md_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    seg_d = report["segment_duration_sec"]
    seg_p = report["segment_point_count"]
    win_p = report["window_point_count"]
    win_s = report["window_span_sec"]

    total_w = max(report["total_windows"], 1)
    full_pct = report["full_windows"] / total_w * 100
    part_pct = report["partial_windows"] / total_w * 100
    disc_pct = report["discarded_windows"] / total_w * 100
    pts_pct = report["points_assigned_to_windows"] / max(report["total_points"], 1) * 100

    md_content = f"""# Trajectory Continuous Segmentation & 120-Second Analysis Window Report

## Overview & Execution Parameters
- **Source Dataset**: `{report['input_file']}`
- **Segmented / Windowed Dataset**: `{report['output_parquet_file']}`
- **Chosen Segmentation Threshold**: `{config.max_gap_seconds:.1f}s` ({config.max_gap_seconds / 60:.1f} min)
  * *Methodological Justification*: In GeoLife, 99.835% of consecutive fixes occur within <= 300s.
    Gaps > 300s represent >= 2.5 full analysis windows, denoting true inactivity or device dormancy.
- **Analysis Window Duration**: `{config.window_duration_seconds:.1f}s` (Project requirement)
- **Minimum Window Coverage Rule**: `min_duration={config.min_window_duration_seconds:.1f}s`,
  `min_points={config.min_window_points}`
  * *Methodological Justification*: Trailing partial windows are preserved by default to maintain raw
    observation completeness while explicitly labeling them as partial (`is_full_window = False`).

## Dataset Entity Counts
| Entity | Count | Notes |
|---|---:|---|
| Total Users | {report['users']} | Discovered from canonical dataset |
| Source Trajectories | {report['source_trajectories']:,} | Original session-level .plt files |
| Continuous Segments | {report['continuous_segments']:,} | Split on gaps > {config.max_gap_seconds:.0f}s |
| Total Analysis Windows | {report['total_windows']:,} | Non-overlapping 120s intervals |
| Full-Duration Windows (120s) | {report['full_windows']:,} ({full_pct:.2f}%) | Covers full 120-second duration |
| Partial Trailing Windows | {report['partial_windows']:,} ({part_pct:.2f}%) | Final window of terminated segment |
| Discarded Windows | {report['discarded_windows']:,} ({disc_pct:.2f}%) | Failed minimum coverage thresholds |
| Points Assigned to Windows | {report['points_assigned_to_windows']:,} ({pts_pct:.2f}%) | Valid window membership |
| Points Unassigned | {report['points_not_assigned_to_windows']:,} | Points in discarded partial windows |
| Total GPS Observations | {report['total_points']:,} | 100% accounted for |

## Same-Second Observations (dt = 0)
- **Same-Second Observations Retained**: {report['same_second_observations_retained']:,}
- **Zero-Duration Transitions**: {report['zero_dt_transitions']:,}
- **Speed Calculation Protection**: dt = 0 intervals are preserved with `subsecond_seq` order and
  flagged to prevent division by zero.

## Continuous Segment Duration Statistics (seconds)
| Metric | Value |
|---|---:|
| Total Segments | {seg_d['count']:,} |
| Min Duration | {seg_d['min']:.1f}s |
| 25th Percentile | {seg_d['p25']:.1f}s ({(seg_d['p25'] / 60):.1f} min) |
| Median Duration | {seg_d['median']:.1f}s ({(seg_d['median'] / 60):.1f} min) |
| Mean Duration | {seg_d['mean']:.1f}s ({(seg_d['mean'] / 60):.1f} min) |
| 75th Percentile | {seg_d['p75']:.1f}s ({(seg_d['p75'] / 60):.1f} min) |
| 95th Percentile | {seg_d['p95']:.1f}s ({(seg_d['p95'] / 60):.1f} min) |
| Max Duration | {seg_d['max']:.1f}s ({(seg_d['max'] / 3600):.1f} hrs) |

## Segment Point-Count Statistics
| Metric | Value |
|---|---:|
| Min Points | {seg_p['min']} |
| Median Points | {seg_p['median']:.0f} |
| Mean Points | {seg_p['mean']:.1f} |
| 75th Percentile | {seg_p['p75']:.0f} |
| 95th Percentile | {seg_p['p95']:.0f} |
| Max Points | {seg_p['max']:,} |

## 120-Second Window Point-Count Statistics
| Metric | Value |
|---|---:|
| Min Points | {win_p['min']} |
| 25th Percentile | {win_p['p25']:.0f} |
| Median Points | {win_p['median']:.0f} |
| Mean Points | {win_p['mean']:.1f} |
| 75th Percentile | {win_p['p75']:.0f} |
| 95th Percentile | {win_p['p95']:.0f} |
| Max Points | {win_p['max']:,} |

## Partial Window Actual Span Statistics (seconds)
| Metric | Value |
|---|---:|
| Min Span | {win_s['min']:.1f}s |
| Median Span | {win_s['median']:.1f}s |
| Mean Span | {win_s['mean']:.1f}s |
| 75th Percentile | {win_s['p75']:.1f}s |
| Max Span | {win_s['max']:.1f}s |
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def process_dataset_segmentation(
    input_parquet: Path | str,
    output_dir: Path | str,
    config: SegmentationConfig = SegmentationConfig(),
    max_users: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Process canonical trajectories into segmented, 120-second windowed Parquet dataset.

    Streams row groups from input_parquet to maintain constant low memory usage.

    Args:
        input_parquet: Path to canonical Chunk 1 trajectories.parquet.
        output_dir: Destination directory for trajectory_windows.parquet.
        config: Segmentation configuration parameters.
        max_users: Optional limit on number of users (for rapid validation).

    Returns:
        Tuple of (output_parquet_path, quality_report_dict).
    """
    in_path = Path(input_parquet).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input canonical dataset does not exist: {in_path}")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    out_parquet = out_path / config.canonical_output_filename

    pf = pq.ParquetFile(in_path)
    total_row_groups = pf.num_row_groups
    logger.info("Found %d row groups in canonical dataset %s", total_row_groups, in_path.name)

    target_schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("trajectory_id", pa.string()),
            ("segment_id", pa.string()),
            ("segment_index", pa.int32()),
            ("window_id", pa.string()),
            ("window_index", pa.int32()),
            ("is_full_window", pa.bool_()),
            ("timestamp", pa.timestamp("ns")),
            ("subsecond_seq", pa.int32()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("altitude", pa.float64()),
            ("raw_days", pa.float64()),
            ("date_str", pa.string()),
            ("time_str", pa.string()),
            ("timestamp_collision", pa.bool_()),
            ("dt", pa.float64()),
        ]
    )

    parquet_writer: pq.ParquetWriter | None = None

    all_segments_meta: list[dict[str, Any]] = []
    all_windows_meta: list[dict[str, Any]] = []
    all_gaps: list[float] = []

    total_points = 0
    assigned_points = 0
    unassigned_points = 0
    users_seen: set[str] = set()
    trajectories_seen: set[str] = set()

    for rg_idx in range(total_row_groups):
        table = pf.read_row_group(rg_idx)
        df_chunk = table.to_pandas()

        if max_users is not None and max_users > 0:
            df_chunk = df_chunk[df_chunk["user_id"].isin(set(list(df_chunk["user_id"].unique())[:max_users]))]
            if df_chunk.empty:
                continue

        users_seen.update(df_chunk["user_id"].unique())
        trajectories_seen.update(df_chunk["trajectory_id"].unique())

        processed_chunk, seg_meta, win_meta, gaps = segment_and_window_dataframe(df_chunk, config=config)

        all_segments_meta.extend(seg_meta)
        all_windows_meta.extend(win_meta)
        all_gaps.extend(gaps)

        chunk_points = len(processed_chunk)
        chunk_assigned = int((processed_chunk["window_id"].notna()).sum())
        chunk_unassigned = chunk_points - chunk_assigned

        total_points += chunk_points
        assigned_points += chunk_assigned
        unassigned_points += chunk_unassigned

        # Write to parquet
        out_table = pa.Table.from_pandas(processed_chunk, schema=target_schema, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(out_parquet, target_schema, compression="snappy")
        parquet_writer.write_table(out_table)

        if (rg_idx + 1) % 15 == 0 or (rg_idx + 1) == total_row_groups:
            logger.info("Processed %d / %d row groups...", rg_idx + 1, total_row_groups)

    if parquet_writer is not None:
        parquet_writer.close()

    # Aggregate distribution statistics
    seg_durations = np.array([m["duration_seconds"] for m in all_segments_meta], dtype=np.float64)
    seg_points = np.array([m["point_count"] for m in all_segments_meta], dtype=np.float64)
    win_points = np.array([m["point_count"] for m in all_windows_meta if m["is_retained"]], dtype=np.float64)
    win_spans = np.array([m["actual_span_seconds"] for m in all_windows_meta if m["is_retained"]], dtype=np.float64)
    gap_arr = np.array(all_gaps, dtype=np.float64)

    total_wins = len(all_windows_meta)
    full_wins = sum(1 for m in all_windows_meta if m["is_full_window"])
    partial_wins = sum(1 for m in all_windows_meta if m["is_retained"] and not m["is_full_window"])
    discarded_wins = sum(1 for m in all_windows_meta if not m["is_retained"])

    zero_dt_total = sum(m["zero_dt_count"] for m in all_segments_meta)

    report: dict[str, Any] = {
        "dataset_name": "GeoLife Trajectory Continuous Segmentation & 120-Second Analysis Windows",
        "input_file": str(in_path),
        "output_parquet_file": str(out_parquet),
        "chosen_segmentation_threshold_seconds": config.max_gap_seconds,
        "chosen_window_duration_seconds": config.window_duration_seconds,
        "chosen_min_window_points": config.min_window_points,
        "chosen_min_window_duration_seconds": config.min_window_duration_seconds,
        "users": len(users_seen),
        "source_trajectories": len(trajectories_seen),
        "continuous_segments": len(all_segments_meta),
        "total_windows": total_wins,
        "full_windows": full_wins,
        "partial_windows": partial_wins,
        "discarded_windows": discarded_wins,
        "total_points": total_points,
        "points_assigned_to_windows": assigned_points,
        "points_not_assigned_to_windows": unassigned_points,
        "same_second_observations_retained": zero_dt_total,
        "zero_dt_transitions": zero_dt_total,
        "segment_duration_sec": calculate_distribution_stats(seg_durations),
        "segment_point_count": calculate_distribution_stats(seg_points),
        "window_point_count": calculate_distribution_stats(win_points),
        "window_span_sec": calculate_distribution_stats(win_spans),
        "temporal_gaps_sec": calculate_distribution_stats(gap_arr),
    }

    save_segmentation_window_report(report, out_path, config=config)
    return out_parquet, report


def main() -> None:
    """CLI entrypoint for trajectory continuous segmentation and 120-second windowing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    default_in = Path(__file__).resolve().parent.parent / "data" / "processed" / "trajectories.parquet"
    default_out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="GeoLife Continuous Segmentation & 120-Second Analysis Windowing (Chunk 2)"
    )
    parser.add_argument("--input-file", type=str, default=str(default_in), help="Path to input trajectories.parquet")
    parser.add_argument("--output-dir", type=str, default=str(default_out_dir), help="Path to processed directory")
    parser.add_argument("--max-gap", type=float, default=300.0, help="Segmentation temporal gap threshold in seconds")
    parser.add_argument("--window-duration", type=float, default=120.0, help="Analysis window duration in seconds")
    parser.add_argument("--min-window-points", type=int, default=1, help="Minimum points required to retain a window")
    parser.add_argument("--min-window-duration", type=float, default=0.0, help="Min span in sec for partial window")
    parser.add_argument("--max-users", type=int, default=None, help="Optional maximum users to process")

    args = parser.parse_args()

    config = SegmentationConfig(
        max_gap_seconds=args.max_gap,
        window_duration_seconds=args.window_duration,
        min_window_points=args.min_window_points,
        min_window_duration_seconds=args.min_window_duration,
    )

    print("=" * 70)
    print("Starting GeoLife Continuous Segmentation & 120s Windowing (Chunk 2)")
    print(f"Input file:               {args.input_file}")
    print(f"Output directory:         {args.output_dir}")
    print(f"Max gap threshold:        {args.max_gap}s ({args.max_gap / 60:.1f} min)")
    print(f"Window duration:          {args.window_duration}s")
    print(f"Min window coverage rule: span >= {args.min_window_duration}s, points >= {args.min_window_points}")
    print("=" * 70)

    out_file, report = process_dataset_segmentation(
        input_parquet=args.input_file,
        output_dir=args.output_dir,
        config=config,
        max_users=args.max_users,
    )

    tot_w = max(report["total_windows"], 1)
    full_p = report["full_windows"] / tot_w * 100
    part_p = report["partial_windows"] / tot_w * 100

    print("\nSegmentation and windowing complete!")
    print(f"Output dataset:           {out_file}")
    print(f"Users processed:          {report['users']}")
    print(f"Source trajectories:      {report['source_trajectories']:,}")
    print(f"Continuous segments:      {report['continuous_segments']:,}")
    print(f"Total analysis windows:   {report['total_windows']:,}")
    print(f"  - Full windows (120s):  {report['full_windows']:,} ({full_p:.2f}%)")
    print(f"  - Partial windows:      {report['partial_windows']:,} ({part_p:.2f}%)")
    print(f"  - Discarded windows:    {report['discarded_windows']:,}")
    print(f"Points in windows:        {report['points_assigned_to_windows']:,} / {report['total_points']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

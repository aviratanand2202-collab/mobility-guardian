"""GeoLife Trajectory Ingestion, Validation & Minimal Cleaning Module.

This module provides reproducible, conservative ingestion and data validation
for Microsoft GeoLife GPS trajectories. It discovers files dynamically, parses
PLT records, validates coordinates and timestamps according to physical constraints,
preserves legitimate high-rate same-second GPS observations with collision metadata,
and outputs a canonical Parquet dataset along with a comprehensive data-quality report.
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
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionConfig:
    """Centralized configuration for GeoLife data ingestion and validation."""

    min_latitude: float = -90.0
    max_latitude: float = 90.0
    min_longitude: float = -180.0
    max_longitude: float = 180.0
    # GeoLife collection formally launched in April 2007. Timestamps prior to 2005
    # represent uninitialized GPS receiver RTC hardware defaults (e.g. 2000-01-01).
    min_valid_timestamp: str = "2005-01-01"
    header_lines: int = 6
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    batch_size: int = 250
    num_workers: int = 8
    canonical_filename: str = "trajectories.parquet"
    report_json_filename: str = "data_quality_report.json"
    report_md_filename: str = "data_quality_report.md"


def discover_trajectory_files(raw_dir: Path | str) -> list[tuple[str, Path]]:
    """Dynamically discover user folders and trajectory (.plt) files.

    No hard-coding of user IDs, user counts, filenames, or directory depths.

    Args:
        raw_dir: Root directory of raw GeoLife files (e.g. ml/data/raw/geolife).

    Returns:
        Sorted list of tuples: (user_id, plt_file_path).

    Raises:
        FileNotFoundError: If raw_dir does not exist.
    """
    raw_path = Path(raw_dir).resolve()
    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist or is not a directory: {raw_path}")

    discovered: list[tuple[str, Path]] = []

    for user_entry in sorted(raw_path.iterdir()):
        if not user_entry.is_dir():
            continue

        user_id = user_entry.name
        trajectory_sub = user_entry / "Trajectory"
        search_dir = trajectory_sub if trajectory_sub.is_dir() else user_entry

        for plt_file in sorted(search_dir.glob("*.plt")):
            if plt_file.is_file():
                discovered.append((user_id, plt_file))

    return discovered


def parse_plt_file(
    file_path: Path | str,
    user_id: str | None = None,
    trajectory_id: str | None = None,
    config: IngestionConfig = IngestionConfig(),
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Parse a single GeoLife .plt file, skipping metadata headers.

    The GeoLife PLT format:
    Line 1-6: Metadata header rows to skip.
    Line 7+: Comma-separated fields:
        Field 0: Latitude in decimal degrees (e.g. 39.984702)
        Field 1: Longitude in decimal degrees (e.g. 116.318417)
        Field 2: Reserved (all 0)
        Field 3: Altitude in feet (-777 if invalid)
        Field 4: Date as fractional days since 1899-12-30
        Field 5: Date string (yyyy-mm-dd)
        Field 6: Time string (hh:mm:ss)

    Args:
        file_path: Path to the .plt file.
        user_id: User identifier. If None, inferred from directory structure.
        trajectory_id: Trajectory identifier. If None, generated as {user_id}_{stem}.
        config: Ingestion configuration parameters.

    Returns:
        Tuple of (raw_dataframe, file_diagnostics).
        If file reading fails, returns (None, file_diagnostics with error info).
    """
    path = Path(file_path).resolve()
    diagnostics: dict[str, Any] = {
        "file_path": str(path),
        "success": False,
        "raw_points": 0,
        "error": None,
    }

    if user_id is None:
        user_id = path.parent.parent.name if path.parent.name.lower() == "trajectory" else path.parent.name

    if trajectory_id is None:
        trajectory_id = f"{user_id}_{path.stem}"

    try:
        df = pd.read_csv(
            path,
            skiprows=config.header_lines,
            header=None,
            dtype={
                0: "float64",
                1: "float64",
                2: "int64",
                3: "float64",
                4: "float64",
                5: "str",
                6: "str",
            },
            on_bad_lines="warn",
        )
    except Exception as err:
        diagnostics["error"] = f"Failed to read CSV: {err}"
        return None, diagnostics

    if df.empty:
        diagnostics["error"] = "Empty file"
        return None, diagnostics

    date_series = df[5].str.strip()
    time_series = df[6].str.strip()
    datetime_series = date_series + " " + time_series
    parsed_timestamps = pd.to_datetime(
        datetime_series,
        format=config.datetime_format,
        errors="coerce",
    )

    normalized = pd.DataFrame(
        {
            "user_id": user_id,
            "trajectory_id": trajectory_id,
            "timestamp": parsed_timestamps,
            "latitude": df[0],
            "longitude": df[1],
            "altitude": df[3],
            "raw_days": df[4],
            "date_str": date_series,
            "time_str": time_series,
        }
    )

    diagnostics["success"] = True
    diagnostics["raw_points"] = len(normalized)
    return normalized, diagnostics


def validate_and_clean_trajectory(
    df: pd.DataFrame,
    config: IngestionConfig = IngestionConfig(),
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Validate and clean a trajectory while preserving legitimate same-second spatial observations.

    Rules applied:
    - Drops points with missing required fields (NaN in timestamp, latitude, longitude).
    - Drops points with physically impossible coordinates (lat outside [-90, 90], lon outside [-180, 180]).
    - Drops points with uninitialized RTC timestamps (prior to config.min_valid_timestamp).
    - Drops exact duplicate records (all columns identical).
    - Drops identical-coordinate collisions at the same timestamp (same timestamp + identical lat/lon).
    - PRESERVES different spatial observations at the same timestamp, annotating them with
      `timestamp_collision = True` and deterministic source order `subsecond_seq` (0, 1, 2, ...).
    - PRESERVES stationary points across advancing timestamps.
    - PRESERVES large movement / high-speed points.
    - PRESERVES raw altitude (including -777 uncalibrated flag).
    - Sorts chronologically if timestamps are out of order, preserving original order for identical timestamps.

    Args:
        df: Raw parsed DataFrame for a single trajectory.
        config: Ingestion configuration parameters.

    Returns:
        Tuple of (clean_df, diagnostics_dict, non_zero_sampling_intervals_seconds_array).
    """
    raw_count = len(df)
    diagnostics: dict[str, Any] = {
        "raw_points": raw_count,
        "clean_points": 0,
        "removed_points": 0,
        "missing_values": 0,
        "invalid_coordinates": 0,
        "unparseable_timestamps": 0,
        "uninitialized_rtc_timestamps": 0,
        "exact_duplicates": 0,
        "duplicate_same_coord_removed": 0,
        "preserved_same_timestamp_observations": 0,
        "chronological_inversions": 0,
        "duration_seconds": 0.0,
        "zero_dt_count": 0,
    }

    if df.empty:
        return df.copy(), diagnostics, np.array([], dtype=np.float64)

    working = df.copy()

    # 1. Unparseable timestamps
    unparseable_ts_mask = working["timestamp"].isna()
    diagnostics["unparseable_timestamps"] = int(unparseable_ts_mask.sum())

    # 2. Missing values in core fields
    missing_coords_mask = working["latitude"].isna() | working["longitude"].isna()
    diagnostics["missing_values"] = int((unparseable_ts_mask | missing_coords_mask).sum())

    valid_core_mask = (~unparseable_ts_mask) & (~missing_coords_mask)
    working = working[valid_core_mask].copy()

    if working.empty:
        diagnostics["removed_points"] = raw_count
        return working, diagnostics, np.array([], dtype=np.float64)

    # 3. Invalid coordinate bounds
    invalid_coord_mask = (
        (working["latitude"] < config.min_latitude)
        | (working["latitude"] > config.max_latitude)
        | (working["longitude"] < config.min_longitude)
        | (working["longitude"] > config.max_longitude)
    )
    diagnostics["invalid_coordinates"] = int(invalid_coord_mask.sum())
    working = working[~invalid_coord_mask].copy()

    if working.empty:
        diagnostics["removed_points"] = raw_count
        return working, diagnostics, np.array([], dtype=np.float64)

    # 4. Uninitialized RTC / default epoch timestamps (e.g. 2000-01-01)
    min_ts_bound = pd.Timestamp(config.min_valid_timestamp)
    uninitialized_rtc_mask = working["timestamp"] < min_ts_bound
    diagnostics["uninitialized_rtc_timestamps"] = int(uninitialized_rtc_mask.sum())
    working = working[~uninitialized_rtc_mask].copy()

    if working.empty:
        diagnostics["removed_points"] = raw_count
        return working, diagnostics, np.array([], dtype=np.float64)

    # 5. Exact duplicate records (all columns identical)
    exact_dup_mask = working.duplicated()
    diagnostics["exact_duplicates"] = int(exact_dup_mask.sum())
    if diagnostics["exact_duplicates"] > 0:
        working = working[~exact_dup_mask].copy()

    # 6. Identical-coordinate collisions at the same timestamp (same timestamp + identical lat/lon)
    # Deduplicate redundant spatial observations at identical timestamp, keeping first
    same_coord_dup_mask = working.duplicated(subset=["timestamp", "latitude", "longitude"], keep="first")
    diagnostics["duplicate_same_coord_removed"] = int(same_coord_dup_mask.sum())
    if diagnostics["duplicate_same_coord_removed"] > 0:
        working = working[~same_coord_dup_mask].copy()

    # 7. Chronological ordering check
    time_diffs = working["timestamp"].diff().dt.total_seconds()
    chronological_inversions = int((time_diffs < 0).sum())
    diagnostics["chronological_inversions"] = chronological_inversions
    if chronological_inversions > 0:
        working = working.sort_values(by="timestamp", kind="mergesort").reset_index(drop=True)
    else:
        working = working.reset_index(drop=True)

    # 8. Annotate legitimate same-second observations
    subsecond_seq = working.groupby("timestamp").cumcount().astype(np.int32)
    timestamp_counts = working.groupby("timestamp")["timestamp"].transform("count")
    timestamp_collision = (timestamp_counts > 1).astype(bool)

    working["subsecond_seq"] = subsecond_seq.values
    working["timestamp_collision"] = timestamp_collision.values

    # Count preserved same-timestamp observations (beyond the first fix of that second)
    preserved_same_ts = int((subsecond_seq > 0).sum())
    diagnostics["preserved_same_timestamp_observations"] = preserved_same_ts

    clean_count = len(working)
    diagnostics["clean_points"] = clean_count
    diagnostics["removed_points"] = raw_count - clean_count

    # 9. Trajectory duration & sampling intervals
    if clean_count >= 2:
        duration = (working["timestamp"].iloc[-1] - working["timestamp"].iloc[0]).total_seconds()
        diagnostics["duration_seconds"] = float(duration)
        all_diffs = working["timestamp"].diff().dt.total_seconds().dropna().to_numpy(dtype=np.float64)
        zero_diffs = all_diffs == 0.0
        diagnostics["zero_dt_count"] = int(np.sum(zero_diffs))
        # Keep non-zero intervals for physical sampling statistics
        positive_intervals = all_diffs[~zero_diffs]
    else:
        diagnostics["duration_seconds"] = 0.0
        diagnostics["zero_dt_count"] = 0
        positive_intervals = np.array([], dtype=np.float64)

    return working, diagnostics, positive_intervals


def _process_file_task(
    args: tuple[str, Path, IngestionConfig]
) -> tuple[pd.DataFrame | None, dict[str, Any], np.ndarray]:
    """Worker task to parse and validate a single file."""
    user_id, plt_path, config = args
    df, parse_diag = parse_plt_file(plt_path, user_id=user_id, config=config)
    if df is None:
        return None, parse_diag, np.array([], dtype=np.float64)

    clean_df, val_diag, intervals = validate_and_clean_trajectory(df, config=config)
    val_diag["file_path"] = str(plt_path)
    val_diag["user_id"] = user_id
    val_diag["success"] = True
    return clean_df, val_diag, intervals


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


def save_data_quality_report(
    report: dict[str, Any],
    output_dir: Path | str,
    config: IngestionConfig = IngestionConfig(),
) -> tuple[Path, Path]:
    """Save data quality report to JSON and Markdown formats."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / config.report_json_filename
    md_path = out_path / config.report_md_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    rem = report["records_removed"]
    dur = report["trajectory_durations_sec"]
    samp = report["sampling_intervals_sec"]

    md_content = f"""# GeoLife Trajectory Data Quality Report

## Dataset Summary
- **Total Users Discovered**: {report['users']}
- **Total Trajectories Discovered**: {report['trajectories_discovered']}
- **Total Trajectories Processed**: {report['trajectories_processed']}
- **Failed / Unreadable Files**: {len(report['failed_unreadable_files'])}
- **Total Raw GPS Points**: {report['total_raw_points']:,}
- **Total Clean GPS Points**: {report['total_clean_points']:,}
- **Total Points Removed**: {rem['total_removed']:,} ({(rem['total_removed'] / max(report['total_raw_points'], 1) * 100):.4f}%)
- **Preserved Same-Second Observations**: {report['preserved_same_second_observations']:,}
- **Earliest Timestamp**: {report['timestamp_range']['earliest']}
- **Latest Timestamp**: {report['timestamp_range']['latest']}

## Records Removed Breakdown
| Reason | Count | Description |
|---|---:|---|
| Missing Values (NaN) | {rem['missing_values']:,} | Missing latitude, longitude, or unparseable timestamp |
| Invalid Coordinates | {rem['invalid_coordinates']:,} | Latitude out of [-90, 90] or Longitude out of [-180, 180] |
| Unparseable Timestamps | {rem['unparseable_timestamps']:,} | Malformed date/time strings |
| Uninitialized RTC Timestamps | {rem['uninitialized_rtc_timestamps']:,} | Pre-2005 hardware clock reset artifacts (e.g. 2000-01-01) |
| Exact Duplicate Records | {rem['exact_duplicates']:,} | Identical rows logged redundantly across all fields |
| Same-Timestamp Identical-Coord | {rem['duplicate_same_coord_removed']:,} | Same-second collisions with identical coordinates |

## High-Rate Same-Second Observations (dt = 0)
- **Distinct Same-Second Observations Preserved**: {report['preserved_same_second_observations']:,}
- **Preservation Schema**: Annotated with `timestamp_collision = True` and deterministic `subsecond_seq` (0, 1, ...).
- **Physical Rate Protection**: Zero-duration intervals are flagged to ensure downstream velocity calculation does not divide by zero.

## Trajectory Duration Statistics (seconds)
| Metric | Value |
|---|---:|
| Count | {dur['count']:,} |
| Min | {dur['min']:.1f}s |
| Mean | {dur['mean']:.1f}s ({(dur['mean'] / 60):.1f} min) |
| Median | {dur['median']:.1f}s ({(dur['median'] / 60):.1f} min) |
| Std Dev | {dur['std']:.1f}s |
| 25th Percentile | {dur['p25']:.1f}s |
| 75th Percentile | {dur['p75']:.1f}s |
| 95th Percentile | {dur['p95']:.1f}s |
| Max | {dur['max']:.1f}s ({(dur['max'] / 3600):.1f} hrs) |

## GPS Sampling Interval Statistics (seconds, dt > 0)
*Calculated from actual timestamps without assuming a fixed sampling rate; excludes dt = 0 collisions.*

| Metric | Value |
|---|---:|
| Positive Interval Transitions | {samp['count']:,} |
| Same-Second Transitions (dt = 0) | {report['zero_dt_transitions']:,} |
| Min | {samp['min']:.1f}s |
| Mean | {samp['mean']:.2f}s |
| Median | {samp['median']:.1f}s |
| Std Dev | {samp['std']:.2f}s |
| 25th Percentile | {samp['p25']:.1f}s |
| 50th Percentile (Median) | {samp['p50']:.1f}s |
| 75th Percentile | {samp['p75']:.1f}s |
| 95th Percentile | {samp['p95']:.1f}s |
| 99th Percentile | {samp['p99']:.1f}s |
| Max | {samp['max']:.1f}s |

## Coordinate Bounding Box
- **Latitude Range**: [{report['coordinate_bounds']['min_latitude']:.6f}, {report['coordinate_bounds']['max_latitude']:.6f}]
- **Longitude Range**: [{report['coordinate_bounds']['min_longitude']:.6f}, {report['coordinate_bounds']['max_longitude']:.6f}]

## Preservation Rules Adherence
- **Same-Second Observations**: Preserved with collision indicators.
- **Stationary Points**: Intentionally preserved (not removed).
- **High Speeds / Kinematic Jumps**: Intentionally preserved (for downstream anomaly detection).
- **Altitude**: Preserved (raw values including sensor uncalibrated -777 preserved).
- **Trip Resampling / Segmentation**: Intentionally omitted at Chunk 1 stage.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def process_geolife_dataset(
    raw_dir: Path | str,
    output_dir: Path | str,
    config: IngestionConfig = IngestionConfig(),
    max_users: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Process the entire GeoLife dataset into a canonical Parquet dataset.

    Args:
        raw_dir: Path to raw GeoLife root directory.
        output_dir: Path to destination processed directory.
        config: Ingestion configuration parameters.
        max_users: Optional limit on number of users to process.

    Returns:
        Tuple of (parquet_file_path, data_quality_report_dict).
    """
    raw_path = Path(raw_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / config.canonical_filename

    discovered_files = discover_trajectory_files(raw_path)
    if not discovered_files:
        raise ValueError(f"No .plt files discovered in {raw_path}")

    if max_users is not None and max_users > 0:
        unique_users = sorted(list(dict.fromkeys(u for u, _ in discovered_files)))[:max_users]
        allowed_users = set(unique_users)
        discovered_files = [(u, f) for u, f in discovered_files if u in allowed_users]

    total_files = len(discovered_files)
    logger.info("Discovered %d trajectory files across users.", total_files)

    total_raw_points = 0
    total_clean_points = 0
    total_missing_values = 0
    total_invalid_coords = 0
    total_unparseable_ts = 0
    total_uninitialized_rtc_ts = 0
    total_exact_dups = 0
    total_same_coord_dups = 0
    total_preserved_same_ts = 0
    total_chronological_inversions = 0
    total_zero_dt = 0
    failed_files: list[str] = []
    durations_list: list[float] = []
    intervals_list: list[np.ndarray] = []
    processed_users: set[str] = set()
    processed_trajectories = 0

    min_lat = float("inf")
    max_lat = float("-inf")
    min_lon = float("inf")
    max_lon = float("-inf")
    earliest_ts = pd.Timestamp.max
    latest_ts = pd.Timestamp.min

    parquet_writer: pq.ParquetWriter | None = None
    target_schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("trajectory_id", pa.string()),
            ("timestamp", pa.timestamp("ns")),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("altitude", pa.float64()),
            ("raw_days", pa.float64()),
            ("date_str", pa.string()),
            ("time_str", pa.string()),
            ("subsecond_seq", pa.int32()),
            ("timestamp_collision", pa.bool_()),
        ]
    )

    batch_frames: list[pd.DataFrame] = []

    tasks = [(user_id, f, config) for user_id, f in discovered_files]
    chunk_size = config.batch_size

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        for i in range(0, total_files, chunk_size):
            task_chunk = tasks[i:i + chunk_size]
            results = executor.map(_process_file_task, task_chunk)

            for clean_df, diag, intervals in results:
                if clean_df is None or not diag.get("success", False):
                    failed_files.append(diag.get("file_path", "unknown"))
                    continue

                user_id = diag["user_id"]
                processed_users.add(user_id)
                processed_trajectories += 1

                total_raw_points += diag["raw_points"]
                total_clean_points += diag["clean_points"]
                total_missing_values += diag["missing_values"]
                total_invalid_coords += diag["invalid_coordinates"]
                total_unparseable_ts += diag["unparseable_timestamps"]
                total_uninitialized_rtc_ts += diag["uninitialized_rtc_timestamps"]
                total_exact_dups += diag["exact_duplicates"]
                total_same_coord_dups += diag["duplicate_same_coord_removed"]
                total_preserved_same_ts += diag["preserved_same_timestamp_observations"]
                total_chronological_inversions += diag["chronological_inversions"]
                total_zero_dt += diag["zero_dt_count"]

                if diag["clean_points"] >= 2:
                    durations_list.append(diag["duration_seconds"])

                if len(intervals) > 0:
                    intervals_list.append(intervals)

                if not clean_df.empty:
                    batch_frames.append(clean_df)

                    cur_min_lat = float(clean_df["latitude"].min())
                    cur_max_lat = float(clean_df["latitude"].max())
                    cur_min_lon = float(clean_df["longitude"].min())
                    cur_max_lon = float(clean_df["longitude"].max())
                    cur_min_ts = clean_df["timestamp"].min()
                    cur_max_ts = clean_df["timestamp"].max()

                    if cur_min_lat < min_lat:
                        min_lat = cur_min_lat
                    if cur_max_lat > max_lat:
                        max_lat = cur_max_lat
                    if cur_min_lon < min_lon:
                        min_lon = cur_min_lon
                    if cur_max_lon > max_lon:
                        max_lon = cur_max_lon
                    if cur_min_ts < earliest_ts:
                        earliest_ts = cur_min_ts
                    if cur_max_ts > latest_ts:
                        latest_ts = cur_max_ts

            if batch_frames:
                combined_batch = pd.concat(batch_frames, ignore_index=True)
                table = pa.Table.from_pandas(combined_batch, schema=target_schema, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(parquet_path, target_schema, compression="snappy")
                parquet_writer.write_table(table)
                batch_frames.clear()

            logger.info("Processed %d / %d files...", min(i + chunk_size, total_files), total_files)

    if batch_frames:
        combined_batch = pd.concat(batch_frames, ignore_index=True)
        table = pa.Table.from_pandas(combined_batch, schema=target_schema, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_path, target_schema, compression="snappy")
        parquet_writer.write_table(table)
        batch_frames.clear()

    if parquet_writer is not None:
        parquet_writer.close()

    all_intervals = np.concatenate(intervals_list) if intervals_list else np.array([], dtype=np.float64)
    all_durations = np.array(durations_list, dtype=np.float64)

    duration_stats = calculate_distribution_stats(all_durations)
    interval_stats = calculate_distribution_stats(all_intervals)

    total_removed = total_raw_points - total_clean_points

    report: dict[str, Any] = {
        "dataset_name": "Microsoft GeoLife GPS Trajectories",
        "users": len(processed_users),
        "trajectories_discovered": total_files,
        "trajectories_processed": processed_trajectories,
        "failed_unreadable_files": failed_files,
        "total_raw_points": total_raw_points,
        "total_clean_points": total_clean_points,
        "records_removed": {
            "total_removed": total_removed,
            "missing_values": total_missing_values,
            "invalid_coordinates": total_invalid_coords,
            "unparseable_timestamps": total_unparseable_ts,
            "uninitialized_rtc_timestamps": total_uninitialized_rtc_ts,
            "exact_duplicates": total_exact_dups,
            "duplicate_same_coord_removed": total_same_coord_dups,
        },
        "preserved_same_second_observations": total_preserved_same_ts,
        "zero_dt_transitions": total_zero_dt,
        "chronological_inversions_detected": total_chronological_inversions,
        "timestamp_range": {
            "earliest": str(earliest_ts) if earliest_ts != pd.Timestamp.max else "",
            "latest": str(latest_ts) if latest_ts != pd.Timestamp.min else "",
        },
        "coordinate_bounds": {
            "min_latitude": min_lat if min_lat != float("inf") else 0.0,
            "max_latitude": max_lat if max_lat != float("-inf") else 0.0,
            "min_longitude": min_lon if min_lon != float("inf") else 0.0,
            "max_longitude": max_lon if max_lon != float("-inf") else 0.0,
        },
        "trajectory_durations_sec": duration_stats,
        "sampling_intervals_sec": interval_stats,
        "output_parquet_file": str(parquet_path),
    }

    save_data_quality_report(report, out_path, config=config)
    return parquet_path, report


def main() -> None:
    """CLI entrypoint for running ingestion, validation, and minimal cleaning."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    default_raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "geolife"
    default_out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    parser = argparse.ArgumentParser(description="GeoLife Ingestion, Validation & Minimal Cleaning Pipeline")
    parser.add_argument("--raw-dir", type=str, default=str(default_raw_dir), help="Path to raw GeoLife directory")
    parser.add_argument("--output-dir", type=str, default=str(default_out_dir), help="Path to processed output directory")
    parser.add_argument("--max-users", type=int, default=None, help="Optional maximum number of users to process")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of concurrent worker threads")
    parser.add_argument(
        "--min-timestamp",
        type=str,
        default="2005-01-01",
        help="Earliest valid timestamp threshold (filters pre-2005 GPS uninitialized RTC reset artifacts)",
    )

    args = parser.parse_args()

    config = IngestionConfig(num_workers=args.num_workers, min_valid_timestamp=args.min_timestamp)
    print("=" * 70)
    print("Starting GeoLife Trajectory Ingestion Pipeline (Chunk 1 - Corrected)")
    print(f"Raw directory:    {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max users:        {args.max_users or 'ALL'}")
    print(f"Workers:          {args.num_workers}")
    print("=" * 70)

    parquet_path, report = process_geolife_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=config,
        max_users=args.max_users,
    )

    print("\nProcessing complete!")
    print(f"Canonical dataset saved to: {parquet_path}")
    print(f"Users processed:            {report['users']}")
    print(f"Trajectories processed:     {report['trajectories_processed']}")
    print(f"Total raw points:           {report['total_raw_points']:,}")
    print(f"Total clean points:         {report['total_clean_points']:,}")
    print(f"Total points removed:       {report['records_removed']['total_removed']:,}")
    print(f"Preserved same-second obs:  {report['preserved_same_second_observations']:,}")
    print(f"Earliest timestamp:         {report['timestamp_range']['earliest']}")
    print(f"Latest timestamp:           {report['timestamp_range']['latest']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

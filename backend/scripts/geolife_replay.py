#!/usr/bin/env python3
"""
GeoLife Trajectory Replay CLI for Mobility Guardian.

Parses GeoLife .plt trajectory files and streams them as valid TelemetryPayload
records to the live backend telemetry ingestion endpoint (/api/telemetry/ingest).

Usage example:
    python backend/scripts/geolife_replay.py \\
        --plt-file path/to/trajectory.plt \\
        --user-id test_geolife_01 \\
        --playback-speed 5.0 \\
        --time-mode live
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Earth radius in meters for spherical geodesy
EARTH_RADIUS_M = 6371000.0


@dataclass
class GeoLifePoint:
    """Raw parsed record from a GeoLife .plt file."""
    latitude: float
    longitude: float
    altitude_m: Optional[float]
    timestamp: datetime
    line_number: int


@dataclass
class ReplayPoint:
    """Enriched telemetry fix ready for ingestion payload construction."""
    latitude: float
    longitude: float
    altitude_m: Optional[float]
    timestamp: datetime
    speed_mps: float
    heading_deg: float
    line_number: int


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two coordinates in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward azimuth bearing from (lat1, lon1) to (lat2, lon2) in degrees [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - (
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    bearing = math.degrees(math.atan2(y, x))
    return round((bearing + 360.0) % 360.0, 1)


def parse_plt_file(
    file_path: Path | str,
    max_points: Optional[int] = None,
) -> Tuple[List[GeoLifePoint], Dict[str, Any]]:
    """Parse a GeoLife .plt file, skipping standard 6-line header and validating points.

    GeoLife .plt format specification:
    Lines 1-6: Metadata header
    Line 7+: Latitude, Longitude, Reserved(0), Altitude(ft), Days(float), Date(yyyy-mm-dd), Time(hh:mm:ss)

    Returns:
        (sorted_valid_points, diagnostic_summary)
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GeoLife .plt file not found: {path}")

    valid_points: List[GeoLifePoint] = []
    skipped_records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_lines_count = 6
    if len(lines) <= header_lines_count:
        return [], {
            "total_lines": len(lines),
            "valid_count": 0,
            "skipped_count": 0,
            "error": "File contains only headers or is empty",
        }

    for idx, raw_line in enumerate(lines[header_lines_count:], start=header_lines_count + 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            skipped_records.append({
                "line": idx,
                "reason": f"Expected at least 7 comma-separated fields, got {len(parts)}",
                "raw": line,
            })
            continue

        try:
            lat = float(parts[0])
            lon = float(parts[1])
            alt_ft = float(parts[3])
            date_str = parts[5]
            time_str = parts[6]
        except (ValueError, IndexError) as err:
            skipped_records.append({
                "line": idx,
                "reason": f"Failed to parse numeric/string values: {err}",
                "raw": line,
            })
            continue

        # Physical coordinate bounds validation
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            skipped_records.append({
                "line": idx,
                "reason": f"Coordinates out of physical bounds: lat={lat}, lon={lon}",
                "raw": line,
            })
            continue

        # Parse timestamp (GeoLife format: yyyy-mm-dd and hh:mm:ss in UTC)
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError as err:
            skipped_records.append({
                "line": idx,
                "reason": f"Invalid date/time format ({date_str} {time_str}): {err}",
                "raw": line,
            })
            continue

        # Convert altitude: -777 indicates uncalibrated / invalid sensor reading
        alt_m = round(alt_ft * 0.3048, 1) if alt_ft != -777.0 else None

        valid_points.append(
            GeoLifePoint(
                latitude=lat,
                longitude=lon,
                altitude_m=alt_m,
                timestamp=dt,
                line_number=idx,
            )
        )

        if max_points is not None and len(valid_points) >= max_points:
            break

    # Ensure strictly non-decreasing chronological order
    valid_points.sort(key=lambda p: p.timestamp)

    diagnostics = {
        "file_path": str(path),
        "total_lines": len(lines),
        "data_lines": max(0, len(lines) - header_lines_count),
        "valid_count": len(valid_points),
        "skipped_count": len(skipped_records),
        "skipped_details": skipped_records,
    }
    return valid_points, diagnostics


def derive_kinematic_features(
    points: List[GeoLifePoint],
    time_mode: str = "original",
    target_start_time: Optional[datetime] = None,
) -> List[ReplayPoint]:
    """Derive speed, bearing, and adjust timestamps if live mode is requested.

    Time modes:
    - 'original': Preserves historical timestamps directly from the .plt file.
    - 'live': Shifts timestamps so the first point anchors to target_start_time (or now UTC),
              preserving exact inter-point time intervals (dt).
    """
    if not points:
        return []

    t0_original = points[0].timestamp
    if target_start_time is None:
        target_start_time = datetime.now(timezone.utc)
    elif target_start_time.tzinfo is None:
        target_start_time = target_start_time.replace(tzinfo=timezone.utc)

    enriched: List[ReplayPoint] = []
    prev_heading = 0.0

    for i, pt in enumerate(points):
        if time_mode == "live":
            elapsed = (pt.timestamp - t0_original).total_seconds()
            effective_time = target_start_time + timedelta(seconds=elapsed)
        else:
            effective_time = pt.timestamp

        if i == 0:
            speed_mps = 0.0
            heading_deg = 0.0
        else:
            prev_pt = points[i - 1]
            dist_m = haversine_distance(
                prev_pt.latitude, prev_pt.longitude,
                pt.latitude, pt.longitude
            )
            dt_sec = (pt.timestamp - prev_pt.timestamp).total_seconds()
            if dt_sec > 0:
                speed_mps = round(dist_m / dt_sec, 2)
                # Keep physically plausible upper bound for pedestrian/urban transit
                speed_mps = min(speed_mps, 120.0)
            else:
                speed_mps = 0.0

            if dist_m > 0.5:
                heading_deg = calculate_bearing(
                    prev_pt.latitude, prev_pt.longitude,
                    pt.latitude, pt.longitude
                )
                prev_heading = heading_deg
            else:
                heading_deg = prev_heading

        enriched.append(
            ReplayPoint(
                latitude=pt.latitude,
                longitude=pt.longitude,
                altitude_m=pt.altitude_m,
                timestamp=effective_time,
                speed_mps=speed_mps,
                heading_deg=heading_deg,
                line_number=pt.line_number,
            )
        )

    return enriched


def build_telemetry_payload(
    point: ReplayPoint,
    user_id: str,
    battery_pct: int = 85,
) -> Dict[str, Any]:
    """Construct schema-valid TelemetryPayload conforming to /shared/schema.json."""
    return {
        "user_id": user_id,
        "timestamp": point.timestamp.isoformat(),
        "location": {
            "lat": round(point.latitude, 7),
            "lng": round(point.longitude, 7),
            "altitude_m": point.altitude_m,
        },
        "sensor_metrics": {
            "horizontal_accuracy_m": 5.0,
            "speed_mps": max(0.0, float(point.speed_mps)),
            "heading_deg": max(0.0, min(360.0, float(point.heading_deg))),
            "activity_type": "WALKING" if point.speed_mps >= 0.2 else "STATIONARY",
            "battery_pct": int(battery_pct),
        },
        "signal_status": {
            "state": "VALID",
            "degraded_since": None,
        },
        "imu_metrics": None,
    }


def register_consent(backend_url: str, user_id: str, timeout_sec: float = 10.0) -> bool:
    """Record dual-consent before streaming telemetry to satisfy authorization check."""
    url = f"{backend_url.rstrip('/')}/api/consent/record"
    payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_geolife_replay",
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(url, json=payload)
        if resp.status_code == 200:
            print(f"[Consent] Granted consent recorded for user '{user_id}'.")
            return True
        else:
            print(f"[Error] Failed to record consent: HTTP {resp.status_code} - {resp.text}")
            return False


def stream_geolife_trajectory(
    points: List[ReplayPoint],
    user_id: str,
    backend_url: str = "http://localhost:8000",
    playback_speed: float = 1.0,
    interval_seconds: Optional[float] = None,
    battery_start_pct: int = 90,
    timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """Stream parsed points to /api/telemetry/ingest with configurable pacing."""
    url = f"{backend_url.rstrip('/')}/api/telemetry/ingest"
    total = len(points)
    accepted = 0
    rejected = 0
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 105)
    print(f"{'Step':<8} {'Location (lat, lng)':<24} {'Spd':<9} {'Bat':<6} {'Op Score':<10} {'Op Tier':<14} {'ML Shadow Evaluation':<30}")
    print("=" * 105)

    with httpx.Client(timeout=timeout_sec) as client:
        for idx, pt in enumerate(points):
            battery = max(5, int(battery_start_pct - (idx * 0.05)))
            payload = build_telemetry_payload(pt, user_id=user_id, battery_pct=battery)

            try:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    accepted += 1
                    data = resp.json()
                    op_score = data.get("risk_score")
                    op_tier = data.get("risk_tier", "UNKNOWN")
                    ml_eval = data.get("ml_evaluation")
                    if ml_eval:
                        ml_eng = ml_eval.get("engine_status", "-")
                        ml_evi = ml_eval.get("evidence_status", "-")
                        cal_p = ml_eval.get("calibrated_probability")
                        p_str = f" ({cal_p * 100:.1f}%)" if cal_p is not None else ""
                        ml_desc = f"{ml_eng} / {ml_evi}{p_str}"
                    else:
                        ml_desc = "OFF (flag disabled)"

                    score_str = f"{op_score:.1f}" if op_score is not None else "N/A"
                    print(
                        f"[{idx+1:02d}/{total:02d}] "
                        f"({pt.latitude:.5f}, {pt.longitude:.5f})   "
                        f"{pt.speed_mps:>4.1f}m/s  "
                        f"{battery:>3d}%  "
                        f"{score_str:>8}  "
                        f"{op_tier:<14}  "
                        f"{ml_desc:<30}"
                    )
                    results.append({
                        "step": idx + 1,
                        "success": True,
                        "status_code": 200,
                        "op_score": op_score,
                        "op_tier": op_tier,
                        "ml_evaluation": ml_eval,
                    })
                else:
                    rejected += 1
                    print(f"[{idx+1:02d}/{total:02d}] HTTP {resp.status_code} Error: {resp.text}")
                    results.append({
                        "step": idx + 1,
                        "success": False,
                        "status_code": resp.status_code,
                        "error": resp.text,
                    })
            except Exception as exc:
                rejected += 1
                print(f"[{idx+1:02d}/{total:02d}] Request exception: {exc}")
                results.append({
                    "step": idx + 1,
                    "success": False,
                    "status_code": None,
                    "error": str(exc),
                })

            # Calculate pacing delay before next point
            if idx < total - 1:
                if interval_seconds is not None:
                    delay = max(0.0, float(interval_seconds))
                elif playback_speed > 0:
                    dt = (points[idx + 1].timestamp - pt.timestamp).total_seconds()
                    delay = max(0.0, dt / playback_speed)
                else:
                    delay = 0.0

                if delay > 0:
                    time.sleep(min(delay, 10.0))

    print("=" * 105)
    print(f"Stream complete: {accepted}/{total} points accepted ({rejected} rejected).")
    return {
        "total_points": total,
        "accepted": accepted,
        "rejected": rejected,
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoLife Trajectory Replay CLI for Mobility Guardian Live Ingestion"
    )
    parser.add_argument("--plt-file", "--input", required=True, help="Path to raw GeoLife .plt file")
    parser.add_argument("--user-id", required=True, help="Target user ID to replay telemetry for")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="FastAPI backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--time-mode",
        choices=["live", "original"],
        default="live",
        help="Timestamp mode: 'live' maps start to now (recommended for live dashboard freshness), 'original' preserves source historical timestamps",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=5.0,
        help="Speed multiplier when pacing by timestamp dt (default: 5.0x)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Explicit fixed delay in seconds between HTTP POSTs (overrides playback-speed)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional limit on maximum number of points to read from the .plt file",
    )
    parser.add_argument(
        "--no-consent",
        action="store_true",
        help="Skip automatic consent registration (assumes consent already recorded)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds (default: 10.0)",
    )

    args = parser.parse_args()

    plt_path = Path(args.plt_file)
    print(f"\n--- GeoLife Trajectory Replay Initialization ---")
    print(f"Input File:     {plt_path}")
    print(f"User ID:        {args.user_id}")
    print(f"Backend URL:    {args.backend_url}")
    print(f"Time Mode:      {args.time_mode}")
    print(f"Playback Speed: {args.playback_speed}x" + (f" (fixed interval: {args.interval_seconds}s)" if args.interval_seconds is not None else ""))

    # 1. Parse .plt file
    raw_points, diag = parse_plt_file(plt_path, max_points=args.max_points)
    print(f"Points parsed:  {diag['valid_count']} valid ({diag['skipped_count']} skipped out of {diag['data_lines']} data lines)")
    if diag.get("skipped_count", 0) > 0:
        for sk in diag["skipped_details"][:3]:
            print(f"  [Notice] Skipped line {sk['line']}: {sk['reason']}")
        if len(diag["skipped_details"]) > 3:
            print(f"  ... and {len(diag['skipped_details']) - 3} more skipped lines.")

    if not raw_points:
        print("[Error] No valid trajectory points found in .plt file. Aborting.")
        sys.exit(1)

    # 2. Enrich points with speed and heading
    replay_points = derive_kinematic_features(raw_points, time_mode=args.time_mode)

    # 3. Register dual-consent
    if not args.no_consent:
        ok = register_consent(args.backend_url, args.user_id, timeout_sec=args.timeout)
        if not ok:
            print("[Error] Cannot proceed without consent. Exiting.")
            sys.exit(1)

    # 4. Stream points
    stream_geolife_trajectory(
        points=replay_points,
        user_id=args.user_id,
        backend_url=args.backend_url,
        playback_speed=args.playback_speed,
        interval_seconds=args.interval_seconds,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    main()

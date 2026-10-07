#!/usr/bin/env python3
"""
Standalone GPS Trajectory Simulator CLI.

Simulates human movement and wandering trajectories according to the
Algase Wandering Typology (Pacing, Lapping, Random Drift) and normal safe movement.
Constructs valid TelemetryPayload data and streams them to the predictive geofencing backend.

Usage example:
    python backend/scripts/trajectory_simulator.py \\
        --user-id sim_01 \\
        --pattern lapping \\
        --anchor-lat 13.0827 \\
        --anchor-lng 80.2707 \\
        --num-points 30 \\
        --interval-seconds 0.1
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Geodesic conversion constants
METERS_PER_DEGREE_LAT = 111320.0


def lat_lng_offset(
    anchor_lat: float, anchor_lng: float, offset_x_m: float, offset_y_m: float
) -> Tuple[float, float]:
    """
    Convert local tangent-plane offsets (meters East, meters North)
    to (lat, lng) coordinates using equirectangular projection.
    """
    lat = anchor_lat + (offset_y_m / METERS_PER_DEGREE_LAT)
    meters_per_degree_lng = METERS_PER_DEGREE_LAT * math.cos(math.radians(anchor_lat))
    if abs(meters_per_degree_lng) < 1e-6:
        meters_per_degree_lng = 1e-6
    lng = anchor_lng + (offset_x_m / meters_per_degree_lng)
    return lat, lng


def calculate_heading(dx: float, dy: float, prev_heading: float) -> float:
    """
    Compute geographic compass heading (0 = North, 90 = East, 180 = South, 270 = West)
    from displacement vector (dx East, dy North).
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return prev_heading
    heading = math.degrees(math.atan2(dx, dy)) % 360.0
    return round(heading, 1)


class MovementGenerator:
    """Generates (x, y) coordinates in meters relative to (0, 0) anchor."""

    def __init__(
        self,
        pattern: str,
        num_points: int,
        interval_seconds: float,
        speed_mps: float = 1.2,
    ):
        self.pattern = pattern
        self.num_points = num_points
        self.interval = interval_seconds
        self.speed = speed_mps

    def generate(self) -> List[Tuple[float, float, float, float]]:
        """
        Returns list of tuples: (x_m, y_m, speed_mps, heading_deg).
        """
        if self.pattern == "normal":
            return self._generate_normal()
        elif self.pattern == "pacing":
            return self._generate_pacing()
        elif self.pattern == "lapping":
            return self._generate_lapping()
        elif self.pattern == "random_drift":
            return self._generate_random_drift()
        else:
            raise ValueError(f"Unknown movement pattern: {self.pattern}")

    def _generate_normal(self) -> List[Tuple[float, float, float, float]]:
        """
        Normal pattern: Random walk bounded within ~200m of anchor,
        realistic walking speed (~1.2 m/s), low tortuosity, periodically returning home.
        """
        points = []
        x, y = 0.0, 0.0
        heading = random.uniform(0, 360)
        step_dist = self.speed * self.interval

        for i in range(self.num_points):
            r = math.hypot(x, y)
            # If drifting beyond 140m or periodically every 15 steps, guide back toward anchor
            if r > 140.0 or (i > 0 and i % 15 == 0):
                target_heading = math.degrees(math.atan2(-x, -y)) % 360.0
                # Smooth turn towards target
                diff = (target_heading - heading + 180.0) % 360.0 - 180.0
                heading = (heading + diff * 0.4) % 360.0
            else:
                # Low tortuosity random heading adjustment (std dev ~ 12 deg)
                heading = (heading + random.gauss(0, 12.0)) % 360.0

            dx = step_dist * math.sin(math.radians(heading))
            dy = step_dist * math.cos(math.radians(heading))
            x += dx
            y += dy

            speed_actual = round(max(0.2, self.speed + random.uniform(-0.1, 0.1)), 2)
            points.append((x, y, speed_actual, round(heading, 1)))

        return points

    def _generate_pacing(self) -> List[Tuple[float, float, float, float]]:
        """
        Algase Pacing phenotype: Repetitive 180-degree directional reversals
        along a short linear segment (e.g. hallway, length ~ 30m).
        """
        points = []
        segment_length = 30.0  # meters
        axis_angle_deg = 45.0  # Orientation bearing of hallway
        direction = 1.0  # +1 moving forward along segment, -1 moving backward
        pos_along_segment = -segment_length / 2.0  # start at one end
        step_dist = self.speed * self.interval

        heading = axis_angle_deg

        for _ in range(self.num_points):
            pos_along_segment += direction * step_dist

            # Check boundary reversal
            if pos_along_segment >= segment_length / 2.0:
                pos_along_segment = segment_length / 2.0
                direction = -1.0
                heading = (axis_angle_deg + 180.0) % 360.0
            elif pos_along_segment <= -segment_length / 2.0:
                pos_along_segment = -segment_length / 2.0
                direction = 1.0
                heading = axis_angle_deg

            x = pos_along_segment * math.sin(math.radians(axis_angle_deg))
            y = pos_along_segment * math.cos(math.radians(axis_angle_deg))

            points.append((x, y, round(self.speed, 2), round(heading, 1)))

        return points

    def _generate_lapping(self) -> List[Tuple[float, float, float, float]]:
        """
        Algase Lapping phenotype: Closed-loop circular/oval trajectory with
        turning radius < 100m (configured at ~ 45m), high turning-angle frequency.
        """
        points = []
        radius = 45.0  # meters (< 100m turning radius)
        # Angular speed omega = v / R in rad/s
        omega = self.speed / radius
        theta = 0.0

        for _ in range(self.num_points):
            x = radius * math.sin(theta)
            y = radius * math.cos(theta)

            # Velocity tangent direction
            tangent_dx = radius * omega * math.cos(theta)
            tangent_dy = -radius * omega * math.sin(theta)
            heading = calculate_heading(tangent_dx, tangent_dy, 0.0)

            points.append((x, y, round(self.speed, 2), heading))
            theta += omega * self.interval

        return points

    def _generate_random_drift(self) -> List[Tuple[float, float, float, float]]:
        """
        Algase Random Drift (Direct-exit) phenotype: Continuous forward progression
        with heading-change variance > 45 degrees per step, drifting progressively
        away from anchor over time.
        """
        points = []
        x, y = 0.0, 0.0
        # Start oriented along an escape bearing (e.g. Northeast)
        exit_bearing = 55.0
        heading = exit_bearing
        step_dist = self.speed * self.interval

        for _ in range(self.num_points):
            # High heading-change variance: std dev = 50 deg (> 45 deg)
            d_heading = random.gauss(0, 50.0)
            heading = (heading + d_heading) % 360.0

            # Forward movement with strong persistent outward drift bias
            dx = step_dist * (math.sin(math.radians(heading)) + 0.8 * math.sin(math.radians(exit_bearing)))
            dy = step_dist * (math.cos(math.radians(heading)) + 0.8 * math.cos(math.radians(exit_bearing)))
            x += dx
            y += dy

            actual_heading = calculate_heading(dx, dy, heading)
            actual_speed = round(math.hypot(dx, dy) / self.interval, 2)
            points.append((x, y, actual_speed, actual_heading))

        return points


def register_consent(backend_url: str, user_id: str) -> bool:
    """
    POST a valid ConsentRecord to /api/consent/record before streaming telemetry.
    Returns True if successfully recorded.
    """
    url = f"{backend_url.rstrip('/')}/api/consent/record"
    consent_payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "guardian_sim_auto",
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=consent_payload)
            if resp.status_code == 200:
                print(f"[Consent] Granted consent recorded for user '{user_id}'.")
                return True
            else:
                print(f"[Error] Failed to register consent: HTTP {resp.status_code} - {resp.text}")
                return False
    except httpx.ConnectError:
        print(f"\n[Error] Could not connect to backend at {backend_url}.")
        print("Please ensure the FastAPI server is running (e.g. uvicorn app.main:app --port 8000).")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Connection error while registering consent: {e}")
        sys.exit(1)


def build_telemetry_payload(
    user_id: str,
    timestamp_iso: str,
    lat: float,
    lng: float,
    speed_mps: float,
    heading_deg: float,
    battery_pct: int,
    is_degraded: bool,
    degraded_since_iso: Optional[str] = None,
    net_displacement_m: float = 0.0,
    step_count: int = 0,
) -> Dict[str, Any]:
    """
    Construct a TelemetryPayload conforming strictly to /shared/schema.json.
    """
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "timestamp": timestamp_iso,
        "location": {
            "lat": round(lat, 7),
            "lng": round(lng, 7),
            "altitude_m": 15.0,
        },
        "sensor_metrics": {
            "horizontal_accuracy_m": 25.0 if is_degraded else 5.0,
            "speed_mps": speed_mps,
            "heading_deg": heading_deg,
            "activity_type": "WALKING" if speed_mps >= 0.2 else "STATIONARY",
            "battery_pct": battery_pct,
        },
        "signal_status": {
            "state": "DEGRADED_SIGNAL" if is_degraded else "VALID",
            "degraded_since": degraded_since_iso if is_degraded else None,
        },
    }

    if is_degraded:
        # Schema-completeness only: client-side pdr_tier_state stub
        payload["imu_metrics"] = {
            "step_count_since_last_gps": step_count,
            "net_displacement_m": round(net_displacement_m, 2),
            "pdr_tier_state": "INDOOR_PACING",
        }
    else:
        payload["imu_metrics"] = None

    return payload


def stream_telemetry(
    backend_url: str,
    payloads: List[Dict[str, Any]],
    interval_seconds: float,
) -> List[Dict[str, Any]]:
    """
    POST telemetry payloads sequentially to /api/telemetry/ingest.
    Print live backend risk evaluation results and return rows for CSV export.
    """
    url = f"{backend_url.rstrip('/')}/api/telemetry/ingest"
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 96)
    print(
        f"{'Step':<8} {'Location (lat, lng)':<24} {'Spd':<8} {'Bat':<6} {'Signal':<10} "
        f"{'Risk Tier':<14} {'Score':<8} {'Polling Instruction'}"
    )
    print("=" * 96)

    with httpx.Client(timeout=10.0) as client:
        for idx, payload in enumerate(payloads, 1):
            try:
                resp = client.post(url, json=payload)
            except httpx.ConnectError:
                print(f"\n[Error] Connection lost to backend at point {idx}/{len(payloads)}.")
                sys.exit(1)
            except Exception as e:
                print(f"\n[Error] HTTP request failed at point {idx}: {e}")
                sys.exit(1)

            if resp.status_code != 200:
                print(f"[Pt {idx:02d}/{len(payloads):02d}] HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()

            # Values sourced strictly from backend response
            risk_tier = data.get("risk_tier", "UNKNOWN")
            risk_score = data.get("risk_score", 0.0)
            polling_inst = data.get("polling_instruction") or {}
            polling_mode = polling_inst.get("mode", "UNKNOWN")

            # Format polling description
            poll_desc = polling_mode
            if polling_mode == "PULSE":
                b_sec = polling_inst.get("burst_seconds")
                s_sec = polling_inst.get("sleep_seconds")
                poll_desc = f"PULSE [burst: {b_sec}s, sleep: {s_sec}s]"
            elif polling_mode == "LAST_GASP":
                poll_desc = "LAST_GASP [SOS protocol]"

            loc = payload["location"]
            metrics = payload["sensor_metrics"]
            sig = payload["signal_status"]["state"]
            bat = metrics["battery_pct"]
            spd = metrics["speed_mps"]

            sig_label = "VALID" if sig == "VALID" else "DEGRADED"

            loc_str = f"({loc['lat']:.5f}, {loc['lng']:.5f})"
            print(
                f"[{idx:02d}/{len(payloads):02d}]  "
                f"{loc_str:<24} "
                f"{spd:4.1f}m/s  "
                f"{bat:3d}%   "
                f"{sig_label:<10} "
                f"{risk_tier:<14} "
                f"{risk_score:5.1f}   "
                f"{poll_desc}"
            )

            # Record for CSV
            imu = payload.get("imu_metrics") or {}
            results.append({
                "step": idx,
                "timestamp": payload["timestamp"],
                "lat": loc["lat"],
                "lng": loc["lng"],
                "speed_mps": spd,
                "heading_deg": metrics["heading_deg"],
                "battery_pct": bat,
                "signal_state": sig,
                "net_displacement_m": imu.get("net_displacement_m", 0.0),
                "risk_tier": risk_tier,
                "risk_score": risk_score,
                "polling_mode": polling_mode,
            })

            if idx < len(payloads):
                time.sleep(interval_seconds)

    print("=" * 96)
    return results


def export_csv(records: List[Dict[str, Any]], filepath: str) -> None:
    """Export run metrics to CSV for plotting and post-run inspection."""
    if not records:
        return
    fieldnames = [
        "step",
        "timestamp",
        "lat",
        "lng",
        "speed_mps",
        "heading_deg",
        "battery_pct",
        "signal_state",
        "net_displacement_m",
        "risk_tier",
        "risk_score",
        "polling_mode",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[Export] Trajectory saved to: {filepath}")


def parse_degraded_window(window_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse 'start:end' string into 0-indexed (start, end) range."""
    if not window_str:
        return None
    try:
        parts = window_str.split(":")
        if len(parts) != 2:
            raise ValueError
        start, end = int(parts[0]), int(parts[1])
        if start < 0 or end <= start:
            raise ValueError
        return start, end
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid --degraded-signal-window '{window_str}'. Expected format 'start:end' (e.g. '20:30')."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPS Trajectory Simulator for Predictive Geofencing System"
    )
    parser.add_argument("--user-id", required=True, help="Simulated user identifier (e.g. sim_01)")
    parser.add_argument(
        "--pattern",
        required=True,
        choices=["normal", "pacing", "lapping", "random_drift"],
        help="Movement phenotype pattern",
    )
    parser.add_argument("--anchor-lat", type=float, required=True, help="Anchor latitude")
    parser.add_argument("--anchor-lng", type=float, required=True, help="Anchor longitude")
    parser.add_argument("--num-points", type=int, default=50, help="Number of points to generate (default: 50)")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Delay between points in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=None,
        help="Simulated time interval between fixes in seconds (default: 2.0 or interval-seconds if >= 2.0)",
    )
    parser.add_argument(
        "--battery-start-pct",
        type=float,
        default=80.0,
        help="Initial battery percentage (default: 80.0)",
    )
    parser.add_argument(
        "--battery-drain-rate",
        type=float,
        default=0.05,
        help="Battery drained percentage per point (default: 0.05)",
    )
    parser.add_argument(
        "--degraded-signal-window",
        type=str,
        default=None,
        help="Point index window for degraded signal, e.g. '20:30'",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=1.2,
        help="Base movement speed in m/s (default: 1.2)",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Custom file path for trajectory CSV export",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable automatic CSV export",
    )

    args = parser.parse_args()

    # Time delta between simulated GPS fixes
    sim_dt = args.step_seconds if args.step_seconds is not None else max(args.interval_seconds, 2.0)
    degraded_range = parse_degraded_window(args.degraded_signal_window)

    print("\n--- GPS Trajectory Simulator Initialization ---")
    print(f"User ID:        {args.user_id}")
    print(f"Pattern:        {args.pattern}")
    print(f"Anchor:         ({args.anchor_lat}, {args.anchor_lng})")
    print(f"Points:         {args.num_points}")
    print(f"Network Delay:  {args.interval_seconds}s")
    print(f"Sim Time-Step:  {sim_dt}s")
    print(f"Battery:        {args.battery_start_pct}% (-{args.battery_drain_rate}%/pt)")
    if degraded_range:
        print(f"Degraded Win:   Points {degraded_range[0]} to {degraded_range[1]}")
    print(f"Backend URL:    {args.backend_url}")

    # 1. Register consent
    register_consent(args.backend_url, args.user_id)

    # 2. Generate trajectory points
    gen = MovementGenerator(
        pattern=args.pattern,
        num_points=args.num_points,
        interval_seconds=sim_dt,
        speed_mps=args.speed_mps,
    )
    coords_m = gen.generate()

    # 3. Construct TelemetryPayloads
    payloads = []
    start_time = datetime.now(timezone.utc)

    # Keep track of last valid GPS position for PDR displacement calculation
    last_valid_x, last_valid_y = 0.0, 0.0
    degraded_start_time_iso = None

    for i, (xm, ym, spd, heading) in enumerate(coords_m):
        pt_time = datetime.fromtimestamp(
            start_time.timestamp() + (i * sim_dt), tz=timezone.utc
        )
        pt_time_iso = pt_time.isoformat()

        is_degraded = False
        if degraded_range and (degraded_range[0] <= i < degraded_range[1]):
            is_degraded = True
            if degraded_start_time_iso is None:
                degraded_start_time_iso = pt_time_iso
        else:
            # When signal is valid, update last valid position
            last_valid_x, last_valid_y = xm, ym
            degraded_start_time_iso = None

        lat, lng = lat_lng_offset(args.anchor_lat, args.anchor_lng, xm, ym)
        battery_current = max(0, int(args.battery_start_pct - (i * args.battery_drain_rate)))

        net_displacement = math.hypot(xm - last_valid_x, ym - last_valid_y) if is_degraded else 0.0
        step_count = (
            int((i - degraded_range[0] + 1) * sim_dt * 1.6)
            if (is_degraded and degraded_range)
            else 0
        )

        payload = build_telemetry_payload(
            user_id=args.user_id,
            timestamp_iso=pt_time_iso,
            lat=lat,
            lng=lng,
            speed_mps=spd,
            heading_deg=heading,
            battery_pct=battery_current,
            is_degraded=is_degraded,
            degraded_since_iso=degraded_start_time_iso,
            net_displacement_m=net_displacement,
            step_count=step_count,
        )
        payloads.append(payload)

    # 4. Stream payloads to backend
    records = stream_telemetry(args.backend_url, payloads, args.interval_seconds)

    # 5. Export CSV
    if not args.no_csv:
        csv_filename = args.csv_out
        if not csv_filename:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"trajectory_{args.user_id}_{args.pattern}_{timestamp_str}.csv"
        export_csv(records, csv_filename)


if __name__ == "__main__":
    main()

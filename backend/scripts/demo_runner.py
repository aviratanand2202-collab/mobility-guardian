#!/usr/bin/env python3
"""
Mobility Guardian — Master Six-Scenario Live Demonstration Runner.

Generates continuous, physically realistic 61-70 point GPS & biometric trajectories
for all six demonstration scenarios according to clinical and algorithmic specifications:

Part I: Safe Everyday Living & Perimeter Boundary Defense (patient_A)
  1. Normal Walking: Smooth indoor/garden movement inside Home A (QUIESCENT risk, baseline biometrics).
  2. Circular Garden Walk: Elliptical path in courtyard (demonstrates safe-area loop suppression).
  3. Exercise Negative Control: Fast movement (1.45 m/s) with tachycardia (125 BPM) inside Home A (EXERTION_CONCORDANT).
  4. Confirmed Home Exit: Continuous path crossing Home A perimeter (2-fix debouncing -> Tier 3 CRITICAL alert).

Part II: Trajectory Machine Learning & Physiological Risk Fusion (patient_B)
  5. Pacing Window 1: 120s observation window of repetitive pacing (ML detects PACING; persistence gate holds).
  6. Pacing Window 2 + HR Corroboration: Monotonic continuation with elevated HR 108 BPM (dual escalation to CRITICAL 85.0).

Usage:
  python backend/scripts/demo_runner.py --list
  python backend/scripts/demo_runner.py --check-preconditions
  python backend/scripts/demo_runner.py --scenario 1 [--dry-run] [--playback-interval 1.0]
  python backend/scripts/demo_runner.py --scenario all [--dry-run]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

# Geodesic constants matching backend / ML feature extractor
METERS_PER_DEGREE_LAT = 111320.0
HOME_A_CENTER_LAT = 39.900000
HOME_A_CENTER_LNG = 116.400000
HOME_A_RADIUS_M = 200.0
GEOFENCE_HYSTERESIS_M = 15.0

PATIENT_B_PACING_LAT = 39.905000
PATIENT_B_PACING_LNG = 116.405000


def lat_lng_offset(
    anchor_lat: float, anchor_lng: float, offset_x_m: float, offset_y_m: float
) -> Tuple[float, float]:
    """Convert tangent-plane offsets (meters East, meters North) to (lat, lng)."""
    lat = anchor_lat + (offset_y_m / METERS_PER_DEGREE_LAT)
    meters_per_degree_lng = METERS_PER_DEGREE_LAT * math.cos(math.radians(anchor_lat))
    if abs(meters_per_degree_lng) < 1e-6:
        meters_per_degree_lng = 1e-6
    lng = anchor_lng + (offset_x_m / meters_per_degree_lng)
    return round(lat, 7), round(lng, 7)


def distance_between_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Equirectangular geodesic distance in meters."""
    dy = (lat2 - lat1) * METERS_PER_DEGREE_LAT
    dx = (lng2 - lng1) * (METERS_PER_DEGREE_LAT * math.cos(math.radians((lat1 + lat2) / 2.0)))
    return math.hypot(dx, dy)


def calculate_heading(dx: float, dy: float, prev_heading: float = 0.0) -> float:
    """Calculate geographic compass heading (0=N, 90=E, 180=S, 270=W)."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return prev_heading
    return round(math.degrees(math.atan2(dx, dy)) % 360.0, 1)


@dataclass
class TrajectoryPoint:
    lat: float
    lng: float
    speed_mps: float
    heading_deg: float
    heart_rate_bpm: Optional[int]
    wear_state: str
    sensor_status: str
    horizontal_accuracy_m: float = 4.2
    battery_pct: int = 90


@dataclass
class ScenarioSpec:
    id: int
    name: str
    user_id: str
    description: str
    num_points: int
    step_seconds: float
    generator: Callable[[], List[TrajectoryPoint]]
    expected_outcome: str


# ==============================================================================
# TRAJECTORY GENERATORS (61-70 POINTS EACH)
# ==============================================================================

def generate_scenario_1_normal() -> List[TrajectoryPoint]:
    """Scenario 1: Normal walking inside Home A (65 points, duration 121.6s)."""
    num_points = 65
    step_sec = 1.9
    speed = 0.95
    step_dist = speed * step_sec
    points: List[TrajectoryPoint] = []

    x, y = 10.0, 15.0
    heading = 45.0
    for i in range(num_points):
        # Smoothly curve through rooms/courtyard, bounded within ~45m of Home A center
        t = i / (num_points - 1)
        target_angle = (360.0 * 1.5 * t + 25.0 * math.sin(t * 7.0)) % 360.0
        diff = (target_angle - heading + 180.0) % 360.0 - 180.0
        heading = (heading + diff * 0.25) % 360.0

        dx = step_dist * math.sin(math.radians(heading))
        dy = step_dist * math.cos(math.radians(heading))
        x += dx
        y += dy

        lat, lng = lat_lng_offset(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, x, y)
        hr = 71 + (i % 4)
        points.append(TrajectoryPoint(
            lat=lat, lng=lng, speed_mps=round(speed, 2), heading_deg=round(heading, 1),
            heart_rate_bpm=hr, wear_state="ON_WRIST", sensor_status="SIMULATED_NORMAL",
            horizontal_accuracy_m=4.2, battery_pct=max(10, 95 - int(i * 0.05)),
        ))
    return points


def generate_scenario_2_garden_walk() -> List[TrajectoryPoint]:
    """Scenario 2: Circular garden walk inside Home A (65 points, duration 121.6s)."""
    num_points = 65
    speed = 0.85
    radius_x = 35.0
    radius_y = 28.0
    center_x = 35.0
    center_y = 40.0
    points: List[TrajectoryPoint] = []

    for i in range(num_points):
        theta = 2.0 * math.pi * 1.25 * (i / (num_points - 1))
        x = center_x + radius_x * math.cos(theta)
        y = center_y + radius_y * math.sin(theta)

        # Tangent vector
        dx = -radius_x * math.sin(theta)
        dy = radius_y * math.cos(theta)
        heading = calculate_heading(dx, dy)

        lat, lng = lat_lng_offset(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, x, y)
        hr = 73 + int((i % 5) * 0.7)
        points.append(TrajectoryPoint(
            lat=lat, lng=lng, speed_mps=round(speed, 2), heading_deg=heading,
            heart_rate_bpm=hr, wear_state="ON_WRIST", sensor_status="SIMULATED_NORMAL",
            horizontal_accuracy_m=3.8, battery_pct=max(10, 93 - int(i * 0.05)),
        ))
    return points


def generate_scenario_3_exercise() -> List[TrajectoryPoint]:
    """Scenario 3: Exercise negative control inside Home A (65 points, duration 121.6s, speed 1.45 m/s)."""
    num_points = 65
    speed = 1.45  # Exertion threshold >= 1.2 m/s
    radius_x = 55.0
    radius_y = 35.0
    center_x = -15.0
    center_y = -10.0
    points: List[TrajectoryPoint] = []

    for i in range(num_points):
        theta = 2.0 * math.pi * 1.6 * (i / (num_points - 1))
        x = center_x + radius_x * math.cos(theta)
        y = center_y + radius_y * math.sin(theta)

        dx = -radius_x * math.sin(theta)
        dy = radius_y * math.cos(theta)
        heading = calculate_heading(dx, dy)

        lat, lng = lat_lng_offset(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, x, y)
        hr = 124 + (i % 5)  # Elevated exercise heart rate
        points.append(TrajectoryPoint(
            lat=lat, lng=lng, speed_mps=round(speed, 2), heading_deg=heading,
            heart_rate_bpm=hr, wear_state="ON_WRIST", sensor_status="SIMULATED_EXERTION",
            horizontal_accuracy_m=4.0, battery_pct=max(10, 92 - int(i * 0.05)),
        ))
    return points


def generate_scenario_4_home_exit() -> List[TrajectoryPoint]:
    """Scenario 4: Confirmed home exit crossing boundary (65 points, duration 121.6s)."""
    num_points = 65
    speed = 1.25
    start_r = 75.0
    bearing_deg = 48.0
    points: List[TrajectoryPoint] = []

    for i in range(num_points):
        # Progressively move outward from 75m to ~302m
        curv = 4.0 * math.sin(i * 0.08)
        b = bearing_deg + curv
        r = start_r + (i * 3.55)
        x = r * math.sin(math.radians(b))
        y = r * math.cos(math.radians(b))

        dx = math.sin(math.radians(b))
        dy = math.cos(math.radians(b))
        heading = calculate_heading(dx, dy)

        lat, lng = lat_lng_offset(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, x, y)
        hr = 76 + (i % 4)
        points.append(TrajectoryPoint(
            lat=lat, lng=lng, speed_mps=round(speed, 2), heading_deg=heading,
            heart_rate_bpm=hr, wear_state="ON_WRIST", sensor_status="SIMULATED_NORMAL",
            horizontal_accuracy_m=4.5, battery_pct=max(10, 90 - int(i * 0.05)),
        ))
    return points


def generate_pacing_corridor(
    num_points: int, step_sec: float, hr_val: int, sensor_status: str, initial_direction: float = 1.0
) -> List[TrajectoryPoint]:
    """Generate Algase pacing phenotype along a 30m corridor."""
    speed = 0.40  # Sedentary/pacing movement speed
    seg_length = 30.0
    axis_deg = 45.0
    step_dist = speed * step_sec
    pos = -seg_length / 2.0 if initial_direction > 0 else seg_length / 2.0
    direction = initial_direction
    heading = axis_deg if direction > 0 else (axis_deg + 180.0) % 360.0
    points: List[TrajectoryPoint] = []

    for i in range(num_points):
        pos += direction * step_dist
        if pos >= seg_length / 2.0:
            pos = seg_length / 2.0
            direction = -1.0
            heading = (axis_deg + 180.0) % 360.0
        elif pos <= -seg_length / 2.0:
            pos = -seg_length / 2.0
            direction = 1.0
            heading = axis_deg

        x = pos * math.sin(math.radians(axis_deg))
        y = pos * math.cos(math.radians(axis_deg))

        lat, lng = lat_lng_offset(PATIENT_B_PACING_LAT, PATIENT_B_PACING_LNG, x, y)
        points.append(TrajectoryPoint(
            lat=lat, lng=lng, speed_mps=round(speed, 2), heading_deg=heading,
            heart_rate_bpm=hr_val, wear_state="ON_WRIST", sensor_status=sensor_status,
            horizontal_accuracy_m=4.0, battery_pct=max(10, 95 - int(i * 0.04)),
        ))
    return points


def generate_scenario_5_pacing_win1() -> List[TrajectoryPoint]:
    """Scenario 5: Pacing Window 1 for patient_B (61 points, duration 120.0s, HR 74 bpm)."""
    return generate_pacing_corridor(
        num_points=61, step_sec=2.0, hr_val=74, sensor_status="SIMULATED_NORMAL", initial_direction=1.0
    )


def generate_scenario_6_pacing_win2() -> List[TrajectoryPoint]:
    """Scenario 6: Pacing Window 2 + HR Corroboration (61 points, duration 120.0s, HR 108 bpm)."""
    return generate_pacing_corridor(
        num_points=61, step_sec=2.0, hr_val=108, sensor_status="SIMULATED_AROUSAL", initial_direction=-1.0
    )


SCENARIOS: Dict[int, ScenarioSpec] = {
    1: ScenarioSpec(
        id=1,
        name="Normal Walking Inside Home A",
        user_id="patient_A",
        description="Intentional everyday walking contained inside Home A. Low operational risk.",
        num_points=65,
        step_seconds=1.9,
        generator=generate_scenario_1_normal,
        expected_outcome="Operational tier remains QUIESCENT (score ~15.0). Zero alerts.",
    ),
    2: ScenarioSpec(
        id=2,
        name="Circular Garden Walk",
        user_id="patient_A",
        description="Repetitive elliptical path in courtyard. Demonstrates safe-area loop suppression.",
        num_points=65,
        step_seconds=1.9,
        generator=generate_scenario_2_garden_walk,
        expected_outcome="Loop alert is suppressed inside safe zone. Operational tier remains QUIESCENT.",
    ),
    3: ScenarioSpec(
        id=3,
        name="Exercise Negative Control",
        user_id="patient_A",
        description="Physical exertion (speed 1.45 m/s, HR 125 bpm). Demonstrates false alarm avoidance.",
        num_points=65,
        step_seconds=1.9,
        generator=generate_scenario_3_exercise,
        expected_outcome="Elevated HR categorized as EXERTION_CONCORDANT. Operational tier remains QUIESCENT.",
    ),
    4: ScenarioSpec(
        id=4,
        name="Confirmed Home Exit",
        user_id="patient_A",
        description="Starts inside, crosses boundary (>215m). Demonstrates 2-reading exit debounce & deduplication.",
        num_points=65,
        step_seconds=1.9,
        generator=generate_scenario_4_home_exit,
        expected_outcome="Fix 41: EXIT_PENDING. Fix 42: BREACHED -> Tier 3 CRITICAL alert. Subsequent fixes update location without duplicate alerts.",
    ),
    5: ScenarioSpec(
        id=5,
        name="Pacing Window 1 (Pure Kinematics)",
        user_id="patient_B",
        description="120s window of repetitive corridor pacing with normal HR. Demonstrates persistence gate holding.",
        num_points=61,
        step_seconds=2.0,
        generator=generate_scenario_5_pacing_win1,
        expected_outcome="Fix 61 evaluates window: ML classifies PACING (conf >= 0.80). Persistence gate holds at consecutive_windows=1. Operational risk un-escalated.",
    ),
    6: ScenarioSpec(
        id=6,
        name="Pacing Window 2 + Physiological Corroboration",
        user_id="patient_B",
        description="Second 120s pacing window with elevated resting HR (108 bpm, z=+11.33). Sustained multi-modal escalation.",
        num_points=61,
        step_seconds=2.0,
        generator=generate_scenario_6_pacing_win2,
        expected_outcome="Window 2 evaluates: consecutive_windows=2. Corroborated by physiological arousal -> escalates to CRITICAL 85.0 with explainability cards.",
    ),
}


# ==============================================================================
# ENVIRONMENT CHECKLIST & PREFLIGHT
# ==============================================================================

def check_preconditions(backend_url: str = "http://127.0.0.1:8000") -> bool:
    """Run read-only inspection against live backend endpoints."""
    print("=" * 80)
    print(">>> MOBILITY GUARDIAN PRE-DEMONSTRATION STATE CHECKLIST (READ-ONLY)")
    print("=" * 80)

    with httpx.Client(timeout=5.0) as client:
        # 1. Target Database Check
        try:
            r = client.get(f"{backend_url}/api/consent/test_ingest_e6557ad7")
            if r.status_code == 200 and r.json().get("user_id") == "test_ingest_e6557ad7":
                print("  [OK] Backend is actively connected to backend/app/data/test_app.db")
            else:
                print("  [FAIL] Backend is not serving test_app.db! Aborting.")
                return False
        except Exception as e:
            print(f"  [FAIL] Cannot connect to backend at {backend_url}: {e}")
            return False

        # 2. Patient A (Perimeter Defense)
        try:
            r_consent_a = client.get(f"{backend_url}/api/consent/patient_A")
            r_zones_a = client.get(f"{backend_url}/api/safe-zones/patient_A")
            cA = r_consent_a.json() if r_consent_a.status_code == 200 else {}
            szA = r_zones_a.json() if r_zones_a.status_code == 200 and isinstance(r_zones_a.json(), list) else []

            if cA.get("guardian_consent", {}).get("granted") and len(szA) >= 1 and szA[0].get("name") == "Home A":
                print("  [OK] patient_A: Active consent and safe zone 'Home A' (radius 200m) confirmed")
            else:
                print(f"  [FAIL] patient_A configuration invalid (Consent: {cA.get('status')}, Zones: {len(szA)})")
                return False
        except Exception as e:
            print(f"  [FAIL] Failed inspecting patient_A: {e}")
            return False

        # 3. Patient B (Clean Baseline & Zero Geofence Interference)
        try:
            r_consent_b = client.get(f"{backend_url}/api/consent/patient_B")
            r_zones_b = client.get(f"{backend_url}/api/safe-zones/patient_B")
            r_rh_b = client.get(f"{backend_url}/api/risk-history/patient_B?limit=1")
            r_wear_b = client.get(f"{backend_url}/api/wearable/patient_B/history?limit=1")
            r_al_b = client.get(f"{backend_url}/api/alerts/user/patient_B")

            cB = r_consent_b.json() if r_consent_b.status_code == 200 else {}
            szB = r_zones_b.json() if r_zones_b.status_code == 200 and isinstance(r_zones_b.json(), list) else []
            rhB = r_rh_b.json() if r_rh_b.status_code == 200 and isinstance(r_rh_b.json(), list) else []
            wearB = r_wear_b.json() if r_wear_b.status_code == 200 and isinstance(r_wear_b.json(), list) else []
            alB = r_al_b.json() if r_al_b.status_code == 200 and isinstance(r_al_b.json(), list) else []

            if not cB.get("guardian_consent", {}).get("granted"):
                print("  [FAIL] patient_B consent not granted!")
                return False
            if len(szB) != 0:
                print(f"  [FAIL] patient_B has {len(szB)} safe zones. Expected 0 for decoupled behavioural testing.")
                return False
            if len(rhB) != 0 or len(wearB) != 0 or len(alB) != 0:
                print(
                    f"  [FAIL] patient_B is NOT pristine ({len(rhB)} history rows, {len(wearB)} wearable rows, "
                    f"{len(alB)} alerts). The demonstration requires a pristine baseline. Do not delete via SQL."
                )
                return False
            print("  [OK] patient_B: Active consent, 0 safe zones, 0 history, 0 wearable, 0 alerts confirmed pristine")
        except Exception as e:
            print(f"  [FAIL] Failed inspecting patient_B: {e}")
            return False

    print("=" * 80)
    print(">>> Environment statically verified and ready for live demonstration.")
    print("=" * 80)
    return True


def fetch_latest_timestamp(backend_url: str, user_id: str) -> Optional[datetime]:
    """Retrieve the single most recent accepted telemetry timestamp for a user."""
    url = f"{backend_url.rstrip('/')}/api/risk-history/{user_id}?limit=1"
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"GET /api/risk-history/{user_id} failed with HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            ts_str = data[0].get("timestamp")
            if ts_str:
                cleaned = ts_str.strip().replace(" ", "+")
                if cleaned.endswith("Z"):
                    cleaned = cleaned[:-1] + "+00:00"
                dt = datetime.fromisoformat(cleaned)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def calculate_scenario_start_time(
    backend_url: str,
    spec: ScenarioSpec,
    now_utc: datetime,
) -> datetime:
    """Compute safe starting timestamp enforcing monotonicity and non-future bounds."""
    latest_ts = fetch_latest_timestamp(backend_url, spec.user_id)
    sim_duration = (spec.num_points - 1) * spec.step_seconds

    # Special preflight for Scenario 5: Must pre-validate full two-window span (242.0s)
    if spec.id == 5:
        total_span = 242.0  # 120s (Win 1) + 2s (Gap) + 120s (Win 2)
        start_time = latest_ts + timedelta(seconds=2.0) if latest_ts else now_utc - timedelta(seconds=total_span + 10.0)
        final_fix_time = start_time + timedelta(seconds=total_span)
        if final_fix_time > now_utc:
            wait_sec = math.ceil((final_fix_time - now_utc).total_seconds())
            raise ValueError(
                f"Two-Window Preflight Guard Failed: Scenario 5 & 6 (242s total duration) would complete {wait_sec}s "
                f"in the future at {final_fix_time.isoformat()}. Please wait {wait_sec} seconds before starting Scenario 5."
            )
        return start_time

    # Standard scenario calculation
    start_time = latest_ts + timedelta(seconds=2.0) if latest_ts else now_utc - timedelta(seconds=sim_duration + 10.0)
    end_time = start_time + timedelta(seconds=sim_duration)

    if end_time > now_utc:
        wait_sec = math.ceil((end_time - now_utc).total_seconds())
        raise ValueError(
            f"Timestamp Guard Failed: Scenario {spec.id} would finish {wait_sec}s into the future at "
            f"{end_time.isoformat()}. Please wait {wait_sec} seconds before running."
        )

    return start_time


# ==============================================================================
# TELEMETRY PAYLOAD BUILDER & STREAMER
# ==============================================================================

def build_telemetry_payload(
    user_id: str,
    timestamp_iso: str,
    pt: TrajectoryPoint,
) -> Dict[str, Any]:
    """Construct schema-compliant TelemetryPayload dictionary."""
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "timestamp": timestamp_iso,
        "location": {
            "lat": pt.lat,
            "lng": pt.lng,
            "altitude_m": 45.0,
        },
        "sensor_metrics": {
            "horizontal_accuracy_m": pt.horizontal_accuracy_m,
            "speed_mps": pt.speed_mps,
            "heading_deg": pt.heading_deg,
            "activity_type": "WALKING" if pt.speed_mps >= 0.2 else "STATIONARY",
            "battery_pct": pt.battery_pct,
        },
        "signal_status": {
            "state": "VALID",
            "degraded_since": None,
        },
        "imu_metrics": None,
    }

    if pt.heart_rate_bpm is not None:
        payload["wearable_metrics"] = {
            "heart_rate_bpm": pt.heart_rate_bpm,
            "timestamp": timestamp_iso,
            "wear_state": pt.wear_state,
            "sensor_status": pt.sensor_status,
        }

    return payload


def run_scenario(
    spec: ScenarioSpec,
    backend_url: str = "http://127.0.0.1:8000",
    playback_interval_sec: float = 1.0,
    dry_run: bool = False,
) -> bool:
    """Execute a single scenario with live console telemetry and risk tracking."""
    now_utc = datetime.now(timezone.utc)
    points = spec.generator()

    print("\n" + "=" * 90)
    print(f"SCENARIO {spec.id}: {spec.name.upper()} ({spec.user_id})")
    print(f"Points:        {len(points)} (Step interval: {spec.step_seconds}s)")
    print(f"Sim Duration:  {(len(points) - 1) * spec.step_seconds:.1f}s")
    print(f"Playback Pace: {playback_interval_sec}s/point (~{(len(points) - 1) * playback_interval_sec:.0f}s live)")
    print(f"Goal:          {spec.description}")
    print("=" * 90)

    # 1. Preflight timestamp evaluation
    try:
        start_time = calculate_scenario_start_time(backend_url, spec, now_utc)
    except Exception as e:
        print(f"\n[PREFLIGHT ABORT] {e}")
        return False

    print(f"Start Timestamp (UTC): {start_time.isoformat()}")
    print(f"Final Fix Time (UTC):  {(start_time + timedelta(seconds=(len(points)-1)*spec.step_seconds)).isoformat()}")
    print(f"Current Wall Clock:    {now_utc.isoformat()}")

    if dry_run:
        print("\n[DRY RUN MODE] Generating and validating payloads without sending HTTP requests...")
        for idx, pt in enumerate(points):
            t_iso = (start_time + timedelta(seconds=idx * spec.step_seconds)).isoformat()
            payload = build_telemetry_payload(spec.user_id, t_iso, pt)
            dist_home = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)
            if idx == 0 or idx == len(points) - 1 or (idx + 1) % 15 == 0:
                print(
                    f"  [DryRun Pt {idx+1:02d}/{len(points):02d}] {t_iso} | Lat: {pt.lat:.6f}, Lng: {pt.lng:.6f} | "
                    f"Dist HomeA: {dist_home:5.1f}m | Spd: {pt.speed_mps}m/s | HR: {pt.heart_rate_bpm}bpm"
                )
        print(f"[DRY RUN SUCCESS] Validated all {len(points)} points. Timestamp bounds strictly satisfied.")
        return True

    # 2. Live streaming loop
    url = f"{backend_url.rstrip('/')}/api/telemetry/ingest"
    print(f"\nStreaming live telemetry to {url}...")

    with httpx.Client(timeout=10.0) as client:
        for idx, pt in enumerate(points):
            t_iso = (start_time + timedelta(seconds=idx * spec.step_seconds)).isoformat()
            payload = build_telemetry_payload(spec.user_id, t_iso, pt)

            try:
                resp = client.post(url, json=payload)
            except Exception as e:
                print(f"\n[ERROR] Ingestion failed at point {idx+1}: {e}")
                return False

            if resp.status_code != 200:
                print(f"\n[ERROR] HTTP {resp.status_code} at point {idx+1}: {resp.text}")
                return False

            data = resp.json()
            score = data.get("risk_score", 0.0)
            tier = data.get("risk_tier", "UNKNOWN")
            polling = (data.get("polling_instruction") or {}).get("mode", "UNKNOWN")

            ml_eval = data.get("ml_evaluation") or {}
            indicator = ml_eval.get("behavioral_indicator", "NORMAL")
            conf = ml_eval.get("behavioral_confidence")
            hr_ev = ml_eval.get("heart_rate_evidence") or {}
            dist_home = distance_between_m(HOME_A_CENTER_LAT, HOME_A_CENTER_LNG, pt.lat, pt.lng)

            # Compact single-line status
            hr_str = f"{pt.heart_rate_bpm}bpm" if pt.heart_rate_bpm else "---"
            conf_str = f"{conf:.2f}" if conf is not None else "0.00"
            print(
                f"[Pt {idx+1:02d}/{len(points):02d}] "
                f"Dist: {dist_home:5.1f}m | "
                f"Spd: {pt.speed_mps:3.1f}m/s | "
                f"HR: {hr_str:<6} | "
                f"Score: {score:4.1f} | "
                f"Tier: {tier:<9} | "
                f"ML: {indicator} ({conf_str}) | "
                f"Poll: {polling}"
            )

            # Highlight significant lifecycle milestones
            if tier == "CRITICAL" and (idx == 0 or points[idx-1].speed_mps != pt.speed_mps or idx >= 40):
                top_f = (data.get("explainability") or {}).get("top_features") or []
                if top_f:
                    print(f"    >>> Explainability: {top_f[0].get('feature')} (SHAP: {top_f[0].get('shap_value'):.2f})")

            if playback_interval_sec > 0 and idx < len(points) - 1:
                time.sleep(playback_interval_sec)

    print("\n>>> Scenario execution complete. Expected outcome:")
    print(f"    {spec.expected_outcome}")
    return True


# ==============================================================================
# CLI PARSER
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mobility Guardian — Master Six-Scenario Live Demonstration Runner"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario ID (1-6) or 'all' to run sequence",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all demonstration scenarios with descriptions and expected outcomes",
    )
    parser.add_argument(
        "--check-preconditions",
        action="store_true",
        help="Run read-only inspection against test_app.db and patient state",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="FastAPI backend URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--playback-interval",
        type=float,
        default=1.0,
        help="Sleep delay in seconds between telemetry POSTs (default: 1.0s for ~60s live playback)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick execution mode (sets playback-interval to 0.05s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate trajectory math, timestamps, and schema locally without sending HTTP requests",
    )

    args = parser.parse_args()

    if args.list:
        print("\n" + "=" * 90)
        print("MOBILITY GUARDIAN — MASTER DEMONSTRATION SCENARIO CATALOG")
        print("=" * 90)
        for s_id, s in SCENARIOS.items():
            print(f"\nScenario {s_id}: {s.name.upper()} (User: {s.user_id})")
            print(f"  Description:      {s.description}")
            print(f"  Points / Timing:  {s.num_points} fixes @ {s.step_seconds}s interval ({(s.num_points-1)*s.step_seconds:.1f}s total duration)")
            print(f"  Expected Outcome: {s.expected_outcome}")
        print("\n" + "=" * 90)
        return

    if args.check_preconditions:
        ok = check_preconditions(args.backend_url)
        sys.exit(0 if ok else 1)

    if not args.scenario:
        parser.print_help()
        print("\nTip: Run with --list to see scenarios or --check-preconditions to inspect the environment.")
        return

    interval = 0.05 if args.fast else args.playback_interval

    if args.scenario.lower() == "all":
        print("\n>>> Preparing complete 6-scenario demonstration run...")
        if not args.dry_run and not check_preconditions(args.backend_url):
            print("\n[ABORT] Precondition checks failed. Aborting full sequence.")
            sys.exit(1)

        for s_id in range(1, 7):
            success = run_scenario(SCENARIOS[s_id], args.backend_url, interval, args.dry_run)
            if not success:
                print(f"\n[ABORT] Scenario {s_id} encountered an error. Stopping sequence.")
                sys.exit(1)
            if s_id < 6 and not args.dry_run:
                print("\nWaiting 2 seconds before next scenario...")
                time.sleep(2.0)
    else:
        try:
            s_num = int(args.scenario)
            if s_num not in SCENARIOS:
                raise ValueError
        except ValueError:
            print(f"[Error] Invalid scenario '{args.scenario}'. Choose 1-6 or 'all'.")
            sys.exit(1)

        success = run_scenario(SCENARIOS[s_num], args.backend_url, interval, args.dry_run)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

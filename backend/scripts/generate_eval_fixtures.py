#!/usr/bin/env python3
"""
Deterministic Evaluation Fixture Generator for Mobility Guardian.

Generates immutable, canonically-hashed JSON trajectory fixtures for the
5-scenario Phase 3B evaluation pilot:
1. Normal Baseline Transit (P1)
2. Stationary Baseline Dwelling (P2)
3. Linear Pacing Excursions (P3)
4. Circular Lapping Loops (P4)
5. Outward Random Drift (P5)

Key Invariants:
- 100% Offline: Zero network calls, zero database connections, zero model mutations.
- Byte-for-Byte Determinism: Explicit PRNG seeding (random.Random(seed)) and fixed UTC clock.
- Complete Provenance: Encapsulates generator equations and parameter envelopes.
- Canonical Hashing: RFC 8785-compatible JSON serialization with SHA-256 verification.
- Decoupled Semantics: Phenotype intent is recorded strictly as metadata, never as ground-truth label.
- Conformance: Every payload strictly validates against TelemetryPayload in backend/app/models.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import TelemetryPayload schema from backend
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.models import (
    ActivityType,
    Location,
    SensorMetrics,
    SignalState,
    SignalStatus,
    TelemetryPayload,
)

# Geographic conversion constant (Haidian District reference: ~39.98N, 116.31E)
METERS_PER_DEGREE_LAT = 111320.0


def lat_lng_offset(
    anchor_lat: float, anchor_lng: float, offset_x_m: float, offset_y_m: float
) -> Tuple[float, float]:
    """Convert local Cartesian meter offsets (East, North) to (lat, lng) using equirectangular projection."""
    lat = anchor_lat + (offset_y_m / METERS_PER_DEGREE_LAT)
    meters_per_degree_lng = METERS_PER_DEGREE_LAT * math.cos(math.radians(anchor_lat))
    if abs(meters_per_degree_lng) < 1e-6:
        meters_per_degree_lng = 1e-6
    lng = anchor_lng + (offset_x_m / meters_per_degree_lng)
    return round(lat, 7), round(lng, 7)


def calculate_heading(dx: float, dy: float, fallback: float = 0.0) -> float:
    """Calculate geographic heading in degrees [0, 360) from displacement vector (dx East, dy North)."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback
    heading = math.degrees(math.atan2(dx, dy)) % 360.0
    return round(heading, 1)


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic, RFC 8785-compatible canonical JSON byte serialization."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def compute_sha256(canonical_bytes: bytes) -> str:
    """Compute SHA-256 digest of canonical byte representation."""
    return hashlib.sha256(canonical_bytes).hexdigest()


# ==============================================================================
# DETERMINISTIC PHENOTYPE GENERATORS
# ==============================================================================


def generate_normal_coords(
    num_points: int,
    step_sec: float,
    speed_mps: float,
    seed: int,
) -> List[Tuple[float, float, float, float]]:
    """Normal random walk bounded within ~140m of anchor with smooth turns toward origin."""
    rng = random.Random(seed)
    points: List[Tuple[float, float, float, float]] = []
    x, y = 0.0, 0.0
    heading = rng.uniform(0.0, 360.0)
    step_dist = speed_mps * step_sec

    for i in range(num_points):
        r = math.hypot(x, y)
        if r > 140.0 or (i > 0 and i % 15 == 0):
            target_heading = math.degrees(math.atan2(-x, -y)) % 360.0
            diff = (target_heading - heading + 180.0) % 360.0 - 180.0
            heading = (heading + diff * 0.4) % 360.0
        else:
            heading = (heading + rng.gauss(0.0, 10.0)) % 360.0

        dx = step_dist * math.sin(math.radians(heading))
        dy = step_dist * math.cos(math.radians(heading))
        x += dx
        y += dy
        actual_speed = round(max(0.2, speed_mps + rng.uniform(-0.05, 0.05)), 2)
        points.append((round(x, 2), round(y, 2), actual_speed, round(heading, 1)))

    return points


def generate_corridor_normal_coords(
    num_points: int,
    step_sec: float,
    speed_mps: float,
    seed: int,
    corridor_bearing_deg: float = 45.0,
    heading_noise_std_deg: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """Simulate unperturbed open-corridor pedestrian transit along a steady bearing.

    Models realistic human walking along a continuous street or walkway corridor:
    - Primary bearing maintained along corridor_bearing_deg.
    - Small Gaussian heading jitter (sigma = 2.0 deg).
    - Natural velocity fluctuations (speed_mps +/- 0.05 m/s).
    - Zero origin-seeking reversal or looping.
    """
    rng = random.Random(seed)
    points: List[Tuple[float, float, float, float]] = []
    x, y = 0.0, 0.0

    for _ in range(num_points):
        heading = (corridor_bearing_deg + rng.gauss(0.0, heading_noise_std_deg)) % 360.0
        actual_speed = round(max(0.2, speed_mps + rng.uniform(-0.05, 0.05)), 2)
        step_dist = actual_speed * step_sec
        dx = step_dist * math.sin(math.radians(heading))
        dy = step_dist * math.cos(math.radians(heading))
        x += dx
        y += dy
        points.append((round(x, 2), round(y, 2), actual_speed, round(heading, 1)))

    return points


def generate_stationary_coords(
    num_points: int,
) -> List[Tuple[float, float, float, float]]:
    """Constant coordinates (zero displacement, zero velocity, zero heading)."""
    return [(0.0, 0.0, 0.0, 0.0) for _ in range(num_points)]


def generate_pacing_coords(
    num_points: int,
    step_sec: float,
    speed_mps: float,
    segment_length_m: float = 30.0,
    axis_angle_deg: float = 45.0,
) -> List[Tuple[float, float, float, float]]:
    """Algase Pacing phenotype: Repetitive 180-degree reversals along a linear corridor."""
    points: List[Tuple[float, float, float, float]] = []
    direction = 1.0
    pos_along = -segment_length_m / 2.0
    step_dist = speed_mps * step_sec
    heading = axis_angle_deg

    for _ in range(num_points):
        pos_along += direction * step_dist
        if pos_along >= segment_length_m / 2.0:
            pos_along = segment_length_m / 2.0
            direction = -1.0
            heading = (axis_angle_deg + 180.0) % 360.0
        elif pos_along <= -segment_length_m / 2.0:
            pos_along = -segment_length_m / 2.0
            direction = 1.0
            heading = axis_angle_deg

        x = pos_along * math.sin(math.radians(axis_angle_deg))
        y = pos_along * math.cos(math.radians(axis_angle_deg))
        points.append((round(x, 2), round(y, 2), round(speed_mps, 2), round(heading, 1)))

    return points


def generate_lapping_coords(
    num_points: int,
    step_sec: float,
    speed_mps: float,
    radius_m: float = 45.0,
) -> List[Tuple[float, float, float, float]]:
    """Algase Lapping phenotype: Closed-loop circular trajectory (R < 100m)."""
    points: List[Tuple[float, float, float, float]] = []
    omega = speed_mps / radius_m
    theta = 0.0

    for _ in range(num_points):
        x = radius_m * math.sin(theta)
        y = radius_m * math.cos(theta)
        tangent_dx = radius_m * omega * math.cos(theta)
        tangent_dy = -radius_m * omega * math.sin(theta)
        heading = calculate_heading(tangent_dx, tangent_dy, 0.0)
        points.append((round(x, 2), round(y, 2), round(speed_mps, 2), heading))
        theta += omega * step_sec

    return points


def generate_random_drift_coords(
    num_points: int,
    step_sec: float,
    speed_mps: float,
    seed: int,
    exit_bearing_deg: float = 55.0,
) -> List[Tuple[float, float, float, float]]:
    """Algase Random Drift (Direct exit): Progressive outward excursion along an escape bearing."""
    rng = random.Random(seed)
    points: List[Tuple[float, float, float, float]] = []
    x, y = 0.0, 0.0
    heading = exit_bearing_deg
    step_dist = speed_mps * step_sec

    for _ in range(num_points):
        d_heading = rng.gauss(0.0, 45.0)
        heading = (heading + d_heading) % 360.0
        dx = step_dist * (math.sin(math.radians(heading)) + 0.85 * math.sin(math.radians(exit_bearing_deg)))
        dy = step_dist * (math.cos(math.radians(heading)) + 0.85 * math.cos(math.radians(exit_bearing_deg)))
        x += dx
        y += dy
        actual_heading = calculate_heading(dx, dy, heading)
        actual_speed = round(math.hypot(dx, dy) / step_sec, 2)
        points.append((round(x, 2), round(y, 2), actual_speed, actual_heading))

    return points


# ==============================================================================
# FIXTURE BUILDER & SERIALIZER
# ==============================================================================


def build_scenario_payloads(
    user_id: str,
    base_time: datetime,
    anchor_lat: float,
    anchor_lng: float,
    step_sec: float,
    coords: List[Tuple[float, float, float, float]],
    battery_start_pct: int = 90,
    battery_drain_rate: float = 0.05,
) -> List[Dict[str, Any]]:
    """Build TelemetryPayload dictionaries validated through Pydantic."""
    payloads: List[Dict[str, Any]] = []

    for i, (xm, ym, spd, heading) in enumerate(coords):
        lat, lng = lat_lng_offset(anchor_lat, anchor_lng, xm, ym)
        pt_time = base_time + timedelta(seconds=i * step_sec)
        # Format explicitly as ISO 8601 with Z indicator
        pt_time_str = pt_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        bat = max(5, int(round(battery_start_pct - (i * battery_drain_rate))))
        activity = ActivityType.STATIONARY if spd < 0.2 else ActivityType.WALKING

        payload_obj = TelemetryPayload(
            user_id=user_id,
            timestamp=pt_time,
            location=Location(lat=lat, lng=lng, altitude_m=15.0),
            sensor_metrics=SensorMetrics(
                horizontal_accuracy_m=5.0,
                speed_mps=spd,
                heading_deg=heading,
                activity_type=activity,
                battery_pct=bat,
            ),
            signal_status=SignalStatus(
                state=SignalState.VALID,
                degraded_since=None,
            ),
            imu_metrics=None,
        )

        # Convert to dictionary with deterministic string timestamp
        d = payload_obj.model_dump(mode="json")
        d["timestamp"] = pt_time_str
        payloads.append(d)

    return payloads


def assemble_fixture(
    fixture_id: str,
    phenotype_intent: str,
    user_id: str,
    seed: int,
    base_time_str: str,
    anchor_lat: float,
    anchor_lng: float,
    step_sec: float,
    payloads: List[Dict[str, Any]],
    generator_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble complete fixture document with canonical SHA-256 payload hash."""
    canonical_bytes = canonical_json_bytes(payloads)
    payload_sha256 = compute_sha256(canonical_bytes)

    duration_sec = (len(payloads) - 1) * step_sec if len(payloads) > 1 else 0.0

    return {
        "schema_version": "1.0.0",
        "provenance": {
            "fixture_id": fixture_id,
            "phenotype_intent": phenotype_intent,
            "user_id": user_id,
            "seed": seed,
            "base_timestamp": base_time_str,
            "anchor": {"lat": anchor_lat, "lng": anchor_lng},
            "step_interval_sec": step_sec,
            "point_count": len(payloads),
            "total_duration_sec": duration_sec,
            "target_horizon_sec": 120,
            "generator_parameters": generator_params,
            "payload_canonical_sha256": payload_sha256,
        },
        "payloads": payloads,
    }


def generate_all_pilot_fixtures(
    out_dir: Path,
    seed: int = 42,
    base_time_iso: str = "2026-10-16T12:00:00Z",
    anchor_lat: float = 39.9830,
    anchor_lng: float = 116.3130,
    scenario_id: Optional[str] = None,
) -> Dict[str, Tuple[Path, str]]:
    """Generate pilot fixtures offline and save them to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_time = datetime.fromisoformat(base_time_iso.replace("Z", "+00:00"))

    # Definition of the canonical pilot scenarios
    configs = [
        {
            "id": "pilot_p1_normal",
            "intent": "NORMAL",
            "user": "eval_norm_01",
            "points": 90,
            "step": 2.0,
            "gen_fn": lambda: generate_normal_coords(90, 2.0, 1.2, seed),
            "params": {"speed_mps": 1.2, "generator": "bounded_random_walk"},
            "time_offset_sec": 0,
        },
        {
            "id": "pilot_p1_corridor_normal",
            "intent": "NORMAL",
            "user": "eval_norm_corridor_01",
            "points": 90,
            "step": 2.0,
            "gen_fn": lambda: generate_corridor_normal_coords(
                90, 2.0, 1.2, seed, corridor_bearing_deg=45.0, heading_noise_std_deg=2.0
            ),
            "params": {
                "speed_mps": 1.2,
                "corridor_bearing_deg": 45.0,
                "heading_noise_std_deg": 2.0,
                "generator": "open_corridor_linear_transit",
            },
            "time_offset_sec": 0,
        },
        {
            "id": "pilot_p2_stationary",
            "intent": "STATIONARY",
            "user": "eval_stat_01",
            "points": 25,
            "step": 5.0,
            "gen_fn": lambda: generate_stationary_coords(25),
            "params": {"speed_mps": 0.0, "generator": "constant_position"},
            "time_offset_sec": 600,
        },
        {
            "id": "pilot_p3_pacing",
            "intent": "PACING",
            "user": "eval_pace_01",
            "points": 90,
            "step": 2.0,
            "gen_fn": lambda: generate_pacing_coords(90, 2.0, 1.2, segment_length_m=30.0),
            "params": {"speed_mps": 1.2, "segment_length_m": 30.0, "axis_deg": 45.0},
            "time_offset_sec": 1200,
        },
        {
            "id": "pilot_p4_lapping",
            "intent": "LAPPING",
            "user": "eval_lap_01",
            "points": 90,
            "step": 2.0,
            "gen_fn": lambda: generate_lapping_coords(90, 2.0, 1.2, radius_m=45.0),
            "params": {"speed_mps": 1.2, "radius_m": 45.0},
            "time_offset_sec": 1800,
        },
        {
            "id": "pilot_p5_random_drift",
            "intent": "RANDOM_DRIFT",
            "user": "eval_drift_01",
            "points": 90,
            "step": 2.0,
            "gen_fn": lambda: generate_random_drift_coords(90, 2.0, 1.4, seed, exit_bearing_deg=55.0),
            "params": {"speed_mps": 1.4, "exit_bearing_deg": 55.0},
            "time_offset_sec": 2400,
        },
    ]

    if scenario_id:
        configs = [c for c in configs if c["id"] == scenario_id]
        if not configs:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

    results: Dict[str, Tuple[Path, str]] = {}

    for cfg in configs:
        scenario_start = base_time + timedelta(seconds=cfg["time_offset_sec"])
        coords = cfg["gen_fn"]()
        payloads = build_scenario_payloads(
            user_id=cfg["user"],
            base_time=scenario_start,
            anchor_lat=anchor_lat,
            anchor_lng=anchor_lng,
            step_sec=cfg["step"],
            coords=coords,
        )
        fixture = assemble_fixture(
            fixture_id=cfg["id"],
            phenotype_intent=cfg["intent"],
            user_id=cfg["user"],
            seed=seed,
            base_time_str=scenario_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            anchor_lat=anchor_lat,
            anchor_lng=anchor_lng,
            step_sec=cfg["step"],
            payloads=payloads,
            generator_params=cfg["params"],
        )

        out_file = out_dir / f"{cfg['id']}.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(fixture, fp, indent=2, sort_keys=True)

        sha = fixture["provenance"]["payload_canonical_sha256"]
        results[cfg["id"]] = (out_file, sha)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Offline Evaluation Fixture Generator for Mobility Guardian"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO_ROOT / "backend" / "app" / "data" / "eval_phase3" / "fixtures"),
        help="Target output directory for fixtures",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Specific scenario ID to generate (e.g. pilot_p1_corridor_normal) without regenerating others.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")
    parser.add_argument(
        "--base-time",
        type=str,
        default="2026-10-16T12:00:00Z",
        help="Fixed base UTC timestamp",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir).resolve()
    print("================================================================================")
    print("MOBILITY GUARDIAN — DETERMINISTIC EVALUATION FIXTURE GENERATOR")
    print("================================================================================")
    print(f"Output Directory: {out_path}")
    print(f"Deterministic Seed: {args.seed}")
    print(f"Base Timestamp:     {args.base_time}")
    if args.scenario:
        print(f"Selected Scenario:  {args.scenario}")
    print("-" * 80)

    fixtures = generate_all_pilot_fixtures(
        out_dir=out_path,
        seed=args.seed,
        base_time_iso=args.base_time,
        scenario_id=args.scenario,
    )

    for fid, (fpath, sha) in fixtures.items():
        print(f"[{fid}] -> {fpath.name}")
        print(f"  Canonical Payloads SHA-256: {sha}")

    print("-" * 80)
    print("Verifying generation idempotency (byte-for-byte repeatability check)...")
    re_generated = generate_all_pilot_fixtures(
        out_dir=out_path,
        seed=args.seed,
        base_time_iso=args.base_time,
        scenario_id=args.scenario,
    )

    all_match = True
    for fid, (_, sha1) in fixtures.items():
        _, sha2 = re_generated[fid]
        if sha1 != sha2:
            print(f"[ERROR] Non-deterministic hash mismatch in {fid}: {sha1} != {sha2}")
            all_match = False

    if all_match:
        print("[SUCCESS] All 5 fixtures generated with 100% byte-for-byte idempotency.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

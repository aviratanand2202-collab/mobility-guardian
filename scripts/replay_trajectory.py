#!/usr/bin/env python3
"""GeoLife Trajectory Replay CLI Utility.

Replays real GPS trajectories from GeoLife .plt files through the predictive
geofencing backend's official /api/telemetry/ingest endpoint.

Features:
- Checks consent via GET /api/consent/{user_id} and stops with an informative message
  if consent is missing (per requirements, never fabricates or auto-grants consent).
- Derives physical speed (m/s) and heading (deg) from consecutive GPS fixes.
- Shifts timestamps by default so the final point aligns with current UTC time,
  preserving exact recorded intervals (supports --preserve-timestamps as an option).
- Uses explicit, configurable test-fixture values for assumed GPS accuracy and battery level.
- Never fabricates IMU data (imu_metrics=None for VALID signal).
- Supports dry-run validation, configurable playback delay / speed multiplier, and batch limits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Ensure repo root is on sys.path so ml.src is importable if present
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Physical geodesy constants (WGS-84)
EARTH_RADIUS_M = 6371000.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two coordinates in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def forward_azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward compass azimuth (bearing) from (lat1, lon1) to (lat2, lon2) in [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return round((bearing + 360.0) % 360.0, 1)


def parse_geolife_plt(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a GeoLife .plt file into raw trajectory point records.

    Format:
    Lines 1-6: Header lines.
    Line 7+: Field 0: lat, Field 1: lon, Field 2: 0, Field 3: altitude (ft),
             Field 4: days, Field 5: date (yyyy-mm-dd), Field 6: time (hh:mm:ss)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    points: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_offset = 6
    for line_idx, line in enumerate(lines[header_offset:], start=header_offset + 1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue

        try:
            lat = float(parts[0])
            lon = float(parts[1])
            alt_ft = float(parts[3])
            date_str = parts[5]
            time_str = parts[6]
            dt_raw = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            dt_utc = dt_raw.replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

        # Physical sanity check: latitude in [-90, 90], longitude in [-180, 180]
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        # Altitude conversion: GeoLife uses feet (-777 represents invalid/uninitialized)
        alt_m = round(alt_ft * 0.3048, 2) if alt_ft > -700.0 else None

        points.append({
            "line_number": line_idx,
            "latitude": lat,
            "longitude": lon,
            "altitude_m": alt_m,
            "original_timestamp": dt_utc,
        })

    return points


def check_consent_status(backend_url: str, user_id: str) -> Tuple[bool, Dict[str, Any]]:
    """Query GET /api/consent/{user_id} and return (is_consented, raw_response)."""
    url = f"{backend_url.rstrip('/')}/api/consent/{user_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Validate granted status per backend logic
            if data.get("status") == "not_found":
                return False, data

            guardian = data.get("guardian_consent", {})
            guardian_granted = bool(guardian.get("granted", False))
            end_user_app = bool(data.get("end_user_consent_applicable", False))
            end_user = data.get("end_user_consent")
            end_user_granted = bool(end_user.get("granted", False)) if end_user else False

            if not guardian_granted:
                return False, data
            if end_user_app and not end_user_granted:
                return False, data
            return True, data

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            err_json = json.loads(body)
        except Exception:
            err_json = {"error": body, "status_code": e.code}
        return False, err_json
    except Exception as exc:
        return False, {"error": str(exc)}


def record_consent_via_api(
    backend_url: str, user_id: str, guardian_id: str = "guardian_admin"
) -> Tuple[bool, Dict[str, Any]]:
    """Legitimately record a ConsentRecord via official POST /api/consent/record endpoint."""
    url = f"{backend_url.rstrip('/')}/api/consent/record"
    consent_payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": guardian_id,
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }
    body_bytes = json.dumps(consent_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"raw_error": err_body}
        return False, err_json
    except Exception as exc:
        return False, {"error": str(exc)}


def setup_isolated_profile(
    user_id: str, source_user_id: str = "000"
) -> Optional[Path]:
    """Copy an existing profile baseline to create an isolated profile for the test user."""
    profiles_dir = _REPO_ROOT / "ml" / "models" / "profiles"
    src_path = profiles_dir / f"{source_user_id}.json"
    dest_path = profiles_dir / f"{user_id}.json"

    if not src_path.exists():
        return None

    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["user_id"] = user_id
        # Update user_id in anchor clusters if present
        for cluster in data.get("anchor_clusters", []):
            cluster["user_id"] = user_id
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return dest_path
    except Exception:
        return None


def fetch_risk_history(
    backend_url: str, user_id: str, limit: int = 10
) -> Tuple[int, List[Dict[str, Any]]]:
    """Query GET /api/risk-history/{user_id}?limit={limit} to verify persisted history."""
    url = f"{backend_url.rstrip('/')}/api/risk-history/{user_id}?limit={limit}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        return e.code, []
    except Exception:
        return 0, []


def build_payloads(
    raw_points: List[Dict[str, Any]],
    user_id: str,
    preserve_timestamps: bool = False,
    assumed_accuracy_m: float = 5.0,
    assumed_battery_pct: int = 85,
    assumed_activity_type: str = "WALKING",
) -> List[Dict[str, Any]]:
    """Derive kinematics and construct schema-valid TelemetryPayload dictionaries."""
    if not raw_points:
        return []

    # 1. Compute shifted timestamps if not preserving original 2008 dates
    if preserve_timestamps:
        timestamps = [p["original_timestamp"] for p in raw_points]
    else:
        now = datetime.now(timezone.utc)
        t_start = raw_points[0]["original_timestamp"]
        t_end = raw_points[-1]["original_timestamp"]
        total_duration_sec = (t_end - t_start).total_seconds()

        timestamps = []
        for p in raw_points:
            offset_sec = (p["original_timestamp"] - t_start).total_seconds()
            shifted_dt = now - timedelta(seconds=(total_duration_sec - offset_sec))
            timestamps.append(shifted_dt)

    # 2. Derive speed and heading between consecutive points
    payloads: List[Dict[str, Any]] = []
    prev_pt: Optional[Dict[str, Any]] = None
    prev_dt: Optional[datetime] = None
    prev_heading = 0.0

    for idx, (pt, dt) in enumerate(zip(raw_points, timestamps)):
        lat = pt["latitude"]
        lon = pt["longitude"]
        alt_m = pt["altitude_m"]

        if prev_pt is None or prev_dt is None:
            speed_mps = 0.0
            heading_deg = 0.0
        else:
            delta_dist_m = haversine_distance_m(prev_pt["latitude"], prev_pt["longitude"], lat, lon)
            delta_t_sec = max(0.0, (dt - prev_dt).total_seconds())

            if delta_t_sec > 0.0:
                speed_mps = round(delta_dist_m / delta_t_sec, 2)
                heading_deg = forward_azimuth_deg(prev_pt["latitude"], prev_pt["longitude"], lat, lon)
                prev_heading = heading_deg
            else:
                speed_mps = 0.0
                heading_deg = prev_heading

        payload: Dict[str, Any] = {
            "user_id": user_id,
            "timestamp": dt.isoformat(),
            "location": {
                "lat": round(lat, 7),
                "lng": round(lon, 7),
                "altitude_m": alt_m,
            },
            "sensor_metrics": {
                "horizontal_accuracy_m": assumed_accuracy_m,
                "speed_mps": speed_mps,
                "heading_deg": heading_deg,
                "activity_type": assumed_activity_type,
                "battery_pct": assumed_battery_pct,
            },
            "signal_status": {
                "state": "VALID",
                "degraded_since": None,
            },
            "imu_metrics": None,
        }

        payloads.append(payload)
        prev_pt = pt
        prev_dt = dt

    return payloads


def post_telemetry(backend_url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """POST TelemetryPayload to /api/telemetry/ingest using standard library HTTP."""
    url = f"{backend_url.rstrip('/')}/api/telemetry/ingest"
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"raw_error": err_body}
        return e.code, err_json
    except Exception as exc:
        return 0, {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay GeoLife .plt trajectory through backend telemetry endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="backend/tests/fixtures/sample_trajectory.plt",
        help="Path to GeoLife .plt trajectory file",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="000",
        help="Target user ID to simulate",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend base URL",
    )
    parser.add_argument(
        "--preserve-timestamps",
        action="store_true",
        help="Preserve original 2008 timestamps instead of shifting to current time window",
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.5,
        help="Fixed transmission delay in seconds between telemetry readings",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=None,
        help="Replay speed multiplier relative to recorded intervals (overrides --delay-sec)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of points to replay (None = all points)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parsing and telemetry generation offline without sending HTTP requests",
    )
    parser.add_argument(
        "--accuracy-m",
        type=float,
        default=5.0,
        help="[ASSUMPTION] Nominal GPS horizontal accuracy fixture in meters",
    )
    parser.add_argument(
        "--battery-pct",
        type=int,
        default=85,
        help="[ASSUMPTION] Nominal mobile battery percentage fixture [0-100]",
    )
    parser.add_argument(
        "--activity-type",
        type=str,
        default="WALKING",
        choices=["STATIONARY", "WALKING", "IN_VEHICLE", "UNKNOWN"],
        help="[ASSUMPTION] Sensor activity classification enum fixture",
    )
    parser.add_argument(
        "--record-consent",
        action="store_true",
        help="Explicitly record legitimate test consent via POST /api/consent/record before replaying",
    )
    parser.add_argument(
        "--setup-profile",
        action="store_true",
        help="Copy source profile to create an isolated personalized profile for the test user",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _REPO_ROOT / input_path
    if not input_path.exists():
        fallback = _REPO_ROOT / "backend" / "tests" / "fixtures" / "sample_trajectory.plt"
        if fallback.exists():
            print(f"[Notice] Trajectory file '{input_path}' not found; falling back to tracked fixture '{fallback.name}'")
            input_path = fallback

    print("=" * 80)
    print("GeoLife Trajectory Replay Utility")
    print("=" * 80)
    print(f"File:               {input_path}")
    print(f"Target User ID:     {args.user_id}")
    print(f"Backend URL:        {args.backend_url}")
    print(f"Timestamp Mode:     {'Preserve Original 2008' if args.preserve_timestamps else 'Shifted to Current UTC Window'}")
    print(f"Assumed Accuracy:   {args.accuracy_m}m  (explicit fixture assumption)")
    print(f"Assumed Battery:    {args.battery_pct}%   (explicit fixture assumption)")
    print(f"Assumed Activity:   {args.activity_type} (explicit fixture assumption)")
    print(f"IMU Metrics:        None   (strictly un-fabricated)")
    print("=" * 80)

    # Setup isolated profile if requested
    if args.setup_profile:
        prof_path = setup_isolated_profile(args.user_id, source_user_id="000")
        if prof_path:
            print(f"[Profile Setup] Isolated personalized profile established at: {prof_path}")
        else:
            print(f"[Profile Setup] Could not establish isolated profile for '{args.user_id}'")

    # Explicit consent recording if requested
    if args.record_consent:
        ok, res = record_consent_via_api(args.backend_url, args.user_id)
        if ok:
            print(f"[Consent Setup] Explicit consent recorded for '{args.user_id}': {res}")
        else:
            print(f"[Consent Setup] Failed to record consent for '{args.user_id}': {res}")
            return 2

    # 1. Parse .plt file
    try:
        raw_points = parse_geolife_plt(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to parse GeoLife trajectory: {exc}")
        return 1

    if not raw_points:
        print("[ERROR] Trajectory file contains 0 valid GPS points.")
        return 1

    if args.limit is not None and args.limit > 0:
        raw_points = raw_points[:args.limit]

    print(f"[OK] Parsed {len(raw_points)} valid GPS points from trajectory.")
    t_start = raw_points[0]["original_timestamp"]
    t_end = raw_points[-1]["original_timestamp"]
    duration_s = (t_end - t_start).total_seconds()
    print(f"[OK] Trajectory span: {duration_s:.1f}s ({duration_s/60:.1f} min), start: {t_start}, end: {t_end}")

    # 2. Build schema-valid TelemetryPayloads
    payloads = build_payloads(
        raw_points=raw_points,
        user_id=args.user_id,
        preserve_timestamps=args.preserve_timestamps,
        assumed_accuracy_m=args.accuracy_m,
        assumed_battery_pct=args.battery_pct,
        assumed_activity_type=args.activity_type,
    )
    print(f"[OK] Constructed {len(payloads)} schema-conforming TelemetryPayload records.")
    print(f"     First point timestamp: {payloads[0]['timestamp']}")
    print(f"     Last point timestamp:  {payloads[-1]['timestamp']}")

    # 3. Check consent via official endpoint
    is_consented, consent_resp = check_consent_status(args.backend_url, args.user_id)
    print(f"\n[Consent Check] Querying {args.backend_url}/api/consent/{args.user_id}...")
    if is_consented:
        print(f"[Consent Check] [GRANTED] User '{args.user_id}' has an active consent record on file.")
    else:
        print(f"[Consent Check] [NOT GRANTED] Consent status for '{args.user_id}': {consent_resp}")
        print("\n" + "!" * 80)
        print("MANDATORY CONSENT REQUIREMENT:")
        print("Per project specifications and medical consent compliance rules:")
        print("This replay utility does NOT automatically fabricate or register consent.")
        print("Telemetry ingestion will be rejected with HTTP 403 until consent is recorded.")
        print("\nTo legitimately register consent for this test user, you can run with --record-consent, or:")
        print(f"curl -X POST \"{args.backend_url}/api/consent/record\" \\")
        print("  -H \"Content-Type: application/json\" \\")
        print(f"  -d '{{\"user_id\": \"{args.user_id}\", \"guardian_consent\": {{\"granted\": true, \"granted_at\": \"{datetime.now(timezone.utc).isoformat()}\", \"guardian_id\": \"guardian_legit\"}}, \"end_user_consent_applicable\": false, \"data_retention_ack\": true}}'")
        print("!" * 80 + "\n")

        if not args.dry_run:
            print("[HALTED] Replay stopped because user is not consented. Grant consent first.")
            return 2

    # 4. Dry-run mode exits cleanly here after full validation
    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Telemetry validation passed with zero network transmissions.")
        print("Sample TelemetryPayload (point 1):")
        print(json.dumps(payloads[0], indent=2))
        if len(payloads) > 1:
            print("\nSample TelemetryPayload (point 2):")
            print(json.dumps(payloads[1], indent=2))
        return 0

    # 5. Live transmission loop
    print("\nStarting live trajectory streaming...")
    print("-" * 100)
    print(
        f"{'Pt':<5} {'Timestamp (UTC)':<24} {'Location (lat, lng)':<24} "
        f"{'Speed':<8} {'Head':<6} {'HTTP':<6} {'Risk Tier':<14} {'Score':<8} {'ML Status'}"
    )
    print("-" * 100)

    success_count = 0
    fail_count = 0

    for idx, p in enumerate(payloads, 1):
        status_code, resp_data = post_telemetry(args.backend_url, p)

        loc_str = f"({p['location']['lat']:.5f}, {p['location']['lng']:.5f})"
        spd_str = f"{p['sensor_metrics']['speed_mps']:.1f}m/s"
        head_str = f"{p['sensor_metrics']['heading_deg']:.0f}°"

        if status_code == 200:
            success_count += 1
            risk_tier = resp_data.get("risk_tier", "-")
            risk_score = f"{resp_data.get('risk_score', 0.0):.1f}"
            ml_eval = resp_data.get("ml_evaluation")
            if ml_eval:
                ml_stat = f"{ml_eval.get('engine_status')}/{ml_eval.get('evidence_status')}"
            else:
                ml_stat = "DISABLED"
        else:
            fail_count += 1
            risk_tier = "ERR"
            risk_score = "-"
            ml_stat = resp_data.get("detail") or resp_data.get("raw_error") or f"HTTP {status_code}"

        print(
            f"[{idx:03d}] {p['timestamp']:<24} {loc_str:<24} {spd_str:<8} {head_str:<6} "
            f"{status_code:<6} {risk_tier:<14} {risk_score:<8} {ml_stat}"
        )

        # Transmission pacing
        if idx < len(payloads):
            if args.speed_multiplier and idx > 0:
                # Interval based on delta_t between consecutive points
                dt_curr = datetime.fromisoformat(payloads[idx]["timestamp"])
                dt_prev = datetime.fromisoformat(p["timestamp"])
                delta_s = max(0.0, (dt_curr - dt_prev).total_seconds())
                sleep_s = delta_s / max(0.1, args.speed_multiplier)
            else:
                sleep_s = max(0.0, args.delay_sec)

            if sleep_s > 0:
                time.sleep(sleep_s)

    print("-" * 100)
    print(f"Replay complete! Sent: {len(payloads)}, Success: {success_count}, Failed: {fail_count}")

    if success_count > 0:
        print("\nVerifying persisted risk history via GET /api/risk-history...")
        h_code, history_rows = fetch_risk_history(args.backend_url, args.user_id, limit=5)
        print(f"[History Query] GET /api/risk-history/{args.user_id}?limit=5 -> HTTP {h_code}, rows retrieved: {len(history_rows)}")
        for h in history_rows[:5]:
            ml_e = h.get("ml_evaluation")
            ml_desc = f"ML: {ml_e.get('engine_status')}/{ml_e.get('evidence_status')}" if ml_e else "ML: None"
            print(f"  Row ID {h.get('id')}: {h.get('timestamp')} | Tier: {h.get('risk_tier')} | Score: {h.get('risk_score')} | {ml_desc}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

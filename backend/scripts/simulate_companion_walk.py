#!/usr/bin/env python3
"""
Development Companion Walking Simulation Script.

Simulates an authorized companion walking alongside a selected patient.
Supports 3 scenarios:
  1. together: Walking side-by-side (< 10m). Status: Together. No alert.
  2. separating: Diverging beyond 50m. Step 1: Separation Pending. Step 2: Separated.
     With elevated HR (>=100 BPM): Escalate to Tier 3 Alert.
  3. rejoining: Moving back within 50m. Step 1: Rejoining. Step 2: Together. Auto-resolves alert.
  4. full_cycle: Executes together -> separating -> rejoining sequentially.

Usage:
  python backend/scripts/simulate_companion_walk.py --patient-id sim_01 --scenario full_cycle
"""
import argparse
import sys
import time
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_scenario(
    base_url: str,
    patient_id: str,
    scenario: str,
    heart_rate_bpm: int | None = None,
    delay_sec: float = 1.0,
):
    print("=" * 80)
    print(f" MOBILITY GUARDIAN - COMPANION WALKING SIMULATION: {scenario.upper()}")
    print("=" * 80)
    print(f" Patient ID : {patient_id}")
    print(f" Backend    : {base_url}")
    print(f" Data Label : SIMULATED_COMPANION_DATA (Dev Test Only)")
    print("-" * 80)

    client = httpx.Client(base_url=base_url, timeout=10.0)

    # 1. Ensure consent
    print("1. Checking patient consent...")
    consent_res = client.get(f"/api/consent/{patient_id}")
    if consent_res.status_code == 404 or not consent_res.json().get("guardian_consent", {}).get("granted"):
        print("   Recording guardian consent for dev test...")
        c_res = client.post(
            "/api/consent/record",
            json={
                "user_id": patient_id,
                "guardian_consent": {
                    "granted": True,
                    "granted_at": "2026-10-23T10:00:00Z",
                    "guardian_id": "guardian_demo",
                },
                "end_user_consent_applicable": False,
                "data_retention_ack": True,
            },
        )
        if c_res.status_code != 200:
            print(f"   [Error] Could not grant consent: {c_res.text}")
            sys.exit(1)
    print("   [OK] Consent verified.")

    # Determine steps to execute
    if scenario == "together":
        steps = [("WALKING_TOGETHER", 1, heart_rate_bpm or 75), ("WALKING_TOGETHER", 2, heart_rate_bpm or 74)]
    elif scenario == "separating":
        steps = [("SEPARATING", 1, heart_rate_bpm or 75), ("SEPARATING", 2, heart_rate_bpm or 105)]
    elif scenario == "rejoining":
        steps = [("REJOINING", 1, heart_rate_bpm or 78), ("REJOINING", 2, heart_rate_bpm or 75)]
    elif scenario == "full_cycle":
        steps = [
            ("WALKING_TOGETHER", 1, 75),
            ("WALKING_TOGETHER", 2, 74),
            ("SEPARATING", 1, 80),
            ("SEPARATING", 2, 105),  # Elevated HR -> Tier 3
            ("REJOINING", 1, 90),
            ("REJOINING", 2, 75),   # Reunited -> Auto-Resolved
        ]
    else:
        print(f"Unknown scenario: {scenario}")
        sys.exit(1)

    print("\n2. Executing Simulation Telemetry Steps:")
    for idx, (sc_name, step_no, hr) in enumerate(steps, 1):
        payload = {
            "patient_id": patient_id,
            "scenario": sc_name,
            "step_number": step_no,
            "companion_id": "comp_alex",
            "companion_alias": "Family Guide (Alex)",
            "heart_rate_bpm": hr,
        }

        resp = client.post("/api/companion/simulation/step", json=payload)
        if resp.status_code != 200:
            print(f"   Step {idx} [{sc_name} #{step_no}] HTTP {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        disp_status = data["display_status"]
        dist = data["distance_meters"]
        action = data["action"]
        alert_id = data["alert_id"]
        tier = data["alert_tier"]
        hr_info = data.get("heart_rate", {})

        hr_str = f"{hr_info.get('bpm', '-')} BPM"
        if hr_info.get("is_arousal"):
            hr_str += f" (Arousal: z={hr_info.get('z_score')})"

        alert_str = f"Alert: {action}"
        if alert_id:
            alert_str += f" [ID: {alert_id[:8]}... Tier {tier}]"

        p_coords = data["patient_coords"]
        c_coords = data["companion_coords"]

        print(
            f"   Step {idx} | {sc_name:<16} | Status: {disp_status:<18} | "
            f"Dist: {dist:5.1f}m | HR: {hr_str:<28} | {alert_str}"
        )
        print(f"          Patient: ({p_coords[0]:.6f}, {p_coords[1]:.6f})  |  Companion: ({c_coords[0]:.6f}, {c_coords[1]:.6f})")

        if idx < len(steps):
            time.sleep(delay_sec)

    print("-" * 80)
    print(" Simulation completed successfully.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Mobility Guardian Companion Walking Simulator")
    parser.add_argument("--patient-id", default="sim_01", help="Target patient identifier (default: sim_01)")
    parser.add_argument(
        "--scenario",
        default="full_cycle",
        choices=["together", "separating", "rejoining", "full_cycle"],
        help="Simulation scenario",
    )
    parser.add_argument("--heart-rate", type=int, default=None, help="Explicit heart rate in BPM")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between steps in seconds")

    args = parser.parse_args()
    run_scenario(
        base_url=args.url,
        patient_id=args.patient_id,
        scenario=args.scenario,
        heart_rate_bpm=args.heart_rate,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()

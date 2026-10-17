#!/usr/bin/env python3
"""
Evaluation Fixture Replay Runner for Mobility Guardian.

Safely replays deterministic JSON trajectory fixtures through the live backend
telemetry ingestion endpoint (/api/telemetry/ingest) and captures atomic operational
and ML evaluation outputs.

Mandatory Safety & Isolation Controls:
1. Dry-Run by Default: Refuses to send any network traffic without explicit --execute opt-in.
2. Fixture Hash Integrity: Re-computes SHA-256 over canonical payloads before transmission.
3. Fail-Closed Database Isolation Gate:
   - Verifies effective DATABASE_URL and DATABASE_PATH point to an isolated evaluation DB.
   - Refuses to run if database path matches app.db or demo_app.db.
   - Takes baseline cryptographic SHA-256 snapshots of app.db and demo_app.db.
   - Runs an isolated probe consent check and verifies app.db remains byte-identical.
4. Output Isolation: Writes evaluation results exclusively to backend/app/data/eval_phase3/results/.
5. Metric Rigor: Distinguishes accumulating/insufficient evidence from true negatives.

NOTE: This script is prepared for Phase 3B and is NOT executed during Phase 3B.1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Protected production & demo database paths
_APP_DB_PATH = _REPO_ROOT / "backend" / "app" / "data" / "app.db"
_DEMO_DB_PATH = _REPO_ROOT / "backend" / "app" / "data" / "demo_app.db"
_DEFAULT_EVAL_DB = _REPO_ROOT / "backend" / "app" / "data" / "eval_phase3" / "eval_phase3.db"

_APP_DB_BASELINE_SHA = "3a1d91b1050ff9f688850578983877872f2bc2c2dd52a9d33cdbbb8d3f9699f9"
_DEMO_DB_BASELINE_SHA = "dbf5012f0c30c4e19387138e20d937c29c8cbe72af81f133fd42d29eda6771e3"


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic, RFC 8785-compatible canonical JSON byte serialization."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 digest of byte content."""
    return hashlib.sha256(data).hexdigest()


def get_file_snapshot(path: Path) -> Tuple[Optional[str], Optional[int]]:
    """Capture (sha256, byte_size) snapshot of a file if it exists."""
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    return compute_sha256(raw), len(raw)


# ==============================================================================
# INTEGRITY & ISOLATION GATES
# ==============================================================================


def verify_fixture_integrity(fixture_data: Dict[str, Any], file_path: Path) -> Tuple[bool, str]:
    """Verify that fixture canonical payload bytes match provenance SHA-256 hash."""
    provenance = fixture_data.get("provenance", {})
    expected_sha = provenance.get("payload_canonical_sha256")
    if not expected_sha:
        return False, f"Fixture {file_path.name} is missing 'payload_canonical_sha256' in provenance."

    payloads = fixture_data.get("payloads", [])
    if not isinstance(payloads, list) or len(payloads) == 0:
        return False, f"Fixture {file_path.name} contains empty or invalid 'payloads' array."

    actual_sha = compute_sha256(canonical_json_bytes(payloads))
    if actual_sha != expected_sha:
        return False, (
            f"Fixture hash divergence in {file_path.name}! "
            f"Expected {expected_sha}, computed {actual_sha}."
        )

    return True, actual_sha


def enforce_database_isolation_gate(
    backend_url: str,
    eval_db_path: Path,
    app_db_baseline_sha: Optional[str],
    demo_db_baseline_sha: Optional[str],
) -> None:
    """Fail-closed safety gate executed prior to sending test telemetry."""
    print("\n--- EXECUTING FAIL-CLOSED DATABASE ISOLATION GATE ---")

    # 1. Inspect environment variable assertions
    env_db_path = os.getenv("DATABASE_PATH", "")
    env_db_url = os.getenv("DATABASE_URL", "")

    print(f"Declared Eval DB Target: {eval_db_path.resolve()}")
    print(f"Process DATABASE_PATH:   {env_db_path or '[UNSET]'}")
    print(f"Process DATABASE_URL:    {env_db_url or '[UNSET]'}")

    # Prohibit pointing to app.db or demo_app.db under any circumstances
    for prohibited in ("app.db", "demo_app.db"):
        if prohibited in env_db_path.lower() or prohibited in env_db_url.lower():
            raise RuntimeError(
                f"[CRITICAL BLOCKER] Environment variable points to protected database '{prohibited}'! "
                "Execution aborted to prevent database contamination."
            )

    # 2. Check backend health
    health_url = f"{backend_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(health_url)
            if resp.status_code != 200:
                raise RuntimeError(f"Backend health check failed: HTTP {resp.status_code} - {resp.text}")
    except Exception as exc:
        raise RuntimeError(f"Could not connect to backend at {health_url}: {exc}")

    print("[Check 1] Backend health check PASSED (HTTP 200).")

    # 3. Post isolated probe consent record to verify active database binding
    probe_user_id = "eval_gate_probe_user"
    consent_url = f"{backend_url.rstrip('/')}/api/consent/record"
    probe_payload = {
        "user_id": probe_user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "eval_isolation_gate",
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(consent_url, json=probe_payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Probe consent registration failed: HTTP {resp.status_code} - {resp.text}")
    except Exception as exc:
        raise RuntimeError(f"Failed to post probe consent to backend: {exc}")

    print(f"[Check 2] Probe consent registered for user '{probe_user_id}'.")

    # 4. Directly verify that probe record was written to eval_db_path
    if not eval_db_path.is_file():
        raise RuntimeError(
            f"[CRITICAL ISOLATION FAILURE] Evaluation database file '{eval_db_path}' does not exist on disk! "
            "Backend is writing to a different, unidentified database location."
        )

    try:
        conn = sqlite3.connect(str(eval_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM consent_records WHERE user_id = ?", (probe_user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise RuntimeError(
                f"[CRITICAL ISOLATION FAILURE] Probe user '{probe_user_id}' NOT found in '{eval_db_path}'! "
                "Backend is bound to another database file."
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect evaluation database '{eval_db_path}': {exc}")

    print(f"[Check 3] Verified probe user presence inside target eval database: {eval_db_path.name}.")

    # 5. Verify that protected app.db was NOT touched (SHA-256 and size comparison)
    if _APP_DB_PATH.is_file():
        current_app_sha, current_app_size = get_file_snapshot(_APP_DB_PATH)
        if current_app_sha != app_db_baseline_sha:
            raise RuntimeError(
                f"[CRITICAL DATA LEAK] Protected database '{_APP_DB_PATH.name}' SHA-256 hash changed! "
                f"Baseline: {app_db_baseline_sha}, Current: {current_app_sha}."
            )
        # Check SQLite table to verify probe does not exist in app.db
        try:
            conn_app = sqlite3.connect(str(_APP_DB_PATH))
            cursor_app = conn_app.cursor()
            cursor_app.execute("SELECT user_id FROM consent_records WHERE user_id = ?", (probe_user_id,))
            leak_row = cursor_app.fetchone()
            conn_app.close()
            if leak_row:
                raise RuntimeError(
                    f"[CRITICAL DATA LEAK] Probe user '{probe_user_id}' was written to '{_APP_DB_PATH.name}'!"
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect protected database '{_APP_DB_PATH}': {exc}")

    print("[Check 4] Verified protected development database (app.db) remains 100% UNCHANGED.")

    # 6. Verify that protected demo_app.db was NOT touched
    if _DEMO_DB_PATH.is_file() and demo_db_baseline_sha is not None:
        current_demo_sha, _ = get_file_snapshot(_DEMO_DB_PATH)
        if current_demo_sha != demo_db_baseline_sha:
            raise RuntimeError(
                f"[CRITICAL DATA LEAK] Protected database '{_DEMO_DB_PATH.name}' SHA-256 hash changed!"
            )

    print("[Check 5] Verified protected demo database (demo_app.db) remains 100% UNCHANGED.")
    print("--- DATABASE ISOLATION GATE VERIFIED: PASS ---\n")


# ==============================================================================
# REPLAY ENGINE
# ==============================================================================


def register_user_consent(backend_url: str, user_id: str) -> None:
    """Register consent record for a pilot test user prior to streaming."""
    url = f"{backend_url.rstrip('/')}/api/consent/record"
    payload = {
        "user_id": user_id,
        "guardian_consent": {
            "granted": True,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "guardian_id": "pilot_evaluator",
        },
        "end_user_consent_applicable": False,
        "end_user_consent": None,
        "data_retention_ack": True,
    }
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to register consent for '{user_id}': HTTP {resp.status_code} - {resp.text}")


def replay_single_fixture(
    backend_url: str,
    fixture_path: Path,
    pacing_delay_sec: float = 0.05,
) -> Dict[str, Any]:
    """Replay a verified fixture through /api/telemetry/ingest preserving timestamps."""
    with open(fixture_path, "r", encoding="utf-8") as fp:
        fixture_data = json.load(fp)

    valid, sha_or_err = verify_fixture_integrity(fixture_data, fixture_path)
    if not valid:
        raise ValueError(f"Integrity check failed: {sha_or_err}")

    provenance = fixture_data["provenance"]
    payloads = fixture_data["payloads"]
    user_id = provenance["user_id"]

    # Register consent for this user
    register_user_consent(backend_url, user_id)

    ingest_url = f"{backend_url.rstrip('/')}/api/telemetry/ingest"
    records: List[Dict[str, Any]] = []

    print(f"\nStreaming fixture '{provenance['fixture_id']}' ({len(payloads)} points)...")

    with httpx.Client(timeout=10.0) as client:
        for idx, p in enumerate(payloads, 1):
            t_send = time.perf_counter()
            resp = client.post(ingest_url, json=p)
            latency_ms = round((time.perf_counter() - t_send) * 1000.0, 2)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Point {idx}/{len(payloads)} failed with HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            ml_eval = data.get("ml_evaluation") or {}

            # Determine evidence status without treating missing as negative
            ev_status = ml_eval.get("evidence_status", "NO_EVALUATION")
            cal_prob = ml_eval.get("calibrated_probability")
            raw_prob = ml_eval.get("raw_probability")

            records.append({
                "step": idx,
                "timestamp": p["timestamp"],
                "speed_mps": p["sensor_metrics"]["speed_mps"],
                "latency_ms": latency_ms,
                "operational_risk_score": data.get("risk_score"),
                "operational_risk_tier": data.get("risk_tier"),
                "polling_tier": data.get("polling_tier"),
                "ml_evidence_status": ev_status,
                "ml_calibrated_probability": cal_prob,
                "ml_raw_probability": raw_prob,
                "ml_risk_tier": ml_eval.get("ml_risk_tier"),
                "ml_decision_state": ml_eval.get("decision_state"),
                "ml_top_features": ml_eval.get("explainability", {}).get("top_features", []),
            })

            if pacing_delay_sec > 0:
                time.sleep(pacing_delay_sec)

    return {
        "fixture_id": provenance["fixture_id"],
        "phenotype_intent": provenance["phenotype_intent"],
        "user_id": user_id,
        "point_count": len(payloads),
        "payload_sha256": sha_or_err,
        "eval_completed_at": datetime.now(timezone.utc).isoformat(),
        "step_records": records,
    }


# ==============================================================================
# CLI DISPATCH
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Trajectory Fixture Replay Runner for Mobility Guardian"
    )
    parser.add_argument(
        "--fixture-dir",
        type=str,
        default=str(_REPO_ROOT / "backend" / "app" / "data" / "eval_phase3" / "fixtures"),
        help="Directory containing pilot_p*.json fixtures",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_REPO_ROOT / "backend" / "app" / "data" / "eval_phase3" / "results"),
        help="Directory for evaluation results output",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend base URL",
    )
    parser.add_argument(
        "--expected-db",
        type=str,
        default=str(_DEFAULT_EVAL_DB),
        help="Expected path of isolated evaluation database",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicit opt-in required to execute replay over HTTP. Without this, runner executes dry-run validation only.",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Specific fixture file name or path (e.g. pilot_p1_normal.json) to replay alone.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results output file if present",
    )

    args = parser.parse_args()

    fixture_dir = Path(args.fixture_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    eval_db_path = Path(args.expected_db).resolve()

    print("================================================================================")
    print("MOBILITY GUARDIAN — EVALUATION FIXTURE REPLAY RUNNER")
    print("================================================================================")
    print(f"Fixture Directory:  {fixture_dir}")
    print(f"Results Directory:  {results_dir}")
    print(f"Target Eval DB:     {eval_db_path}")
    print(f"Execution Mode:     {'LIVE EXECUTION' if args.execute else 'DRY-RUN / VERIFY-ONLY'}")
    if args.fixture:
        print(f"Target Fixture:     {args.fixture}")
    print("-" * 80)

    # 1. Discover and verify fixtures offline
    if args.fixture:
        target_path = Path(args.fixture)
        if not target_path.is_file():
            target_path = fixture_dir / args.fixture
        if not target_path.is_file():
            print(f"[ERROR] Specified fixture file '{args.fixture}' not found.")
            sys.exit(1)
        fixture_files = [target_path.resolve()]
    else:
        fixture_files = sorted(fixture_dir.glob("pilot_p*.json"))

    if not fixture_files:
        print(f"[ERROR] No fixture files matching 'pilot_p*.json' found in {fixture_dir}.")
        print("Please run backend/scripts/generate_eval_fixtures.py first.")
        sys.exit(1)

    print(f"Discovered {len(fixture_files)} fixture(s). Verifying integrity...")
    all_valid = True
    for ff in fixture_files:
        with open(ff, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        valid, sha_or_err = verify_fixture_integrity(data, ff)
        if valid:
            print(f"  [PASS] {ff.name} -> SHA-256: {sha_or_err}")
        else:
            print(f"  [FAIL] {ff.name} -> {sha_or_err}")
            all_valid = False

    if not all_valid:
        print("\n[CRITICAL ERROR] One or more fixtures failed integrity verification. Aborting.")
        sys.exit(1)

    print("\n[SUCCESS] Fixture integrity verified for byte-for-byte canonical SHA-256.")

    # 2. Check execution opt-in
    if not args.execute:
        print("\n[DRY-RUN COMPLETE] --execute flag was not specified. Network streaming skipped.")
        print("To run live evaluation against an active isolated server, provide --execute.")
        sys.exit(0)

    # 3. If --execute is provided, enforce isolation gates and run replay
    app_sha, _ = get_file_snapshot(_APP_DB_PATH)
    demo_sha, _ = get_file_snapshot(_DEMO_DB_PATH)

    if app_sha != _APP_DB_BASELINE_SHA:
        raise RuntimeError(
            f"[FATAL ERROR] app.db does not match baseline SHA-256! "
            f"Expected: {_APP_DB_BASELINE_SHA}, Got: {app_sha}"
        )
    if demo_sha != _DEMO_DB_BASELINE_SHA:
        raise RuntimeError(
            f"[FATAL ERROR] demo_app.db does not match baseline SHA-256! "
            f"Expected: {_DEMO_DB_BASELINE_SHA}, Got: {demo_sha}"
        )

    enforce_database_isolation_gate(
        backend_url=args.backend_url,
        eval_db_path=eval_db_path,
        app_db_baseline_sha=app_sha,
        demo_db_baseline_sha=demo_sha,
    )

    # 4. Prepare results output file (non-overwriting by default)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        base_name = fixture_files[0].stem
        summary_file = results_dir / f"{base_name}_results.json"
    else:
        summary_file = results_dir / "pilot_summary.json"

    if summary_file.exists() and not args.overwrite:
        timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_file = results_dir / f"{summary_file.stem}_{timestamp_suffix}.json"

    # 5. Execute replay for selected fixture(s)
    all_results: List[Dict[str, Any]] = []
    for ff in fixture_files:
        res = replay_single_fixture(backend_url=args.backend_url, fixture_path=ff)
        all_results.append(res)

    # 6. Post-execution database immutability verification
    print("\n--- POST-EXECUTION DATABASE IMMUTABILITY VERIFICATION ---")
    post_app_sha, _ = get_file_snapshot(_APP_DB_PATH)
    if post_app_sha != app_sha:
        raise RuntimeError(
            f"[FATAL BREACH] app.db hash mutated during evaluation run! "
            f"Pre: {app_sha}, Post: {post_app_sha}."
        )
    print("[PASS] app.db SHA-256 verified identical to pre-run baseline.")

    post_demo_sha, _ = get_file_snapshot(_DEMO_DB_PATH)
    if post_demo_sha != demo_sha:
        raise RuntimeError(
            f"[FATAL BREACH] demo_app.db hash mutated during evaluation run! "
            f"Pre: {demo_sha}, Post: {post_demo_sha}."
        )
    print("[PASS] demo_app.db SHA-256 verified identical to pre-run baseline.")

    # 7. Write results
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump({
            "pilot_evaluation_version": "1.0.0",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "target_eval_db": str(eval_db_path),
            "fixture_count": len(all_results),
            "scenarios": all_results,
        }, fp, indent=2)

    print(f"\n[COMPLETE] Evaluation results written to: {summary_file}")



if __name__ == "__main__":
    main()

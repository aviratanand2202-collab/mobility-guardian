"""Focused Test Suite for ML Input and User Profile Flow (Chunk 1).

Verifies:
1. New user / Cold start without history: uses population baseline, profile_mode=COLD_START, no fabricated history.
2. Existing user / Personalized with history: uses personal baseline, profile_mode=PERSONALIZED, trip_count >= 7.
3. Empty or invalid history: safe fallback to COLD_START without crashing.
4. No future-data leakage: current trajectory strictly isolated from historical profile.
5. Cold-start -> Personalized transition: 1-6 trips remain COLD_START, 7th trip promotes to PERSONALIZED.
6. Conceptual stored user profile flow: history treated as stored user data without manual folder uploads.
"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile

import pytest

from ml.src.inference import RiskInferenceEngine
from ml.src.manual_inference import (
    build_personalized_baseline_from_history,
    run_manual_inference,
)


def _generate_synthetic_plt_content(start_dt: datetime, n_points: int = 240, speed_mps: float = 1.4) -> str:
    """Generate synthetically valid GeoLife .plt format content."""
    lines = [
        "Geolife trajectory",
        "WGS 84",
        "Altitude is in Feet",
        "Reserved 3",
        "0,2,255,My Track,0,0,2,8421376",
        "0",
    ]
    lat = 39.98470
    lon = 116.31840
    curr_t = start_dt

    for _ in range(n_points):
        # 1-second cadence, small spatial displacement
        lat += 0.00001 * (speed_mps / 1.4)
        lon += 0.00001 * (speed_mps / 1.4)
        date_str = curr_t.strftime("%Y-%m-%d")
        time_str = curr_t.strftime("%H:%M:%S")
        curr_t += timedelta(seconds=1)
        lines.append(f"{lat:.6f},{lon:.6f},0,100.0,40000.0000,{date_str},{time_str}")

    return "\n".join(lines)


# 1. New User Without History -> COLD_START
def test_new_user_without_history_cold_start():
    """Verify new user without history runs in COLD_START mode with population prior."""
    sample_plt = "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"
    if not os.path.exists(sample_plt):
        pytest.skip("GeoLife trajectory file not present.")

    engine = RiskInferenceEngine()
    test_user_id = "test_brand_new_patient_001"
    # Ensure no stored profile exists
    engine.delete_user_profile(test_user_id)

    summary = run_manual_inference(
        input_file=sample_plt,
        history_path=None,
        horizon="120",
        user_id=test_user_id,
    )

    assert summary["profile_mode"] == "COLD_START"
    assert summary["cold_start_status"] == "COLD_START"
    assert summary["trip_count"] == 0  # Never fabricate personal history
    assert summary["baseline_type"] == "POPULATION"

    # Verify predictions metadata
    windows_120 = summary["window_predictions"]["120s"]
    assert len(windows_120) > 0
    for w in windows_120:
        meta = w["metadata"]
        assert meta["profile_mode"] == "COLD_START"
        assert meta["cold_start_status"] == "COLD_START"
        assert meta["trip_count"] == 0
        assert meta["baseline_type"] == "POPULATION"


# 2. Existing User With History -> PERSONALIZED
def test_existing_user_with_history_personalized():
    """Verify existing user with >= 7 historical trips runs in PERSONALIZED mode."""
    target_plt = "ml/data/raw/geolife/000/Trajectory/20081122012309.plt"
    history_dir = "ml/data/raw/geolife/000/Trajectory"
    if not os.path.exists(target_plt) or not os.path.exists(history_dir):
        pytest.skip("GeoLife raw trajectory dataset not present.")

    summary = run_manual_inference(
        input_file=target_plt,
        history_path=history_dir,
        horizon="120",
        user_id="000",
    )

    assert summary["profile_mode"] == "PERSONALIZED"
    assert summary["cold_start_status"] == "ML_DRIVEN"
    assert summary["trip_count"] >= 7
    assert summary["baseline_type"] == "PERSONALIZED"

    windows_120 = summary["window_predictions"]["120s"]
    assert len(windows_120) > 0
    for w in windows_120:
        meta = w["metadata"]
        assert meta["profile_mode"] == "PERSONALIZED"
        assert meta["cold_start_status"] == "ML_DRIVEN"
        assert meta["baseline_type"] == "PERSONALIZED"


# 3. Empty or Invalid History -> Safe Fallback
def test_empty_or_invalid_history_safe_fallback():
    """Verify empty or invalid history directories safely fall back to COLD_START without crashing."""
    sample_plt = "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"
    if not os.path.exists(sample_plt):
        pytest.skip("GeoLife trajectory file not present.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # A: Non-existent directory
        summary_nonexistent = run_manual_inference(
            input_file=sample_plt,
            history_path=os.path.join(tmp_dir, "non_existent_subdir"),
            horizon="120",
        )
        assert summary_nonexistent["profile_mode"] == "COLD_START"
        assert summary_nonexistent["baseline_type"] == "POPULATION"

        # B: Empty directory
        empty_hist_dir = os.path.join(tmp_dir, "empty_history")
        os.makedirs(empty_hist_dir, exist_ok=True)
        summary_empty = run_manual_inference(
            input_file=sample_plt,
            history_path=empty_hist_dir,
            horizon="120",
        )
        assert summary_empty["profile_mode"] == "COLD_START"
        assert summary_empty["baseline_type"] == "POPULATION"

        # C: Directory containing corrupt/empty .plt files
        corrupt_hist_dir = os.path.join(tmp_dir, "corrupt_history")
        os.makedirs(corrupt_hist_dir, exist_ok=True)
        with open(os.path.join(corrupt_hist_dir, "corrupt_1.plt"), "w") as f:
            f.write("Corrupted unparseable header and no data\n")
        with open(os.path.join(corrupt_hist_dir, "empty_2.plt"), "w") as f:
            f.write("")

        summary_corrupt = run_manual_inference(
            input_file=sample_plt,
            history_path=corrupt_hist_dir,
            horizon="120",
        )
        assert summary_corrupt["profile_mode"] == "COLD_START"
        assert summary_corrupt["baseline_type"] == "POPULATION"


# 4. Strict Isolation and No Future-Data Leakage
def test_no_future_data_leakage():
    """Verify current trajectory and future data are strictly isolated from historical profile."""
    engine = RiskInferenceEngine()

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_ref = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Create target trajectory (at t_ref + 10 hours)
        t_target_start = t_ref + timedelta(hours=10)
        target_path = Path(tmp_dir) / "target_trajectory.plt"
        target_path.write_text(_generate_synthetic_plt_content(t_target_start, n_points=120, speed_mps=1.5))

        # 2. Create historical directory containing:
        #    - 7 legitimate past trips (t < t_target_start) with slow speed (1.0 m/s)
        #    - 1 future trip (t > t_target_start) with very fast speed (10.0 m/s)
        #    - A duplicate copy of target_trajectory.plt itself
        hist_dir = Path(tmp_dir) / "history"
        hist_dir.mkdir()

        # Legitimate past trips
        for i in range(7):
            past_t = t_ref + timedelta(hours=i)
            p_file = hist_dir / f"past_trip_{i:02d}.plt"
            p_file.write_text(_generate_synthetic_plt_content(past_t, n_points=240, speed_mps=1.0))

        # Future trip (must be excluded by as_of_time!)
        future_t = t_target_start + timedelta(hours=2)
        future_file = hist_dir / "future_trip_leakage_candidate.plt"
        future_file.write_text(_generate_synthetic_plt_content(future_t, n_points=240, speed_mps=10.0))

        # Duplicate of target file (must be excluded by current_trajectory_file!)
        dup_file = hist_dir / "target_trajectory.plt"
        dup_file.write_text(target_path.read_text())

        # Build baseline with isolation
        baseline, trip_count, mode = build_personalized_baseline_from_history(
            history_path=hist_dir,
            cfg=engine.risk_cfg,
            feat_cfg=engine.feature_cfg,
            min_trips=7,
            current_trajectory_file=target_path,
            as_of_time=t_target_start,
        )

        assert mode == "PERSONALIZED"
        assert trip_count == 7  # Exactly the 7 past trips; future and duplicate excluded!
        assert baseline is not None

        # Verify speed median in baseline reflects ONLY the 1.0 m/s past trips (not 10.0 m/s future trip)
        speed_median = baseline["mean_speed_mps"]["median"]
        assert 0.8 <= speed_median <= 1.2, f"Expected speed ~1.0 m/s from past trips, got {speed_median}"


# 5. Cold-Start -> Personalized Transition
def test_cold_start_to_personalized_transition():
    """Verify exact transition from COLD_START (< 7 trips) to PERSONALIZED (>= 7 trips)."""
    engine = RiskInferenceEngine()

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_base = datetime(2026, 2, 1, 8, 0, 0, tzinfo=timezone.utc)
        hist_dir = Path(tmp_dir) / "transition_history"
        hist_dir.mkdir()

        # Create trips 1 through 6
        for i in range(6):
            t_trip = t_base + timedelta(days=i)
            p_file = hist_dir / f"trip_{i + 1:02d}.plt"
            p_file.write_text(_generate_synthetic_plt_content(t_trip, n_points=120, speed_mps=1.3))

        # Check with 6 trips: must be COLD_START
        baseline_6, count_6, mode_6 = build_personalized_baseline_from_history(
            history_path=hist_dir,
            cfg=engine.risk_cfg,
            feat_cfg=engine.feature_cfg,
            min_trips=7,
        )
        assert mode_6 == "COLD_START"
        assert count_6 == 6
        assert baseline_6 is None

        # Add 7th trip
        t_trip_7 = t_base + timedelta(days=6)
        p_file_7 = hist_dir / "trip_07.plt"
        p_file_7.write_text(_generate_synthetic_plt_content(t_trip_7, n_points=120, speed_mps=1.3))

        # Check with 7 trips: must promote to PERSONALIZED
        baseline_7, count_7, mode_7 = build_personalized_baseline_from_history(
            history_path=hist_dir,
            cfg=engine.risk_cfg,
            feat_cfg=engine.feature_cfg,
            min_trips=7,
        )
        assert mode_7 == "PERSONALIZED"
        assert count_7 == 7
        assert baseline_7 is not None
        assert "mean_speed_mps" in baseline_7
        assert "tortuosity_index" in baseline_7


# 6. Conceptual Stored User Profile Flow
def test_conceptual_stored_user_profile_flow():
    """Verify stored user profile repository enables PERSONALIZED inference without folder upload."""
    sample_plt = "ml/data/raw/geolife/000/Trajectory/20081023025304.plt"
    if not os.path.exists(sample_plt):
        pytest.skip("GeoLife trajectory file not present.")

    engine = RiskInferenceEngine()
    test_user = "stored_patient_user_777"

    # Clean any prior state
    engine.delete_user_profile(test_user)

    # 1. Before profile storage: inference is COLD_START
    res_cold = run_manual_inference(
        input_file=sample_plt,
        history_path=None,
        horizon="120",
        user_id=test_user,
    )
    assert res_cold["profile_mode"] == "COLD_START"
    assert res_cold["baseline_type"] == "POPULATION"

    # 2. Store personalized profile conceptually
    synthetic_profile = {
        "user_id": test_user,
        "profile_mode": "PERSONALIZED",
        "cold_start_status": "ML_DRIVEN",
        "trip_count": 14,
        "baseline_distribution": {
            "mean_speed_mps": {"median": 1.4, "mad": 0.2, "robust_scale": 0.296, "p95": 2.1},
            "tortuosity_index": {"median": 1.1, "mad": 0.1, "robust_scale": 0.148, "p95": 1.5},
            "entropy_directional": {"median": 1.0, "mad": 0.2, "robust_scale": 0.296, "p95": 1.8},
            "path_distance_m": {"median": 160.0, "mad": 20.0, "robust_scale": 29.6, "p95": 250.0},
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    saved_path = engine.save_user_profile(test_user, synthetic_profile)
    assert saved_path.exists()

    # 3. Re-run inference with NO --history folder upload
    res_personalized = run_manual_inference(
        input_file=sample_plt,
        history_path=None,  # No folder upload!
        horizon="120",
        user_id=test_user,
    )
    assert res_personalized["profile_mode"] == "PERSONALIZED"
    assert res_personalized["cold_start_status"] == "ML_DRIVEN"
    assert res_personalized["trip_count"] == 14
    assert res_personalized["baseline_type"] == "PERSONALIZED"

    # 4. Clean up stored profile
    engine.delete_user_profile(test_user)
    assert not saved_path.exists()

    # 5. Verify revert back to COLD_START
    res_reverted = run_manual_inference(
        input_file=sample_plt,
        history_path=None,
        horizon="120",
        user_id=test_user,
    )
    assert res_reverted["profile_mode"] == "COLD_START"
    assert res_reverted["baseline_type"] == "POPULATION"

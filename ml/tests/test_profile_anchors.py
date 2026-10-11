"""Focused Regression Test Suite for Spatial Anchor Persistence and Familiarity Flow.

Verifies:
1. History with valid anchor evidence produces persisted anchor_clusters.
2. History with insufficient anchor evidence does NOT fabricate anchors.
3. Current trajectory is strictly excluded from anchor construction.
4. Stored profile can be reloaded with anchor_clusters intact.
5. A current point near a persisted learned anchor produces FAMILIAR_LOCATION.
6. A current point outside persisted anchors produces UNFAMILIAR_LOCATION.
7. Caregiver confirmation remains false unless explicitly configured.
8. Existing safe-area behavior remains unchanged.
9. Existing decision precedence remains unchanged.
10. Frozen risk prediction artifacts remain byte-identical.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import tempfile

from ml.src.decision import DecisionConfig, interpret_decision
from ml.src.geofence import (
    SafeAreaBoundary,
    SafeAreaState,
    evaluate_geospatial_context,
)
from ml.src.inference import RiskInferenceEngine
from ml.src.manual_inference import (
    build_personalized_baseline_from_history,
    run_manual_inference,
)


def _generate_synthetic_plt_content(
    start_dt: datetime,
    n_points: int = 240,
    speed_mps: float = 1.4,
    base_lat: float = 39.98470,
    base_lon: float = 116.31840,
) -> str:
    """Generate synthetically valid GeoLife .plt format content."""
    lines = [
        "Geolife trajectory",
        "WGS 84",
        "Altitude is in Feet",
        "Reserved 3",
        "0,2,255,My Track,0,0,2,8421376",
        "0",
    ]
    lat = base_lat
    lon = base_lon
    curr_t = start_dt

    for _ in range(n_points):
        lat += 0.000005 * (speed_mps / 1.4)
        lon += 0.000005 * (speed_mps / 1.4)
        date_str = curr_t.strftime("%Y-%m-%d")
        time_str = curr_t.strftime("%H:%M:%S")
        curr_t += timedelta(seconds=1)
        lines.append(f"{lat:.6f},{lon:.6f},0,100.0,40000.0000,{date_str},{time_str}")

    return "\n".join(lines)


def _generate_synthetic_plt_journey(
    start_dt: datetime,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    n_points: int = 240,
) -> str:
    """Generate valid GeoLife .plt content connecting start and end coordinates."""
    lines = [
        "Geolife trajectory",
        "WGS 84",
        "Altitude is in Feet",
        "Reserved 3",
        "0,2,255,My Track,0,0,2,8421376",
        "0",
    ]
    curr_t = start_dt
    for i in range(n_points):
        alpha = i / max(n_points - 1, 1)
        lat = start_lat + alpha * (end_lat - start_lat)
        lon = start_lon + alpha * (end_lon - start_lon)
        date_str = curr_t.strftime("%Y-%m-%d")
        time_str = curr_t.strftime("%H:%M:%S")
        curr_t += timedelta(seconds=1)
        lines.append(f"{lat:.6f},{lon:.6f},0,100.0,40000.0000,{date_str},{time_str}")

    return "\n".join(lines)


# 1. History with valid anchor evidence produces persisted anchor_clusters
def test_history_with_valid_anchor_evidence_produces_persisted_anchors():
    """Verify that historical trajectories sharing consistent endpoints discover and persist anchor clusters."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_valid_anchors_001"
    engine.delete_user_profile(test_user_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_ref = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        hist_dir = Path(tmp_dir) / "history"
        hist_dir.mkdir()

        # Generate 8 historical commuter trips between Home and Work (800m apart)
        home_lat, home_lon = 39.98470, 116.31840
        work_lat, work_lon = 39.99200, 116.32500
        for i in range(8):
            trip_t = t_ref + timedelta(days=i // 2, hours=8 if i % 2 == 0 else 17)
            p_file = hist_dir / f"commute_trip_{i:02d}.plt"
            if i % 2 == 0:
                content = _generate_synthetic_plt_journey(trip_t, home_lat, home_lon, work_lat, work_lon)
            else:
                content = _generate_synthetic_plt_journey(trip_t, work_lat, work_lon, home_lat, home_lon)
            p_file.write_text(content)

        # Target trajectory starting later near Home
        target_t = t_ref + timedelta(days=10)
        target_path = Path(tmp_dir) / "target.plt"
        target_path.write_text(_generate_synthetic_plt_journey(
            target_t, home_lat, home_lon, work_lat, work_lon, n_points=120
        ))

        summary = run_manual_inference(
            input_file=str(target_path),
            history_path=str(hist_dir),
            horizon="120",
            user_id=test_user_id,
            output_dir=os.path.join(tmp_dir, "out"),
        )

        assert summary["profile_mode"] == "PERSONALIZED"
        assert summary["trip_count"] == 8
        assert summary["persisted_anchor_clusters_count"] >= 1

        # Check persisted profile on disk
        prof = engine.load_user_profile(test_user_id)
        assert prof is not None
        assert "anchor_clusters" in prof
        assert len(prof["anchor_clusters"]) >= 1

        # Check primary anchor properties
        primary_anchor = sorted(prof["anchor_clusters"], key=lambda a: a["observation_count"], reverse=True)[0]
        assert "anchor_id" in primary_anchor
        assert "center_latitude" in primary_anchor
        assert "center_longitude" in primary_anchor
        assert "radius_m" in primary_anchor
        assert primary_anchor["observation_count"] >= 3

    engine.delete_user_profile(test_user_id)


# 2. History with insufficient anchor evidence does NOT fabricate anchors
def test_history_insufficient_anchor_evidence_no_fabrication():
    """Verify that dispersed endpoints do not fabricate anchors merely to make familiarity work."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_dispersed_002"
    engine.delete_user_profile(test_user_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_ref = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        hist_dir = Path(tmp_dir) / "history"
        hist_dir.mkdir()

        # Generate 8 trips with endpoints widely scattered across different cities/coordinates (> 20 km apart)
        for i in range(8):
            trip_t = t_ref + timedelta(days=i)
            p_file = hist_dir / f"scattered_trip_{i:02d}.plt"
            scattered_lat = 39.0 + i * 0.5  # 55 km apart
            scattered_lon = 116.0 + i * 0.5
            p_file.write_text(_generate_synthetic_plt_content(
                trip_t, n_points=240, speed_mps=1.0, base_lat=scattered_lat, base_lon=scattered_lon
            ))

        target_t = t_ref + timedelta(days=10)
        target_path = Path(tmp_dir) / "target.plt"
        target_path.write_text(_generate_synthetic_plt_content(
            target_t, n_points=120, speed_mps=1.0, base_lat=39.0, base_lon=116.0
        ))

        summary = run_manual_inference(
            input_file=str(target_path),
            history_path=str(hist_dir),
            horizon="120",
            user_id=test_user_id,
            output_dir=os.path.join(tmp_dir, "out"),
        )

        assert summary["profile_mode"] == "PERSONALIZED"
        assert summary["trip_count"] == 8
        assert summary["persisted_anchor_clusters_count"] == 0
        # When user has trips but 0 anchors formed, familiarity must be INSUFFICIENT_EVIDENCE
        assert summary["familiarity_state"] == "INSUFFICIENT_EVIDENCE"

        prof = engine.load_user_profile(test_user_id)
        assert prof is not None
        assert "anchor_clusters" in prof
        assert len(prof["anchor_clusters"]) == 0

    engine.delete_user_profile(test_user_id)


# 3. Current trajectory is excluded from anchor construction
def test_current_trajectory_excluded_from_anchor_construction():
    """Verify that current trajectory endpoints do NOT contribute to DBSCAN anchor clustering."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_exclusion_003"
    engine.delete_user_profile(test_user_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_ref = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        hist_dir = Path(tmp_dir) / "history"
        hist_dir.mkdir()

        # In DBSCAN with min_samples=3, 2 endpoints at a cluster location cannot form an anchor.
        # But if the current trajectory (with 2 endpoints at that same location) leaked,
        # total would be 4 endpoints >= 3, forming a cluster.
        special_lat, special_lon = 40.12345, 116.54321

        # 1 trip at special location (produces 2 endpoints: start and end)
        p_special = hist_dir / "hist_special_trip.plt"
        p_special.write_text(_generate_synthetic_plt_content(
            t_ref, n_points=240, speed_mps=1.0, base_lat=special_lat, base_lon=special_lon
        ))

        # 6 other trips at a completely different, unrelated location
        for i in range(1, 7):
            trip_t = t_ref + timedelta(days=i)
            p_file = hist_dir / f"other_trip_{i:02d}.plt"
            p_file.write_text(_generate_synthetic_plt_content(
                trip_t, n_points=240, speed_mps=1.0, base_lat=30.0 + i, base_lon=100.0 + i
            ))

        # Current trajectory also at the special location!
        target_t = t_ref + timedelta(days=10)
        target_path = Path(tmp_dir) / "current_target.plt"
        target_path.write_text(_generate_synthetic_plt_content(
            target_t, n_points=120, speed_mps=1.0, base_lat=special_lat, base_lon=special_lon
        ))

        # Build baseline with strict isolation
        res = build_personalized_baseline_from_history(
            history_path=hist_dir,
            cfg=engine.risk_cfg,
            feat_cfg=engine.feature_cfg,
            min_trips=7,
            current_trajectory_file=target_path,
            as_of_time=target_t,
            user_id=test_user_id,
        )

        assert res.mode == "PERSONALIZED"
        assert res.trip_count == 7
        # Since current trajectory was excluded, special location only had 2 endpoints < min_samples(3),
        # so NO anchor could be formed at special_lat, special_lon!
        assert len(res.anchor_clusters) == 0

    engine.delete_user_profile(test_user_id)


# 4. Stored profile can be reloaded with anchor_clusters intact
def test_stored_profile_reloads_with_anchor_clusters_intact():
    """Verify that a saved profile with anchor_clusters reloads completely intact."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_reload_004"
    engine.delete_user_profile(test_user_id)

    sample_clusters = [
        {
            "user_id": test_user_id,
            "anchor_id": f"{test_user_id}_anchor_0",
            "center_latitude": 39.98470,
            "center_longitude": 116.31840,
            "radius_m": 120.5,
            "observation_count": 9,
            "first_seen": "2026-01-01T08:00:00Z",
            "last_seen": "2026-01-09T08:00:00Z",
            "status": "CONFIRMED",
            "typical_hours": [8, 9, 17, 18],
            "raw_endpoint_count": 18,
        }
    ]

    prof_data = {
        "user_id": test_user_id,
        "profile_mode": "PERSONALIZED",
        "cold_start_status": "ML_DRIVEN",
        "trip_count": 9,
        "baseline_distribution": engine.population_baseline,
        "anchor_clusters": sample_clusters,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    engine.save_user_profile(test_user_id, prof_data)

    # Instantiate fresh engine instance to avoid cache
    fresh_engine = RiskInferenceEngine()
    reloaded_prof = fresh_engine.load_user_profile(test_user_id)

    assert reloaded_prof is not None
    assert "anchor_clusters" in reloaded_prof
    assert len(reloaded_prof["anchor_clusters"]) == 1
    reloaded_anchor = reloaded_prof["anchor_clusters"][0]
    assert reloaded_anchor["anchor_id"] == f"{test_user_id}_anchor_0"
    assert reloaded_anchor["center_latitude"] == 39.98470
    assert reloaded_anchor["center_longitude"] == 116.31840
    assert reloaded_anchor["radius_m"] == 120.5
    assert reloaded_anchor["observation_count"] == 9
    assert reloaded_anchor["status"] == "CONFIRMED"

    fresh_engine.delete_user_profile(test_user_id)


# 5. A current point near a persisted learned anchor produces: FAMILIAR_LOCATION
def test_current_point_near_persisted_anchor_produces_familiar_location():
    """Verify that coordinates within an anchor catchment radius produce FAMILIAR_LOCATION."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_near_anchor_005"
    engine.delete_user_profile(test_user_id)

    anchor_lat, anchor_lon = 39.98470, 116.31840
    prof_data = {
        "user_id": test_user_id,
        "profile_mode": "PERSONALIZED",
        "trip_count": 10,
        "anchor_clusters": [
            {
                "user_id": test_user_id,
                "anchor_id": f"{test_user_id}_home",
                "center_latitude": anchor_lat,
                "center_longitude": anchor_lon,
                "radius_m": 100.0,
                "observation_count": 8,
                "first_seen": "2026-01-01T08:00:00Z",
                "last_seen": "2026-01-10T08:00:00Z",
                "status": "CONFIRMED",
            }
        ],
    }
    engine.save_user_profile(test_user_id, prof_data)

    # Point ~10 meters away from anchor center
    eval_lat = anchor_lat + 0.00008
    eval_lon = anchor_lon + 0.00008
    geo_ctx = engine.evaluate_geospatial_context(
        lat=eval_lat,
        lon=eval_lon,
        user_id=test_user_id,
    )

    assert geo_ctx["familiarity_state"] == "FAMILIAR_LOCATION"
    assert geo_ctx["nearest_anchor_id"] == f"{test_user_id}_home"
    assert geo_ctx["distance_to_nearest_anchor_m"] < 30.0
    assert geo_ctx["nearest_anchor_status"] == "CONFIRMED"
    assert geo_ctx["is_caregiver_confirmed"] is False

    engine.delete_user_profile(test_user_id)


# 6. A current point outside persisted anchors produces: UNFAMILIAR_LOCATION
def test_current_point_outside_persisted_anchors_produces_unfamiliar_location():
    """Verify that coordinates outside all anchor radii produce UNFAMILIAR_LOCATION."""
    engine = RiskInferenceEngine()
    test_user_id = "test_user_far_anchor_006"
    engine.delete_user_profile(test_user_id)

    anchor_lat, anchor_lon = 39.98470, 116.31840
    prof_data = {
        "user_id": test_user_id,
        "profile_mode": "PERSONALIZED",
        "trip_count": 10,
        "anchor_clusters": [
            {
                "user_id": test_user_id,
                "anchor_id": f"{test_user_id}_home",
                "center_latitude": anchor_lat,
                "center_longitude": anchor_lon,
                "radius_m": 50.0,
                "observation_count": 8,
                "first_seen": "2026-01-01T08:00:00Z",
                "last_seen": "2026-01-10T08:00:00Z",
                "status": "CONFIRMED",
            }
        ],
    }
    engine.save_user_profile(test_user_id, prof_data)

    # Point ~2 km away from anchor center
    eval_lat = anchor_lat + 0.02
    eval_lon = anchor_lon + 0.02
    geo_ctx = engine.evaluate_geospatial_context(
        lat=eval_lat,
        lon=eval_lon,
        user_id=test_user_id,
    )

    assert geo_ctx["familiarity_state"] == "UNFAMILIAR_LOCATION"
    assert geo_ctx["nearest_anchor_id"] == f"{test_user_id}_home"
    assert geo_ctx["distance_to_nearest_anchor_m"] > 500.0
    assert geo_ctx["is_caregiver_confirmed"] is False

    engine.delete_user_profile(test_user_id)


# 7. Caregiver confirmation remains false unless explicitly configured
def test_caregiver_confirmation_remains_false_unless_explicitly_configured():
    """Verify algorithmic anchors never claim caregiver confirmation unless explicitly marked."""
    anchor_lat, anchor_lon = 39.98470, 116.31840

    # Algorithmic CONFIRMED anchor
    prof_algo = {
        "trip_count": 10,
        "anchor_clusters": [
            {
                "anchor_id": "algo_anchor",
                "center_latitude": anchor_lat,
                "center_longitude": anchor_lon,
                "radius_m": 100.0,
                "status": "CONFIRMED",
            }
        ],
    }
    ctx_algo = evaluate_geospatial_context(anchor_lat, anchor_lon, user_profile=prof_algo)
    assert ctx_algo.is_caregiver_confirmed is False
    assert ctx_algo.nearest_anchor_status == "CONFIRMED"

    # Explicitly caregiver-confirmed anchor
    prof_caregiver = {
        "trip_count": 10,
        "anchor_clusters": [
            {
                "anchor_id": "caregiver_home",
                "center_latitude": anchor_lat,
                "center_longitude": anchor_lon,
                "radius_m": 100.0,
                "status": "CAREGIVER_CONFIRMED",
            }
        ],
    }
    ctx_caregiver = evaluate_geospatial_context(anchor_lat, anchor_lon, user_profile=prof_caregiver)
    assert ctx_caregiver.is_caregiver_confirmed is True
    assert ctx_caregiver.nearest_anchor_status == "CAREGIVER_CONFIRMED"


# 8. Existing safe-area behavior remains unchanged
def test_existing_safe_area_behavior_unchanged():
    """Verify safe area boundary evaluation behaves exactly as designed."""
    boundary = SafeAreaBoundary.from_spec({
        "type": "circle",
        "center": [39.98470, 116.31840],
        "radius_m": 200.0,
        "name": "HomeSafeZone",
    })

    # Point inside circle
    res_inside = evaluate_geospatial_context(39.98475, 116.31845, safe_area=boundary)
    assert res_inside.safe_area_state == SafeAreaState.INSIDE_SAFE_AREA
    assert res_inside.is_safe_area_available is True
    assert res_inside.matched_safe_area_name == "HomeSafeZone"

    # Point outside circle
    res_outside = evaluate_geospatial_context(40.05000, 116.40000, safe_area=boundary)
    assert res_outside.safe_area_state == SafeAreaState.OUTSIDE_SAFE_AREA
    assert res_outside.is_safe_area_available is True

    # No safe area configured
    res_none = evaluate_geospatial_context(39.98470, 116.31840, safe_area=None)
    assert res_none.safe_area_state == SafeAreaState.SAFE_AREA_UNAVAILABLE
    assert res_none.is_safe_area_available is False


# 9. Existing decision precedence remains unchanged
def test_existing_decision_precedence_unchanged():
    """Verify deterministic decision rules preserve exact risk priorities and non-clinical neutrality."""
    cfg = DecisionConfig()

    # Inside safe area + familiar + Quiescent -> SAFE / NORMAL
    d1 = interpret_decision({
        "safe_area_state": "INSIDE_SAFE_AREA",
        "familiarity_state": "FAMILIAR_LOCATION",
        "risk_tier": "QUIESCENT",
        "risk_score": 5.0,
    }, cfg)
    assert d1.decision_state == "SAFE / NORMAL"

    # Outside safe area alone + Quiescent -> OUTSIDE_SAFE_AREA (not SUSPICIOUS or CRITICAL)
    d2 = interpret_decision({
        "safe_area_state": "OUTSIDE_SAFE_AREA",
        "familiarity_state": "FAMILIAR_LOCATION",
        "risk_tier": "QUIESCENT",
        "risk_score": 10.0,
    }, cfg)
    assert d2.decision_state == "OUTSIDE_SAFE_AREA"

    # Familiar location alone does NOT mask CRITICAL behavioral anomaly
    d3 = interpret_decision({
        "safe_area_state": "INSIDE_SAFE_AREA",
        "familiarity_state": "FAMILIAR_LOCATION",
        "risk_tier": "CRITICAL",
        "risk_score": 85.0,
    }, cfg)
    assert d3.decision_state == "CRITICAL"

    # Unfamiliar location alone does NOT imply danger
    d4 = interpret_decision({
        "safe_area_state": "SAFE_AREA_UNAVAILABLE",
        "familiarity_state": "UNFAMILIAR_LOCATION",
        "risk_tier": "QUIESCENT",
        "risk_score": 12.0,
    }, cfg)
    assert d4.decision_state == "UNFAMILIAR_MOVEMENT"


# 10. Frozen risk prediction artifacts remain byte-identical
def test_frozen_risk_prediction_artifacts_byte_identical():
    """Verify canonical benchmark evaluation artifacts remain 100% byte-identical."""
    canonical_files = {
        "ml/data/processed/risk_predictions.parquet": (
            "d716f7a268a7e96c874fb89c0e3311ebb343fb4c8fa7557f5eaa26c98b4a204f"
        ),
        "predictive_risk_report.json": (
            "c7b57bdff9c126c382f5e535b376df101d432daa0da2a3c064c424f95bafced1"
        ),
        "predictive_risk_report.md": (
            "181c2b108b1f4f8fc044e63f4cdd70e0447726925bf120da6818c9db945df0da"
        ),
    }

    for rel_path, expected_hash in canonical_files.items():
        p = Path(rel_path)
        assert p.exists(), f"Canonical artifact missing: {rel_path}"
        computed_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert computed_hash == expected_hash, (
            f"Artifact {rel_path} hash altered! Expected {expected_hash}, got {computed_hash}"
        )

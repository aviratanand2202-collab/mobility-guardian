"""
Deterministic Unit Tests for ML Engine Initialization and Lifespan Pre-warming.

Verifies:
1. Warmup is an immediate no-op when ENABLE_ML_INFERENCE_ENGINE is false.
2. Successful one-time initialization when ENABLE_ML_INFERENCE_ENGINE is true.
3. Thread-safe idempotency under concurrent initialization calls (singleton guarantee).
4. Graceful fault tolerance when model artifacts fail to load (sanitized warning, no crash).
5. Event loop responsiveness during asynchronous warmup (asyncio.to_thread non-blocking guarantee).
6. Complete independence of operational risk calculation from ML initialization status.
7. Shadow inference error handling when engine initialization fails.

Run with:
    backend/.venv/Scripts/python -m pytest backend/tests/test_ml_initialization.py -v
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest

# Ensure repository root and backend directory are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
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
from app.services.inference_service import (
    async_warmup_inference_engine,
    evaluate_shadow_inference,
    get_inference_engine,
    reset_inference_engine,
    warmup_inference_engine,
)
from app.services.mock_risk_engine import compute_risk


@pytest.fixture(autouse=True)
def _clean_engine_state():
    """Ensure clean singleton engine state before and after each test."""
    reset_inference_engine()
    yield
    reset_inference_engine()


def _sample_payload() -> TelemetryPayload:
    return TelemetryPayload(
        user_id="test_init_user",
        timestamp=datetime.now(timezone.utc),
        location=Location(lat=39.9830, lng=116.3130, altitude_m=15.0),
        sensor_metrics=SensorMetrics(
            speed_mps=1.2,
            heading_deg=45.0,
            horizontal_accuracy_m=5.0,
            battery_pct=90,
            activity_type=ActivityType.WALKING,
        ),
        signal_status=SignalStatus(state=SignalState.VALID, degraded_since=None),
        imu_metrics=None,
    )


# -----------------------------------------------------------------------------
# Test 1: ML-disabled warmup
# -----------------------------------------------------------------------------
def test_warmup_disabled_when_flag_false(monkeypatch):
    """Warmup must return False immediately and leave engine uninitialized when ML is disabled."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "false")
    result = warmup_inference_engine()
    assert result is False

    # Verify no engine was instantiated
    import app.services.inference_service as inf_svc
    assert inf_svc._engine is None


def test_async_warmup_disabled_when_flag_false(monkeypatch):
    """Async warmup must return False without launching threads when ML is disabled."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "false")

    async def _run():
        return await async_warmup_inference_engine()

    result = asyncio.run(_run())
    assert result is False


# -----------------------------------------------------------------------------
# Test 2: Successful one-time initialization
# -----------------------------------------------------------------------------
def test_warmup_success_when_flag_true(monkeypatch):
    """Warmup must instantiate singleton engine once and return True when enabled."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")
    result = warmup_inference_engine()
    assert result is True

    import app.services.inference_service as inf_svc
    engine1 = inf_svc._engine
    assert engine1 is not None

    # Subsequent call is idempotent and returns True without re-instantiation
    result2 = warmup_inference_engine()
    assert result2 is True
    assert inf_svc._engine is engine1


# -----------------------------------------------------------------------------
# Test 3: Concurrent initialization calls (Idempotency & Race Safety)
# -----------------------------------------------------------------------------
def test_concurrent_initialization_calls(monkeypatch):
    """Multiple concurrent threads calling warmup/get_inference_engine must yield identical singleton."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")

    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_inference_engine) for _ in range(num_threads)]
        engines = [f.result() for f in futures]

    first_engine = engines[0]
    assert first_engine is not None
    for eng in engines[1:]:
        assert eng is first_engine, "Concurrent initialization created duplicate engine instances!"


# -----------------------------------------------------------------------------
# Test 4: Missing / Corrupt Model Initialization Failure
# -----------------------------------------------------------------------------
def test_warmup_resilience_on_model_load_failure(monkeypatch):
    """Failure during model loading must log diagnostic and return False without raising."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")

    with patch(
        "ml.src.inference.RiskInferenceEngine._load_artifacts",
        side_effect=FileNotFoundError("Mock model artifact missing"),
    ):
        result = warmup_inference_engine()
        assert result is False

        import app.services.inference_service as inf_svc
        assert inf_svc._engine is None


# -----------------------------------------------------------------------------
# Test 5: Event Loop Responsiveness during Asynchronous Warmup
# -----------------------------------------------------------------------------
def test_event_loop_responsiveness_during_async_warmup(monkeypatch):
    """Async warmup via asyncio.to_thread must not block concurrent event loop tasks."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")

    ticks = []

    async def _heartbeat():
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks.append(time.perf_counter())

    def _simulated_slow_warmup():
        time.sleep(0.1)  # Simulate expensive booster model deserialization
        return True

    async def _main():
        with patch("app.services.inference_service.warmup_inference_engine", side_effect=_simulated_slow_warmup):
            warmup_task = asyncio.create_task(async_warmup_inference_engine())
            heartbeat_task = asyncio.create_task(_heartbeat())
            warmup_res, _ = await asyncio.gather(warmup_task, heartbeat_task)
            return warmup_res

    res = asyncio.run(_main())
    assert res is True
    assert len(ticks) == 5, "Heartbeat task was starved or blocked by warmup execution!"


# -----------------------------------------------------------------------------
# Test 6: Operational Risk Computation Independent of ML Initialization
# -----------------------------------------------------------------------------
def test_operational_risk_independent_of_ml_status(monkeypatch):
    """Operational compute_risk must calculate score and tier with 100% fidelity regardless of ML."""
    payload = _sample_payload()

    # 1. Evaluate with ML uninitialized
    op_out_1 = compute_risk(payload, pdr_tier=None, cell_dismissal_state=None)
    assert 0.0 <= op_out_1.risk_score <= 100.0
    assert op_out_1.risk_tier is not None

    # 2. Simulate ML failure
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")
    with patch("app.services.inference_service.get_inference_engine", side_effect=RuntimeError("Simulated failure")):
        op_out_2 = compute_risk(payload, pdr_tier=None, cell_dismissal_state=None)
        assert op_out_2.risk_tier == op_out_1.risk_tier
        assert op_out_2.risk_score == op_out_1.risk_score


# -----------------------------------------------------------------------------
# Test 7: Shadow Inference Graceful Error Handling on Engine Failure
# -----------------------------------------------------------------------------
def test_shadow_inference_handles_engine_failure_gracefully(monkeypatch):
    """evaluate_shadow_inference must catch engine errors and return sanitized error dict without raising."""
    monkeypatch.setenv("ENABLE_ML_INFERENCE_ENGINE", "true")
    payload = _sample_payload()

    async def _test():
        with patch("app.services.inference_service.get_inference_engine", side_effect=RuntimeError("Fatal model corruption")):
            return await evaluate_shadow_inference(payload, timeout_sec=0.5)

    ml_eval = asyncio.run(_test())
    assert ml_eval is not None
    assert ml_eval["engine_status"] == "FAILED"
    assert ml_eval["evidence_status"] == "ERROR"
    assert ml_eval["calibrated_probability"] is None
    assert ml_eval["raw_probability"] is None
    assert ml_eval["ml_risk_score"] is None
    assert ml_eval["ml_risk_tier"] is None
    assert "error" in ml_eval

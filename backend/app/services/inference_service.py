"""Inference service bridging FastAPI backend to the ML RiskInferenceEngine in Shadow Mode.

Maintains singleton inference engine instance, evaluates real-time telemetry
in observational shadow mode, and preserves existing operational risk scoring and alert lifecycles.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import threading
from typing import TYPE_CHECKING, Any, Optional

from app.config import is_ml_inference_enabled
from app.models import RiskScoreOutput, TelemetryPayload
from app.websocket.risk_stream import push_risk_update

# Ensure repository root is on sys.path so ml.src is importable regardless of launch directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from ml.src.inference import InferenceConfig, RiskInferenceEngine

logger = logging.getLogger(__name__)

_engine: Optional[RiskInferenceEngine] = None
_engine_init_lock = threading.Lock()
_user_locks: dict[str, asyncio.Lock] = {}
_user_thread_locks: dict[str, threading.Lock] = {}
_locks_guard = asyncio.Lock()
_thread_locks_guard = threading.Lock()


def get_inference_engine() -> RiskInferenceEngine:
    """Lazy initialize singleton RiskInferenceEngine with paths anchored to repo root."""
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                from ml.src.inference import InferenceConfig, RiskInferenceEngine

                cfg = InferenceConfig(
                    model_dir=str(_REPO_ROOT / "ml" / "models" / "xgboost"),
                    profiles_dir=str(_REPO_ROOT / "ml" / "models" / "profiles"),
                )
                _engine = RiskInferenceEngine(config=cfg)
    return _engine


def reset_inference_engine() -> None:
    """Reset the singleton engine, per-user locks, and behavioral state (useful for testing and isolation)."""
    global _engine
    with _engine_init_lock:
        _engine = None
    _user_locks.clear()
    with _thread_locks_guard:
        _user_thread_locks.clear()
    try:
        from app.state_machines.behavioral_escalation import reset_behavioral_state
        reset_behavioral_state()
    except Exception:
        pass



def warmup_inference_engine() -> bool:
    """Pre-warm singleton RiskInferenceEngine during server startup.

    Guarantees:
    - No-op returning False when ENABLE_ML_INFERENCE_ENGINE is false.
    - Idempotent and thread-safe via _engine_init_lock.
    - Fault-tolerant: if model files are missing or deserialization fails, catches
      the exception, logs a sanitized diagnostic, and returns False.
    - Operational risk calculation and service availability remain independent of ML.
    """
    if not is_ml_inference_enabled():
        return False
    try:
        get_inference_engine()
        logger.info("Shadow ML inference engine successfully pre-warmed.")
        return True
    except Exception as exc:
        logger.warning(
            f"Shadow ML engine pre-warming failed during startup (will fall back gracefully): {exc}",
            exc_info=True,
        )
        return False


async def async_warmup_inference_engine() -> bool:
    """Asynchronously pre-warm the ML inference engine off the main event loop thread."""
    if not is_ml_inference_enabled():
        return False
    return await asyncio.to_thread(warmup_inference_engine)


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Retrieve or create an asyncio.Lock for the specified user to serialize stateful buffers."""
    if user_id not in _user_locks:
        async with _locks_guard:
            if user_id not in _user_locks:
                _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


def _get_user_thread_lock(user_id: str) -> threading.Lock:
    """Retrieve or create a threading.Lock for synchronous thread-level mutual exclusion."""
    if user_id not in _user_thread_locks:
        with _thread_locks_guard:
            if user_id not in _user_thread_locks:
                _user_thread_locks[user_id] = threading.Lock()
    return _user_thread_locks[user_id]


async def evaluate_shadow_inference(
    payload: TelemetryPayload,
    timeout_sec: float = 1.0,
) -> Optional[dict[str, Any]]:
    """Execute observational shadow-mode inference against the incoming telemetry payload.

    Guarantees:
    - Returns None immediately when ENABLE_ML_INFERENCE_ENGINE is disabled.
    - Evaluates strictly observational ML output without altering operational risk scores.
    - Serializes inference calls per-user to prevent buffer race conditions.
    - Protects against timeout concurrency: if worker thread continues past timeout,
      subsequent inferences for that user cannot overlap with the running thread.
    - Bounded execution with timeout; handles any engine failure gracefully without raising.
    - Ensures ml_risk_score, calibrated_probability, and raw_probability are null
      when evidence is insufficient or when an error/timeout occurs.
    - Redacts internal error messages/paths from the response, logging diagnostics server-side.
    """
    if not is_ml_inference_enabled():
        return None

    try:
        engine = get_inference_engine()
    except Exception as exc:
        logger.warning(
            f"Shadow ML inference engine initialization encountered error for user '{payload.user_id}': {exc}",
            exc_info=True,
        )
        return {
            "engine_status": "FAILED",
            "evidence_status": "ERROR",
            "calibrated_probability": None,
            "raw_probability": None,
            "ml_risk_score": None,
            "ml_risk_tier": None,
            "error": "ML inference execution encountered an unexpected error.",
        }

    loop = asyncio.get_running_loop()
    start_time = loop.time()
    user_lock = await _get_user_lock(payload.user_id)

    # 1. Bounded acquire of user lock to prevent worker queue pileup
    remaining_time = max(0.0, timeout_sec - (loop.time() - start_time))
    try:
        await asyncio.wait_for(user_lock.acquire(), timeout=remaining_time)
    except asyncio.TimeoutError:
        logger.warning(
            f"Shadow ML inference timed out acquiring per-user lock after {timeout_sec}s for '{payload.user_id}'. "
            "A previous inference is still in progress."
        )
        return {
            "engine_status": "TIMEOUT",
            "evidence_status": "ERROR",
            "calibrated_probability": None,
            "raw_probability": None,
            "ml_risk_score": None,
            "ml_risk_tier": None,
            "error": "ML inference execution exceeded time limit.",
        }

    # Lock is held. Worker must release it upon completion, even if coroutine times out.
    thread_dispatched = False
    try:

        # Prepare telemetry dictionary matching RiskInferenceEngine input schema
        payload_dict: dict[str, Any] = payload.model_dump(mode="json")
        if isinstance(payload.timestamp, datetime):
            payload_dict["timestamp"] = payload.timestamp.isoformat()
        if payload.signal_status.degraded_since:
            payload_dict["signal_status"]["degraded_since"] = (
                payload.signal_status.degraded_since.isoformat()
            )

        thread_lock = _get_user_thread_lock(payload.user_id)

        def _worker() -> dict[str, Any]:
            try:
                with thread_lock:
                    return engine.predict_telemetry_reading(payload_dict)
            finally:
                loop.call_soon_threadsafe(user_lock.release)

        fut = loop.run_in_executor(None, _worker)
        thread_dispatched = True

    except Exception as exc:
        if not thread_dispatched:
            user_lock.release()
        logger.warning(
            f"Shadow ML inference preparation encountered error for user '{payload.user_id}': {exc}",
            exc_info=True,
        )
        return {
            "engine_status": "FAILED",
            "evidence_status": "ERROR",
            "calibrated_probability": None,
            "raw_probability": None,
            "ml_risk_score": None,
            "ml_risk_tier": None,
            "error": "ML inference execution encountered an unexpected error.",
        }

    # 2. Wait for worker future with remaining timeout budget, shielded to avoid cancelling thread
    remaining_time = max(0.0, timeout_sec - (loop.time() - start_time))
    try:
        raw_out: dict[str, Any] = await asyncio.wait_for(
            asyncio.shield(fut),
            timeout=remaining_time,
        )

        # Determine evidence status
        trigger_st = raw_out.get("trigger_state", {})
        is_accumulating = (
            trigger_st.get("status") == "ACCUMULATING_WINDOW"
            or trigger_st.get("calibrated_probability") is None
        )
        evidence_status = "INSUFFICIENT_EVIDENCE" if is_accumulating else "WINDOW_EVALUATED"

        if is_accumulating:
            cal_prob = None
            raw_prob = None
            ml_risk_score = None
            ml_risk_tier = None
        else:
            cal_prob = trigger_st.get("calibrated_probability")
            if cal_prob is not None:
                try:
                    cal_prob = float(cal_prob)
                except (ValueError, TypeError):
                    cal_prob = None

            raw_prob = trigger_st.get("raw_probability")
            if raw_prob is not None:
                try:
                    raw_prob = float(raw_prob)
                except (ValueError, TypeError):
                    raw_prob = None

            ml_risk_score = raw_out.get("risk_score")
            if ml_risk_score is not None:
                try:
                    ml_risk_score = float(ml_risk_score)
                except (ValueError, TypeError):
                    ml_risk_score = None

            ml_risk_tier = raw_out.get("risk_tier")

        kin_feats = raw_out.get("kinematic_features", {}) if isinstance(raw_out.get("kinematic_features"), dict) else {}
        metadata = raw_out.get("metadata", {}) if isinstance(raw_out.get("metadata"), dict) else {}
        behavioral_indicator = kin_feats.get("behavioral_indicator")
        behavioral_confidence = kin_feats.get("behavioral_confidence")
        window_id = metadata.get("window_id")
        safe_area_state = raw_out.get("safe_area_state") or raw_out.get("geospatial_context", {}).get("safe_area_state")

        return {
            "engine_status": "ACTIVE",
            "evidence_status": evidence_status,
            "calibrated_probability": cal_prob,
            "raw_probability": raw_prob,
            "ml_risk_score": ml_risk_score,
            "ml_risk_tier": ml_risk_tier,
            "decision_state": raw_out.get("decision_state"),
            "display_state": raw_out.get("display_state"),
            "reason": raw_out.get("human_readable_reason"),
            "trigger_state": trigger_st,
            "explainability": raw_out.get("explainability"),
            "location": raw_out.get("location"),
            "behavioral_indicator": behavioral_indicator,
            "behavioral_confidence": behavioral_confidence,
            "window_id": window_id,
            "safe_area_state": safe_area_state,
            "kinematic_features": kin_feats,
        }


    except asyncio.TimeoutError:
        logger.warning(
            f"Shadow ML inference execution timed out after {timeout_sec}s for user '{payload.user_id}'. "
            "Worker thread will finish in background without allowing overlapping inferences."
        )
        return {
            "engine_status": "TIMEOUT",
            "evidence_status": "ERROR",
            "calibrated_probability": None,
            "raw_probability": None,
            "ml_risk_score": None,
            "ml_risk_tier": None,
            "error": "ML inference execution exceeded time limit.",
        }
    except Exception as exc:
        logger.warning(
            f"Shadow ML inference encountered error for user '{payload.user_id}': {exc}",
            exc_info=True,
        )
        return {
            "engine_status": "FAILED",
            "evidence_status": "ERROR",
            "calibrated_probability": None,
            "raw_probability": None,
            "ml_risk_score": None,
            "ml_risk_tier": None,
            "error": "ML inference execution encountered an unexpected error.",
        }


async def process_telemetry(payload: TelemetryPayload) -> RiskScoreOutput:
    """Process incoming TelemetryPayload and return schema-valid RiskScoreOutput.

    Retained for backward compatibility with existing tests and callers.
    """
    engine = get_inference_engine()

    payload_dict = payload.model_dump(mode="json")
    if isinstance(payload.timestamp, datetime):
        payload_dict["timestamp"] = payload.timestamp.isoformat()
    if payload.signal_status.degraded_since:
        payload_dict["signal_status"]["degraded_since"] = (
            payload.signal_status.degraded_since.isoformat()
        )

    raw_output = engine.predict_telemetry_reading(payload_dict)

    risk_output = RiskScoreOutput(
        user_id=raw_output["user_id"],
        timestamp=raw_output["timestamp"],
        risk_tier=raw_output["risk_tier"],
        risk_score=raw_output["risk_score"],
        polling_tier=raw_output["polling_tier"],
        predicted_lead_time_sec=raw_output.get("predicted_lead_time_sec"),
        battery_override_active=raw_output.get("battery_override_active", False),
        location=raw_output.get("location"),
    )

    try:
        await push_risk_update(payload.user_id, risk_output.model_dump(mode="json"))
    except Exception as exc:
        logger.debug(f"WebSocket broadcast skipped or failed: {exc}")

    return risk_output

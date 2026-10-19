"""Application configuration settings for Mobility Guardian backend."""
from __future__ import annotations

import os

# Centralized configuration flag for ML inference engine integration (disabled by default)
ENABLE_ML_INFERENCE_ENGINE: bool = (
    os.getenv("ENABLE_ML_INFERENCE_ENGINE", "false").lower() in ("true", "1", "yes")
)


def is_ml_inference_enabled() -> bool:
    """Return whether the shadow ML inference engine is enabled.

    Reads the ENABLE_ML_INFERENCE_ENGINE environment variable dynamically,
    falling back to module-level ENABLE_ML_INFERENCE_ENGINE.
    """
    env_val = os.getenv("ENABLE_ML_INFERENCE_ENGINE")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return ENABLE_ML_INFERENCE_ENGINE


# Centralized configuration flag for behavioral risk escalation bridge (disabled by default)
ENABLE_BEHAVIORAL_RISK_ESCALATION: bool = (
    os.getenv("ENABLE_BEHAVIORAL_RISK_ESCALATION", "false").lower() in ("true", "1", "yes")
)


def is_behavioral_risk_escalation_enabled() -> bool:
    """Return whether behavioral risk escalation to operational decisions is enabled.

    Reads the ENABLE_BEHAVIORAL_RISK_ESCALATION environment variable dynamically,
    falling back to module-level ENABLE_BEHAVIORAL_RISK_ESCALATION.
    Escalation requires ML inference to be enabled as well.
    """
    env_val = os.getenv("ENABLE_BEHAVIORAL_RISK_ESCALATION")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return ENABLE_BEHAVIORAL_RISK_ESCALATION

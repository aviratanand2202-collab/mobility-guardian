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

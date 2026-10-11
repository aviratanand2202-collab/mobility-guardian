"""Deterministic ML Decision & Interpretation Layer (Chunk 3).

Synthesizes:
1. Safe-Area State (INSIDE_SAFE_AREA, OUTSIDE_SAFE_AREA, SAFE_AREA_UNAVAILABLE)
2. Familiarity State (FAMILIAR_LOCATION, UNFAMILIAR_LOCATION, INSUFFICIENT_EVIDENCE, NO_HISTORY)
3. Behavioral ML Risk Tier (QUIESCENT, NORMAL_TRANSIT, SUSPICIOUS, CRITICAL)

Methodological & Behavioral Guarantees:
- INSIDE safe area + normal behavior must remain SAFE / NORMAL.
- OUTSIDE safe area alone must NOT become SUSPICIOUS or CRITICAL.
- FAMILIAR location alone must NOT imply safety if behavioral risk is high.
- UNFAMILIAR location alone must NOT imply danger.
- Existing behavioral ML risk remains the exclusive source of behavioral anomaly severity.
- Zero clinical, dementia, wandering, or ground-truth safety claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ==============================================================================
# CANONICAL CONTEXTUAL DECISION STATES
# ==============================================================================


class DecisionState(str, Enum):
    """Categorical structured decision states for downstream application consumption."""

    SAFE_NORMAL = "SAFE / NORMAL"
    OUTSIDE_SAFE_AREA = "OUTSIDE_SAFE_AREA"
    FAMILIAR_MOVEMENT = "FAMILIAR_MOVEMENT"
    UNFAMILIAR_MOVEMENT = "UNFAMILIAR_MOVEMENT"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


# ==============================================================================
# CENTRALIZED CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class DecisionConfig:
    """Configurable decision rules and disclaimers for ML output interpretation."""

    # Rule precedence flags
    critical_takes_precedence: bool = True
    suspicious_takes_precedence: bool = True
    inside_safe_area_guarantees_safe_normal: bool = True

    # Standardized non-clinical disclaimer
    disclaimer: str = (
        "Associational kinematic and geospatial decision output. "
        "Reflects mathematical outlier probabilities and configured geographic boundaries. "
        "Does NOT constitute clinical diagnosis, dementia evaluation, wandering detection, "
        "or ground-truth safety guarantees."
    )


# ==============================================================================
# STRUCTURED DECISION OUTPUT DATA STRUCTURE
# ==============================================================================


@dataclass
class DecisionOutput:
    """Consolidated human-readable decision output conforming to downstream contracts."""

    # 1. Primary Contextual Decision State
    decision_state: str  # One of DecisionState values
    display_state: str  # User-facing title
    headline: str  # Short human-readable headline
    reason: str  # Detailed non-clinical factual reason
    severity_level: str  # "NORMAL", "ADVISORY", "WARNING", "ALERT"

    # 2. Contributing Dimensions
    profile_mode: str
    safe_area_state: str
    familiarity_state: str
    behavioral_tier: str
    behavioral_pattern: str

    # 3. Preserved Raw Research / Debug Metrics
    risk_score: float
    calibrated_probability: float
    raw_probability: float
    horizon_sec: int
    binary_alert: bool
    top_features: List[Dict[str, Any]]

    # 4. Contextual Spatial Details
    distance_to_safe_boundary_m: Optional[float]
    safe_area_name: Optional[str]
    nearest_anchor_id: Optional[str]
    distance_to_nearest_anchor_m: Optional[float]
    is_caregiver_confirmed_anchor: bool
    trip_count: int

    # 5. Methodological Notice
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision output to deterministic dictionary."""
        return asdict(self)


# ==============================================================================
# DETERMINISTIC DECISION RULES ENGINE
# ==============================================================================


def interpret_decision(
    risk_output: Dict[str, Any],
    config: DecisionConfig = DecisionConfig(),
) -> DecisionOutput:
    """Deterministically interpret combined ML risk, geospatial state, and familiarity.

    Evaluates exact rules:
    1. If Behavioral ML tier is CRITICAL:
       -> State: CRITICAL (behavioral severity dominates; location does not override).
    2. If Behavioral ML tier is SUSPICIOUS:
       -> State: SUSPICIOUS (behavioral anomaly elevated; location provides context).
    3. If Behavioral ML tier is Normal (QUIESCENT / NORMAL_TRANSIT):
       - If OUTSIDE_SAFE_AREA:
         -> State: OUTSIDE_SAFE_AREA (never promoted to SUSPICIOUS/CRITICAL solely by location).
       - If INSIDE_SAFE_AREA:
         -> State: SAFE / NORMAL (guaranteed SAFE/NORMAL when kinematics are normal).
       - If SAFE_AREA_UNAVAILABLE:
         - If FAMILIAR_LOCATION:
           -> State: FAMILIAR_MOVEMENT.
         - If UNFAMILIAR_LOCATION:
           -> State: UNFAMILIAR_MOVEMENT (never implies danger).
         - If INSUFFICIENT_EVIDENCE or NO_HISTORY:
           -> State: SAFE / NORMAL.
    """
    # 1. Extract dimensions
    behavioral_tier = str(risk_output.get("risk_tier", "QUIESCENT")).upper()
    geo_ctx = risk_output.get("geospatial_context", {})
    safe_area_state = str(
        risk_output.get("safe_area_state")
        or geo_ctx.get("safe_area_state")
        or "SAFE_AREA_UNAVAILABLE"
    ).upper()
    familiarity_state = str(
        risk_output.get("familiarity_state")
        or geo_ctx.get("familiarity_state")
        or "NO_HISTORY"
    ).upper()

    metadata = risk_output.get("metadata", {})
    profile_mode = str(metadata.get("profile_mode", "COLD_START"))
    trip_count = int(metadata.get("trip_count", 0))

    trigger_state = risk_output.get("trigger_state", {})
    horizon_sec = int(trigger_state.get("horizon_sec", 120))
    cal_prob = float(trigger_state.get("calibrated_probability", 0.0))
    raw_prob = float(trigger_state.get("raw_probability", 0.0))
    binary_alert = bool(trigger_state.get("binary_alert", False))
    risk_score = float(risk_output.get("risk_score", 0.0))

    kin_feats = risk_output.get("kinematic_features", {})
    behavioral_pattern = str(kin_feats.get("behavioral_indicator", "NORMAL"))

    explainability = risk_output.get("explainability", {})
    top_features = explainability.get("top_features", [])

    geo_ctx = risk_output.get("geospatial_context", {})
    dist_safe = geo_ctx.get("distance_to_safe_boundary_m")
    safe_name = geo_ctx.get("matched_safe_area_name")
    anchor_id = geo_ctx.get("nearest_anchor_id")
    dist_anchor = geo_ctx.get("distance_to_nearest_anchor_m")
    confirmed_anchor = bool(geo_ctx.get("is_caregiver_confirmed", False))

    # 2. Rule evaluation
    # RULE 1: CRITICAL behavioral risk takes precedence
    if config.critical_takes_precedence and behavioral_tier == "CRITICAL":
        decision_state = DecisionState.CRITICAL.value
        display_state = "CRITICAL KINEMATIC EXCURSION"
        severity_level = "ALERT"
        headline = "Critical Kinematic Excursion Detected"

        # Context-aware factual reason
        if safe_area_state == "INSIDE_SAFE_AREA":
            reason = (
                f"Severe kinematic deviation observed (score: {risk_score:.1f}/100) "
                f"despite position remaining inside safe area '{safe_name or 'Zone'}'. "
                f"Pattern indicates anomalous movement ({behavioral_pattern})."
            )
        elif safe_area_state == "OUTSIDE_SAFE_AREA":
            reason = (
                f"Severe kinematic deviation observed (score: {risk_score:.1f}/100) "
                f"while outside configured safe area '{safe_name or 'Zone'}' ({dist_safe}m away). "
                f"Pattern indicates anomalous movement ({behavioral_pattern})."
            )
        elif familiarity_state == "FAMILIAR_LOCATION":
            reason = (
                f"Severe kinematic deviation observed (score: {risk_score:.1f}/100) "
                f"within catchment of historical anchor '{anchor_id or 'Anchor'}'. "
                f"Familiar location does not override acute kinematic anomaly."
            )
        else:
            reason = (
                f"Severe kinematic deviation observed (score: {risk_score:.1f}/100, horizon: {horizon_sec}s). "
                f"Kinematic pattern indicates anomalous excursion ({behavioral_pattern})."
            )

    # RULE 2: SUSPICIOUS behavioral risk takes precedence
    elif config.suspicious_takes_precedence and behavioral_tier == "SUSPICIOUS":
        decision_state = DecisionState.SUSPICIOUS.value
        display_state = "SUSPICIOUS MOVEMENT PATTERN"
        severity_level = "WARNING"
        headline = "Elevated Kinematic Deviation Detected"

        if safe_area_state == "OUTSIDE_SAFE_AREA":
            reason = (
                f"Elevated kinematic score ({risk_score:.1f}/100) observed outside configured safe boundary "
                f"'{safe_name or 'Zone'}' ({dist_safe}m away). Pattern: {behavioral_pattern}."
            )
        elif familiarity_state == "UNFAMILIAR_LOCATION":
            reason = (
                f"Elevated kinematic score ({risk_score:.1f}/100) observed in unfamiliar territory "
                f"({dist_anchor}m from nearest anchor). Pattern: {behavioral_pattern}."
            )
        elif safe_area_state == "INSIDE_SAFE_AREA":
            reason = (
                f"Elevated kinematic score ({risk_score:.1f}/100) observed within safe area '{safe_name}'. "
                f"Pattern: {behavioral_pattern}."
            )
        else:
            reason = (
                f"Elevated kinematic score ({risk_score:.1f}/100) detected relative to baseline. "
                f"Pattern: {behavioral_pattern}."
            )

    # RULE 3: Normal behavioral kinematics (QUIESCENT / NORMAL_TRANSIT)
    else:
        # Sub-rule 3A: OUTSIDE safe area alone must NOT become SUSPICIOUS or CRITICAL
        if safe_area_state == "OUTSIDE_SAFE_AREA":
            decision_state = DecisionState.OUTSIDE_SAFE_AREA.value
            display_state = "OUTSIDE SAFE AREA (NORMAL KINEMATICS)"
            severity_level = "ADVISORY"
            headline = "Outside Configured Safe Area Boundary"

            if familiarity_state == "FAMILIAR_LOCATION":
                reason = (
                    f"Position is {dist_safe}m outside configured safe area '{safe_name}', but is near "
                    f"historical anchor '{anchor_id}'. Kinematic indicators remain within routine limits."
                )
            elif familiarity_state == "UNFAMILIAR_LOCATION":
                reason = (
                    f"Position is {dist_safe}m outside configured safe area '{safe_name}' in an unobserved area. "
                    f"Kinematic indicators remain routine; location alone does not denote danger."
                )
            else:
                reason = (
                    f"Position is {dist_safe}m outside configured safe area '{safe_name}'. "
                    f"Kinematic metrics (score: {risk_score:.1f}/100) indicate normal transit without excursion."
                )

        # Sub-rule 3B: INSIDE safe area + normal behavior must remain SAFE/NORMAL
        elif config.inside_safe_area_guarantees_safe_normal and safe_area_state == "INSIDE_SAFE_AREA":
            decision_state = DecisionState.SAFE_NORMAL.value
            display_state = "SAFE / NORMAL"
            severity_level = "NORMAL"
            headline = "Routine Movement Within Safe Area"

            if familiarity_state == "FAMILIAR_LOCATION":
                reason = (
                    f"Position is inside safe area '{safe_name}' and within familiar anchor catchment "
                    f"'{anchor_id}'. Kinematic movement is normal (score: {risk_score:.1f}/100)."
                )
            elif familiarity_state == "UNFAMILIAR_LOCATION":
                reason = (
                    f"Position is within safe area '{safe_name}', navigating an unobserved local sector. "
                    f"Kinematic indicators remain completely normal."
                )
            else:
                reason = (
                    f"Position is within safe area '{safe_name}' ({abs(dist_safe or 0)}m margin). "
                    f"Kinematic indicators are routine."
                )

        # Sub-rule 3C: Safe area is unavailable / unconfigured
        else:
            if familiarity_state == "FAMILIAR_LOCATION":
                decision_state = DecisionState.FAMILIAR_MOVEMENT.value
                display_state = "FAMILIAR MOVEMENT"
                severity_level = "NORMAL"
                headline = "Routine Movement Near Familiar Anchor"
                reason = (
                    f"Trajectory is traversing learned anchor '{anchor_id}' catchment ({dist_anchor}m away). "
                    f"Kinematics are routine (score: {risk_score:.1f}/100)."
                )
            elif familiarity_state == "UNFAMILIAR_LOCATION":
                # Unfamiliar location alone must NOT imply danger
                decision_state = DecisionState.UNFAMILIAR_MOVEMENT.value
                display_state = "UNFAMILIAR MOVEMENT (NORMAL KINEMATICS)"
                severity_level = "ADVISORY"
                headline = "Navigation in Unobserved Territory"
                reason = (
                    f"Location is {dist_anchor}m outside learned historical anchors. "
                    f"Unfamiliar location does not imply danger; kinematics remain fully normal."
                )
            else:
                # INSUFFICIENT_EVIDENCE or NO_HISTORY with normal kinematics
                decision_state = DecisionState.SAFE_NORMAL.value
                display_state = "SAFE / NORMAL"
                severity_level = "NORMAL"
                headline = "Routine Baseline Movement"
                reason = (
                    f"Kinematic indicators are within expected routine limits (score: {risk_score:.1f}/100). "
                    f"No anomalous trajectory excursion detected."
                )

    return DecisionOutput(
        decision_state=decision_state,
        display_state=display_state,
        headline=headline,
        reason=reason,
        severity_level=severity_level,
        profile_mode=profile_mode,
        safe_area_state=safe_area_state,
        familiarity_state=familiarity_state,
        behavioral_tier=behavioral_tier,
        behavioral_pattern=behavioral_pattern,
        risk_score=risk_score,
        calibrated_probability=cal_prob,
        raw_probability=raw_prob,
        horizon_sec=horizon_sec,
        binary_alert=binary_alert,
        top_features=top_features,
        distance_to_safe_boundary_m=dist_safe,
        safe_area_name=safe_name,
        nearest_anchor_id=anchor_id,
        distance_to_nearest_anchor_m=dist_anchor,
        is_caregiver_confirmed_anchor=confirmed_anchor,
        trip_count=trip_count,
        disclaimer=config.disclaimer,
    )

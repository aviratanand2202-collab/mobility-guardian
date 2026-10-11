"""Geospatial and Safe-Area Contextual Engine.

This module implements the orthogonal geospatial context and familiarity layer
for mobility analysis. It operates independently of the kinematic XGBoost risk model.

Methodological Guarantees:
1. Safe-Area State:
   - INSIDE_SAFE_AREA, OUTSIDE_SAFE_AREA, or SAFE_AREA_UNAVAILABLE.
   - Safe areas are externally supplied/configurable boundaries (caregiver geofences).
   - GeoLife does not contain ground-truth safe-zone labels.
   - Leaving a safe area does NOT automatically imply elevated kinematic risk.

2. Familiarity State:
   - FAMILIAR_LOCATION, UNFAMILIAR_LOCATION, INSUFFICIENT_EVIDENCE, or NO_HISTORY.
   - Leverages learned DBSCAN spatial anchor clusters from historical mobility profiles.
   - Algorithmic anchors are NOT caregiver-confirmed.
   - Unfamiliar locations do NOT inherently denote dangerous movement.

3. Complete Orthogonality:
   - This contextual layer NEVER modifies XGBoost features, training data, targets,
     decision thresholds, or benchmark artifacts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ml.src.profile import EARTH_RADIUS_METERS, haversine_distance

logger = logging.getLogger(__name__)


# ==============================================================================
# CANONICAL STATES & ENUMS
# ==============================================================================


class SafeAreaState(str, Enum):
    """Categorical evaluation of current location relative to configured safe areas."""

    INSIDE_SAFE_AREA = "INSIDE_SAFE_AREA"
    OUTSIDE_SAFE_AREA = "OUTSIDE_SAFE_AREA"
    SAFE_AREA_UNAVAILABLE = "SAFE_AREA_UNAVAILABLE"


class FamiliarityState(str, Enum):
    """Categorical evaluation of location familiarity relative to learned anchors."""

    FAMILIAR_LOCATION = "FAMILIAR_LOCATION"
    UNFAMILIAR_LOCATION = "UNFAMILIAR_LOCATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_HISTORY = "NO_HISTORY"


# ==============================================================================
# CENTRALIZED CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class GeospatialConfig:
    """Centralized configuration for geospatial context and spatial familiarity.

    Centralizes all thresholds, buffers, and disclaimers to prevent magic numbers.
    """

    # Earth radius in meters (aligned with profile.py)
    earth_radius_m: float = EARTH_RADIUS_METERS

    # Familiarity parameters (aligned with ProfileConfig / test_risk safeguards)
    familiarity_min_trips: int = 7
    """Minimum historical trips required before spatial familiarity can be evaluated.
    Matches cold_start_min_trips in ProfileConfig."""

    familiarity_anchor_expansion_factor: float = 1.0
    """Multiplier applied to anchor cluster radius (radius_m) for catchment evaluation."""

    familiarity_anchor_min_radius_m: float = 25.0
    """Floor for anchor radius in meters, reflecting standard GPS accuracy floor."""

    # Safe Area parameters
    safe_area_default_buffer_m: float = 0.0
    """Tolerance buffer in meters applied to safe area boundaries."""

    # Disclaimers and scientific policy notes
    disclaimers: Dict[str, str] = field(
        default_factory=lambda: {
            "safe_area": (
                "Safe area boundaries are externally configured geographic boundaries. "
                "The GeoLife dataset does not provide ground-truth safe-zone labels. "
                "Leaving a safe area does not inherently indicate danger or elevated risk."
            ),
            "familiarity": (
                "Familiarity is evaluated against algorithmic spatial anchors derived from DBSCAN "
                "endpoint clustering. Algorithmic anchors are NOT caregiver-confirmed. "
                "An unfamiliar location indicates an unobserved spatial area, not necessarily danger."
            ),
            "orthogonality": (
                "Geospatial context is strictly decoupled from kinematic XGBoost excursion predictions "
                "and does not modify risk scores, features, or alert thresholds."
            ),
        }
    )


# ==============================================================================
# SAFE AREA GEOMETRY ABSTRACTIONS
# ==============================================================================


@dataclass
class CircularSafeArea:
    """Circular geographic safe area defined by center coordinate and radius."""

    name: str
    center_latitude: float
    center_longitude: float
    radius_m: float
    safe_area_id: Optional[str] = None

    def contains(self, lat: float, lon: float, buffer_m: float = 0.0) -> bool:
        """Check if coordinate is within circular boundary plus buffer."""
        dist = float(
            haversine_distance(lat, lon, self.center_latitude, self.center_longitude)
        )
        return dist <= (self.radius_m + buffer_m)

    def distance_to_boundary_m(self, lat: float, lon: float) -> float:
        """Signed distance to boundary in meters (negative = inside, positive = outside)."""
        dist = float(
            haversine_distance(lat, lon, self.center_latitude, self.center_longitude)
        )
        return dist - self.radius_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "circle",
            "name": self.name,
            "safe_area_id": self.safe_area_id or self.name,
            "center": [self.center_latitude, self.center_longitude],
            "radius_m": self.radius_m,
        }


@dataclass
class PolygonSafeArea:
    """Polygonal geographic safe area defined by a sequence of vertex coordinates."""

    name: str
    vertices: List[Tuple[float, float]]  # [(lat, lon), ...]
    safe_area_id: Optional[str] = None

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError(
                f"PolygonSafeArea '{self.name}' requires at least 3 vertices."
            )

    def contains(self, lat: float, lon: float, buffer_m: float = 0.0) -> bool:
        """Point-in-polygon ray casting algorithm."""
        n = len(self.vertices)
        inside = False
        p1_lat, p1_lon = self.vertices[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = self.vertices[i % n]
            if min(p1_lat, p2_lat) < lat <= max(p1_lat, p2_lat):
                if p1_lat != p2_lat:
                    x_inters = (lat - p1_lat) * (p2_lon - p1_lon) / (
                        p2_lat - p1_lat
                    ) + p1_lon
                    if lon <= x_inters:
                        inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon

        if inside:
            return True
        if buffer_m > 0:
            # Check if within buffer distance of any edge
            dist = self.distance_to_boundary_m(lat, lon)
            return dist <= buffer_m
        return False

    def distance_to_boundary_m(self, lat: float, lon: float) -> float:
        """Approximate distance in meters from point to polygon boundary."""
        # Convert to local Cartesian coordinates centered at (lat, lon)
        rad_lat = math.radians(lat)
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * rad_lat)
        m_per_deg_lon = 111412.84 * math.cos(rad_lat)

        min_dist_m = float("inf")
        n = len(self.vertices)
        for i in range(n):
            v1_lat, v1_lon = self.vertices[i]
            v2_lat, v2_lon = self.vertices[(i + 1) % n]

            # Vector v1 to v2 in meters
            x1 = (v1_lon - lon) * m_per_deg_lon
            y1 = (v1_lat - lat) * m_per_deg_lat
            x2 = (v2_lon - lon) * m_per_deg_lon
            y2 = (v2_lat - lat) * m_per_deg_lat

            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                dist_seg = math.sqrt(x1 * x1 + y1 * y1)
            else:
                # Project origin (0, 0) onto segment
                t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg_len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist_seg = math.sqrt(proj_x * proj_x + proj_y * proj_y)

            if dist_seg < min_dist_m:
                min_dist_m = dist_seg

        return min_dist_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "polygon",
            "name": self.name,
            "safe_area_id": self.safe_area_id or self.name,
            "vertices": [[lat, lon] for lat, lon in self.vertices],
        }


@dataclass
class SafeAreaBoundary:
    """Container holding one or more circular or polygonal safe areas."""

    zones: List[Union[CircularSafeArea, PolygonSafeArea]] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: Any) -> Optional[SafeAreaBoundary]:
        """Parse arbitrary user-supplied safe area specification into a SafeAreaBoundary."""
        if spec is None:
            return None
        if isinstance(spec, SafeAreaBoundary):
            return spec if spec.zones else None

        zones: List[Union[CircularSafeArea, PolygonSafeArea]] = []

        if isinstance(spec, (CircularSafeArea, PolygonSafeArea)):
            zones.append(spec)
            return cls(zones=zones)

        if isinstance(spec, (str, Path)):
            path = Path(spec)
            if path.exists() and path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        spec = json.load(f)
                except Exception as e:
                    logger.warning(
                        f"Failed to read safe area JSON from file {spec}: {e}"
                    )
                    return None
            else:
                try:
                    spec = json.loads(str(spec))
                except Exception:
                    # Check comma separated string: lat,lon,radius_m
                    parts = [p.strip() for p in str(spec).split(",")]
                    if len(parts) >= 3:
                        try:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            rad = float(parts[2])
                            name = parts[3] if len(parts) > 3 else "Safe Zone"
                            zones.append(
                                CircularSafeArea(
                                    name=name,
                                    center_latitude=lat,
                                    center_longitude=lon,
                                    radius_m=rad,
                                )
                            )
                            return cls(zones=zones)
                        except ValueError:
                            pass
                    return None

        if isinstance(spec, dict):
            # Check wrapped list under "safe_areas" or "zones"
            if "safe_areas" in spec:
                spec = spec["safe_areas"]
            elif "zones" in spec:
                spec = spec["zones"]
            elif "type" in spec:
                parsed = cls._parse_single_zone_dict(spec)
                if parsed:
                    zones.append(parsed)
                return cls(zones=zones) if zones else None
            elif "center_latitude" in spec or "latitude" in spec or "center" in spec:
                parsed = cls._parse_single_zone_dict(spec)
                if parsed:
                    zones.append(parsed)
                return cls(zones=zones) if zones else None

        if isinstance(spec, (list, tuple)):
            for item in spec:
                if isinstance(item, (CircularSafeArea, PolygonSafeArea)):
                    zones.append(item)
                elif isinstance(item, dict):
                    parsed = cls._parse_single_zone_dict(item)
                    if parsed:
                        zones.append(parsed)

        return cls(zones=zones) if zones else None

    @staticmethod
    def _parse_single_zone_dict(
        d: Dict[str, Any],
    ) -> Optional[Union[CircularSafeArea, PolygonSafeArea]]:
        """Parse a single zone dictionary representation."""
        z_type = str(d.get("type", "")).lower()
        name = d.get("name", "Configured Safe Zone")
        safe_id = d.get("safe_area_id", name)

        # 1. Circle specification
        if (
            z_type == "circle"
            or "radius_m" in d
            or "radius" in d
            or "center_latitude" in d
        ):
            rad = float(d.get("radius_m", d.get("radius", 250.0)))
            if "center" in d and isinstance(d["center"], (list, tuple)):
                lat, lon = float(d["center"][0]), float(d["center"][1])
            else:
                lat = float(d.get("center_latitude", d.get("latitude", 0.0)))
                lon = float(d.get("center_longitude", d.get("longitude", 0.0)))
            return CircularSafeArea(
                name=name,
                center_latitude=lat,
                center_longitude=lon,
                radius_m=rad,
                safe_area_id=safe_id,
            )

        # 2. Polygon specification
        if (
            z_type in ("polygon", "feature")
            or "vertices" in d
            or "coordinates" in d
        ):
            verts: List[Tuple[float, float]] = []
            if "vertices" in d:
                for v in d["vertices"]:
                    verts.append((float(v[0]), float(v[1])))
            elif "coordinates" in d:
                raw_coords = d["coordinates"]
                # Handle GeoJSON coordinates: [[[lon, lat], ...]]
                if (
                    raw_coords
                    and isinstance(raw_coords[0], list)
                    and isinstance(raw_coords[0][0], list)
                ):
                    raw_coords = raw_coords[0]
                for c in raw_coords:
                    # GeoJSON is [lon, lat]; convert to (lat, lon)
                    verts.append((float(c[1]), float(c[0])))
            if len(verts) >= 3:
                return PolygonSafeArea(
                    name=name, vertices=verts, safe_area_id=safe_id
                )

        return None

    def evaluate(
        self, lat: float, lon: float, buffer_m: float = 0.0
    ) -> Tuple[SafeAreaState, Optional[float], Optional[str]]:
        """Evaluate a location point against all safe areas in this boundary.

        Returns:
            Tuple of (SafeAreaState, distance_to_boundary_m, matched_zone_name)
        """
        if not self.zones or math.isnan(lat) or math.isnan(lon):
            return SafeAreaState.SAFE_AREA_UNAVAILABLE, None, None

        # Check if inside any zone
        inside_matches: List[Tuple[float, str]] = []
        outside_distances: List[Tuple[float, str]] = []

        for zone in self.zones:
            if zone.contains(lat, lon, buffer_m=buffer_m):
                margin = zone.distance_to_boundary_m(lat, lon)
                inside_matches.append((margin, zone.name))
            else:
                dist = zone.distance_to_boundary_m(lat, lon)
                outside_distances.append((dist, zone.name))

        if inside_matches:
            # Sort by deepest inside (most negative margin)
            inside_matches.sort(key=lambda x: x[0])
            best_margin, best_name = inside_matches[0]
            return SafeAreaState.INSIDE_SAFE_AREA, round(best_margin, 1), best_name

        if outside_distances:
            # Sort by closest to boundary
            outside_distances.sort(key=lambda x: x[0])
            min_dist, closest_name = outside_distances[0]
            return (
                SafeAreaState.OUTSIDE_SAFE_AREA,
                round(min_dist, 1),
                closest_name,
            )

        return SafeAreaState.SAFE_AREA_UNAVAILABLE, None, None


# ==============================================================================
# GEOSPATIAL CONTEXT RESULT
# ==============================================================================


@dataclass
class GeospatialContextResult:
    """Comprehensive, orthogonal geospatial context record."""

    # 1. Safe Area State
    safe_area_state: SafeAreaState
    is_safe_area_available: bool
    distance_to_safe_boundary_m: Optional[float]
    matched_safe_area_name: Optional[str]

    # 2. Familiarity State
    familiarity_state: FamiliarityState
    nearest_anchor_id: Optional[str]
    distance_to_nearest_anchor_m: Optional[float]
    nearest_anchor_status: Optional[str]
    is_caregiver_confirmed: bool
    trip_count: int

    # 3. Disclaimers & Notes
    notes: List[str]
    disclaimers: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to deterministic dictionary for serialization and reporting."""
        return {
            "safe_area_state": self.safe_area_state.value,
            "is_safe_area_available": self.is_safe_area_available,
            "distance_to_safe_boundary_m": self.distance_to_safe_boundary_m,
            "matched_safe_area_name": self.matched_safe_area_name,
            "familiarity_state": self.familiarity_state.value,
            "nearest_anchor_id": self.nearest_anchor_id,
            "distance_to_nearest_anchor_m": self.distance_to_nearest_anchor_m,
            "nearest_anchor_status": self.nearest_anchor_status,
            "is_caregiver_confirmed": self.is_caregiver_confirmed,
            "trip_count": self.trip_count,
            "notes": self.notes,
            "disclaimers": self.disclaimers,
        }


# ==============================================================================
# FAMILIARITY EVALUATION ENGINE
# ==============================================================================


def evaluate_location_familiarity(
    lat: float,
    lon: float,
    user_profile: Optional[Dict[str, Any]],
    config: GeospatialConfig = GeospatialConfig(),
) -> Tuple[
    FamiliarityState,
    Optional[str],
    Optional[float],
    Optional[str],
    bool,
    int,
    List[str],
]:
    """Evaluate location familiarity against learned profile anchors.

    Returns:
        (familiarity_state, nearest_anchor_id, distance_m, anchor_status,
         is_caregiver_confirmed, trip_count, notes)
    """
    notes: List[str] = []

    # 1. Coordinate check
    if math.isnan(lat) or math.isnan(lon):
        notes.append("Latitude or longitude coordinate is missing or NaN.")
        return (
            FamiliarityState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            None,
            False,
            0,
            notes,
        )

    # 2. Profile presence check
    if user_profile is None:
        notes.append("No historical user profile found; cold-start user.")
        return FamiliarityState.NO_HISTORY, None, None, None, False, 0, notes

    trip_count = int(user_profile.get("trip_count", 0))

    if trip_count == 0:
        notes.append(
            "User profile has 0 recorded historical trips; pure cold start."
        )
        return (
            FamiliarityState.NO_HISTORY,
            None,
            None,
            None,
            False,
            trip_count,
            notes,
        )

    # 3. Evidence sufficiency check (requires >= 7 trips per standardized transition)
    if trip_count < config.familiarity_min_trips:
        notes.append(
            f"Historical trip count ({trip_count}) is below minimum threshold ({config.familiarity_min_trips}) "
            "to establish a reliable spatial familiarity baseline."
        )
        return (
            FamiliarityState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            None,
            False,
            trip_count,
            notes,
        )

    # 4. Extract anchor clusters
    raw_anchors = user_profile.get("anchor_clusters", [])
    if isinstance(raw_anchors, str):
        try:
            raw_anchors = json.loads(raw_anchors)
        except Exception:
            raw_anchors = []

    if not raw_anchors:
        notes.append(
            "User has completed trips but no spatial anchor clusters were detected by DBSCAN."
        )
        return (
            FamiliarityState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            None,
            False,
            trip_count,
            notes,
        )

    # 5. Evaluate distance to learned anchors
    closest_anchor_id: Optional[str] = None
    min_dist_m: float = float("inf")
    closest_status: Optional[str] = None
    matched_familiar = False

    for anchor in raw_anchors:
        if not isinstance(anchor, dict):
            continue

        a_id = str(anchor.get("anchor_id", anchor.get("name", "anchor")))
        status = str(anchor.get("status", "PENDING_CAREGIVER_REVIEW"))

        # Coordinate resolution
        if "center" in anchor and isinstance(anchor["center"], (list, tuple)):
            a_lat = float(anchor["center"][0])
            a_lon = float(anchor["center"][1])
        else:
            a_lat = float(anchor.get("center_latitude", 0.0))
            a_lon = float(anchor.get("center_longitude", 0.0))

        raw_radius = float(anchor.get("radius_m", config.familiarity_anchor_min_radius_m))
        eff_radius = max(
            raw_radius * config.familiarity_anchor_expansion_factor,
            config.familiarity_anchor_min_radius_m,
        )

        dist_m = float(haversine_distance(lat, lon, a_lat, a_lon))

        if dist_m < min_dist_m:
            min_dist_m = dist_m
            closest_anchor_id = a_id
            closest_status = status

        if dist_m <= eff_radius:
            matched_familiar = True

    # Algorithmic anchors are never caregiver confirmed unless explicitly marked
    # CAREGIVER_CONFIRMED. By design in profile.py, status is CONFIRMED (algorithmic DBSCAN)
    # or PENDING_CAREGIVER_REVIEW.
    is_caregiver_confirmed = (
        closest_status == "CAREGIVER_CONFIRMED"
    )

    if matched_familiar:
        notes.append(
            f"Location falls within catchment radius of anchor '{closest_anchor_id}' "
            f"({min_dist_m:.1f}m away, status: {closest_status})."
        )
        return (
            FamiliarityState.FAMILIAR_LOCATION,
            closest_anchor_id,
            round(min_dist_m, 1),
            closest_status,
            is_caregiver_confirmed,
            trip_count,
            notes,
        )
    else:
        notes.append(
            f"Location is outside all {len(raw_anchors)} known anchor clusters; "
            f"closest is '{closest_anchor_id}' at {min_dist_m:.1f}m."
        )
        return (
            FamiliarityState.UNFAMILIAR_LOCATION,
            closest_anchor_id,
            round(min_dist_m, 1),
            closest_status,
            is_caregiver_confirmed,
            trip_count,
            notes,
        )


# ==============================================================================
# UNIFIED GEOSPATIAL CONTEXT EVALUATOR
# ==============================================================================


def evaluate_geospatial_context(
    latitude: Optional[float],
    longitude: Optional[float],
    safe_area: Optional[Union[SafeAreaBoundary, Dict[str, Any], Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    config: GeospatialConfig = GeospatialConfig(),
) -> GeospatialContextResult:
    """Evaluate safe-area state and location familiarity for a coordinate observation.

    Guarantees:
    - Pure contextual layer: decoupled from XGBoost inference and training.
    - Safe-area boundaries are externally supplied; returns SAFE_AREA_UNAVAILABLE if none provided.
    - Evaluates familiarity state across FAMILIAR_LOCATION, UNFAMILIAR_LOCATION,
      INSUFFICIENT_EVIDENCE, and NO_HISTORY.
    - Attaches explicit scientific disclaimers to prevent misleading interpretations.
    """
    notes: List[str] = []

    # Handle missing / NaN coordinates
    if latitude is None or longitude is None or math.isnan(latitude) or math.isnan(longitude):
        return GeospatialContextResult(
            safe_area_state=SafeAreaState.SAFE_AREA_UNAVAILABLE,
            is_safe_area_available=False,
            distance_to_safe_boundary_m=None,
            matched_safe_area_name=None,
            familiarity_state=FamiliarityState.INSUFFICIENT_EVIDENCE,
            nearest_anchor_id=None,
            distance_to_nearest_anchor_m=None,
            nearest_anchor_status=None,
            is_caregiver_confirmed=False,
            trip_count=int(user_profile.get("trip_count", 0)) if user_profile else 0,
            notes=["Coordinates are unavailable or invalid (NaN)."],
            disclaimers=config.disclaimers,
        )

    # 1. Safe Area Evaluation
    safe_boundary = SafeAreaBoundary.from_spec(safe_area)
    if safe_boundary is None or not safe_boundary.zones:
        safe_state = SafeAreaState.SAFE_AREA_UNAVAILABLE
        is_safe_avail = False
        safe_dist = None
        safe_name = None
        notes.append(
            "No safe area boundary configured for this evaluation (SAFE_AREA_UNAVAILABLE)."
        )
    else:
        is_safe_avail = True
        safe_state, safe_dist, safe_name = safe_boundary.evaluate(
            latitude, longitude, buffer_m=config.safe_area_default_buffer_m
        )
        if safe_state == SafeAreaState.INSIDE_SAFE_AREA:
            notes.append(
                f"Location is inside safe area '{safe_name}' (margin: {safe_dist}m)."
            )
        else:
            notes.append(
                f"Location is outside safe area '{safe_name}' (distance: {safe_dist}m to nearest edge)."
            )

    # 2. Familiarity State Evaluation
    (
        fam_state,
        anchor_id,
        anchor_dist,
        anchor_status,
        caregiver_confirmed,
        t_count,
        fam_notes,
    ) = evaluate_location_familiarity(
        lat=latitude,
        lon=longitude,
        user_profile=user_profile,
        config=config,
    )
    notes.extend(fam_notes)

    return GeospatialContextResult(
        safe_area_state=safe_state,
        is_safe_area_available=is_safe_avail,
        distance_to_safe_boundary_m=safe_dist,
        matched_safe_area_name=safe_name,
        familiarity_state=fam_state,
        nearest_anchor_id=anchor_id,
        distance_to_nearest_anchor_m=anchor_dist,
        nearest_anchor_status=anchor_status,
        is_caregiver_confirmed=caregiver_confirmed,
        trip_count=t_count,
        notes=notes,
        disclaimers=config.disclaimers,
    )

"""
Per-Patient, Per-Zone Boundary Transition State Machine (Phase 2).

Manages geofence boundary states (INSIDE, EXIT_PENDING, BREACHED, REENTRY_PENDING, RESOLVED)
with configurable GPS accuracy filtering, debouncing, and hysteresis to prevent
false alarms caused by location jitter.

Integrates with AlertRecord lifecycle:
- Confirmed breach creates/updates one persistent AlertRecord.
- Continued movement across H3 cells updates the active alert location without alert duplication.
- Confirmed re-entry auto-resolves the active breach and notifies WebSocket clients.
- Multi-zone support (e.g. Home & Company) resolves transit breaches upon entering another safe zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import select

from app.db import async_session, SafeZoneRow, GeofenceStateRow, AlertRecordRow
from app.models import (
    AlertRecord,
    AlertStatus,
    BoundaryState,
    GeofenceEvaluationStatus,
    SafeZoneGeometryType,
)
from app.services.geospatial import location_to_h3_cell
from app.websocket.risk_stream import push_alert_update
from ml.src.geofence import CircularSafeArea, PolygonSafeArea

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeofenceBoundaryConfig:
    """Configurable thresholds for geofence debouncing and hysteresis."""

    exit_consecutive_readings: int = 2
    """Number of consecutive outside readings required to confirm a boundary exit."""

    reentry_consecutive_readings: int = 2
    """Number of consecutive inside readings required to confirm safe re-entry."""

    max_horizontal_accuracy_m: float = 25.0
    """GPS accuracy worse than this is considered degraded/insufficient evidence for state transitions."""

    hysteresis_buffer_m: float = 15.0
    """Tolerance buffer beyond boundary edge to avoid chatter around perimeter."""

    alert_tier: int = 3
    """Default alert tier for safe-zone breach alerts (Tier 3 = CRITICAL)."""


DEFAULT_GEOFENCE_CONFIG = GeofenceBoundaryConfig()


@dataclass
class GeofenceEvaluationResult:
    """Evaluation summary returned after processing a telemetry coordinate against safe zones."""

    status: GeofenceEvaluationStatus
    active_breaches: List[Dict[str, Any]] = field(default_factory=list)
    zone_states: List[Dict[str, Any]] = field(default_factory=list)
    alerts_triggered: List[AlertRecord] = field(default_factory=list)
    alerts_updated: List[AlertRecord] = field(default_factory=list)
    alerts_resolved: List[AlertRecord] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "active_breaches": self.active_breaches,
            "zone_states": self.zone_states,
            "alerts_triggered_count": len(self.alerts_triggered),
            "alerts_updated_count": len(self.alerts_updated),
            "alerts_resolved_count": len(self.alerts_resolved),
            "message": self.message,
        }


def _safe_zone_row_to_geometry(row: SafeZoneRow) -> Optional[CircularSafeArea | PolygonSafeArea]:
    """Convert a persisted SafeZoneRow into an ml.src.geofence boundary object."""
    if row.geometry_type == SafeZoneGeometryType.CIRCLE.value:
        if row.center_lat is not None and row.center_lng is not None and row.radius_m is not None:
            return CircularSafeArea(
                name=row.name,
                center_latitude=row.center_lat,
                center_longitude=row.center_lng,
                radius_m=row.radius_m,
                safe_area_id=row.zone_id,
            )
    elif row.geometry_type == SafeZoneGeometryType.POLYGON.value:
        if row.polygon_vertices_json:
            try:
                verts = json.loads(row.polygon_vertices_json)
                tuples = [(float(v[0]), float(v[1])) for v in verts]
                if len(tuples) >= 3:
                    return PolygonSafeArea(
                        name=row.name,
                        vertices=tuples,
                        safe_area_id=row.zone_id,
                    )
            except Exception:
                pass
    return None


def calculate_signed_distance_to_zone(
    zone_geom: CircularSafeArea | PolygonSafeArea,
    lat: float,
    lng: float,
) -> float:
    """
    Calculate signed distance to safe-zone boundary in meters:
    - Negative: point is INSIDE boundary.
    - Positive: point is OUTSIDE boundary.
    """
    if isinstance(zone_geom, CircularSafeArea):
        return float(zone_geom.distance_to_boundary_m(lat, lng))
    elif isinstance(zone_geom, PolygonSafeArea):
        is_inside = zone_geom.contains(lat, lng)
        d_edge = float(zone_geom.distance_to_boundary_m(lat, lng))
        return -d_edge if is_inside else d_edge
    return 0.0


async def evaluate_patient_geofence_boundaries(
    user_id: str,
    lat: float,
    lng: float,
    accuracy_m: float,
    timestamp: datetime,
    is_signal_degraded: bool = False,
    config: GeofenceBoundaryConfig = DEFAULT_GEOFENCE_CONFIG,
) -> GeofenceEvaluationResult:
    """
    Evaluate telemetry coordinates against all enabled safe zones for a patient.
    Updates boundary state machines, creates/updates breach alerts, and resolves breaches upon re-entry.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    # 1. Fetch enabled safe zones for patient
    try:
        async with async_session() as session:
            result = await session.execute(
                select(SafeZoneRow).where(
                    SafeZoneRow.user_id == user_id,
                    SafeZoneRow.is_enabled.is_(True),
                )
            )
            zone_rows = result.scalars().all()
    except Exception as e:
        if "no such table" in str(e).lower():
            from app.db import init_db
            await init_db()
            async with async_session() as session:
                result = await session.execute(
                    select(SafeZoneRow).where(
                        SafeZoneRow.user_id == user_id,
                        SafeZoneRow.is_enabled.is_(True),
                    )
                )
                zone_rows = result.scalars().all()
        else:
            raise e

    # Requirement 9: Report absence of configured zones explicitly
    if not zone_rows:
        return GeofenceEvaluationResult(
            status=GeofenceEvaluationStatus.NO_ZONES_CONFIGURED,
            message=f"No enabled safe zones configured for patient '{user_id}'.",
        )

    # Requirement 9: Report insufficient GPS evidence explicitly
    if is_signal_degraded or accuracy_m > config.max_horizontal_accuracy_m:
        return GeofenceEvaluationResult(
            status=GeofenceEvaluationStatus.INSUFFICIENT_GPS,
            message=(
                f"GPS accuracy ({accuracy_m:.1f}m) exceeds threshold ({config.max_horizontal_accuracy_m}m) "
                "or signal is degraded. Boundary state transitions held to prevent false alarms."
            ),
        )

    grid_cell_id = location_to_h3_cell(lat, lng)
    alerts_triggered: List[AlertRecord] = []
    alerts_updated: List[AlertRecord] = []
    alerts_resolved: List[AlertRecord] = []
    active_breaches: List[Dict[str, Any]] = []
    zone_states_summary: List[Dict[str, Any]] = []

    # Map zones and compute signed distances
    zone_evaluations: List[Tuple[SafeZoneRow, CircularSafeArea | PolygonSafeArea, float, bool]] = []
    any_zone_inside = False

    for z_row in zone_rows:
        geom = _safe_zone_row_to_geometry(z_row)
        if geom is None:
            continue
        signed_dist = calculate_signed_distance_to_zone(geom, lat, lng)
        is_inside = signed_dist <= 0.0
        if is_inside:
            any_zone_inside = True
        zone_evaluations.append((z_row, geom, signed_dist, is_inside))

    # Process each zone's boundary state machine
    async with async_session() as session:
        for z_row, geom, signed_dist, is_inside in zone_evaluations:
            # Load or initialize persisted state for (user_id, zone_id)
            state_res = await session.execute(
                select(GeofenceStateRow).where(
                    GeofenceStateRow.user_id == user_id,
                    GeofenceStateRow.zone_id == z_row.zone_id,
                )
            )
            state_row = state_res.scalar_one_or_none()
            if state_row is None:
                initial_state = BoundaryState.INSIDE if is_inside else BoundaryState.OUTSIDE
                state_row = GeofenceStateRow(
                    user_id=user_id,
                    zone_id=z_row.zone_id,
                    state=initial_state.value,
                    consecutive_outside=0 if is_inside else 1,
                    consecutive_inside=1 if is_inside else 0,
                    last_distance_m=signed_dist,
                    last_evaluated_at=timestamp,
                    active_alert_id=None,
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(state_row)
                await session.flush()

            curr_state = BoundaryState(state_row.state)
            effective_buffer = max(config.hysteresis_buffer_m, min(accuracy_m, 20.0))

            # -------------------------------------------------------------
            # Case 1: Point is INSIDE this zone
            # -------------------------------------------------------------
            if is_inside:
                state_row.consecutive_inside += 1
                state_row.consecutive_outside = 0

                if curr_state in (BoundaryState.BREACHED, BoundaryState.REENTRY_PENDING):
                    # Patient in breach is returning inside
                    if state_row.consecutive_inside >= config.reentry_consecutive_readings:
                        # Confirmed re-entry!
                        state_row.state = BoundaryState.RESOLVED.value

                        # Resolve active alert if present
                        if state_row.active_alert_id:
                            alert_res = await session.execute(
                                select(AlertRecordRow).where(
                                    AlertRecordRow.alert_id == state_row.active_alert_id
                                )
                            )
                            alert_row = alert_res.scalar_one_or_none()
                            if alert_row and alert_row.status == AlertStatus.PENDING.value:
                                alert_row.status = AlertStatus.AUTO_RESOLVED.value
                                alert_row.timestamp = timestamp
                                resolved_rec = _db_row_to_alert_record(alert_row)
                                alerts_resolved.append(resolved_rec)
                                await push_alert_update(user_id, resolved_rec.model_dump(mode="json"))
                            state_row.active_alert_id = None

                        # Transition back to stable INSIDE
                        state_row.state = BoundaryState.INSIDE.value
                    else:
                        state_row.state = BoundaryState.REENTRY_PENDING.value

                elif curr_state == BoundaryState.EXIT_PENDING:
                    # Jitter outside reversed back inside
                    state_row.state = BoundaryState.INSIDE.value

                else:
                    state_row.state = BoundaryState.INSIDE.value

            # -------------------------------------------------------------
            # Case 2: Point is OUTSIDE this zone
            # -------------------------------------------------------------
            else:
                state_row.consecutive_outside += 1
                state_row.consecutive_inside = 0

                if curr_state == BoundaryState.INSIDE:
                    # Initial boundary departure -> debouncing
                    state_row.state = BoundaryState.EXIT_PENDING.value
                    # If single reading is already past debouncing and confirmed outside
                    if (
                        config.exit_consecutive_readings <= 1
                        and signed_dist > effective_buffer
                    ):
                        state_row.state = BoundaryState.BREACHED.value

                elif curr_state == BoundaryState.EXIT_PENDING:
                    # Check if exit is confirmed
                    if (
                        state_row.consecutive_outside >= config.exit_consecutive_readings
                        and signed_dist > effective_buffer
                    ):
                        state_row.state = BoundaryState.BREACHED.value

                elif curr_state == BoundaryState.REENTRY_PENDING:
                    # False re-entry reversed back outside
                    state_row.state = BoundaryState.BREACHED.value

            # -------------------------------------------------------------
            # Multi-zone transit resolution: If patient is confirmed inside
            # ANY safe zone (e.g. entered Company), resolve active breach for Home
            # -------------------------------------------------------------
            if any_zone_inside and state_row.state == BoundaryState.BREACHED.value:
                # Patient arrived inside another safe zone
                state_row.state = BoundaryState.OUTSIDE.value
                if state_row.active_alert_id:
                    alert_res = await session.execute(
                        select(AlertRecordRow).where(
                            AlertRecordRow.alert_id == state_row.active_alert_id
                        )
                    )
                    alert_row = alert_res.scalar_one_or_none()
                    if alert_row and alert_row.status == AlertStatus.PENDING.value:
                        alert_row.status = AlertStatus.AUTO_RESOLVED.value
                        alert_row.timestamp = timestamp
                        resolved_rec = _db_row_to_alert_record(alert_row)
                        alerts_resolved.append(resolved_rec)
                        await push_alert_update(user_id, resolved_rec.model_dump(mode="json"))
                    state_row.active_alert_id = None

            # -------------------------------------------------------------
            # Handle Persistent Breach Alert Creation & Deduplication
            # -------------------------------------------------------------
            if state_row.state == BoundaryState.BREACHED.value:
                active_breaches.append({
                    "zone_id": z_row.zone_id,
                    "name": z_row.name,
                    "distance_m": round(signed_dist, 1),
                })

                if not state_row.active_alert_id:
                    # Create new breach alert
                    new_alert_id = uuid4().hex
                    new_alert_row = AlertRecordRow(
                        alert_id=new_alert_id,
                        user_id=user_id,
                        grid_cell_id=grid_cell_id,
                        timestamp=timestamp,
                        alert_tier=config.alert_tier,
                        status=AlertStatus.PENDING.value,
                        zone_id=z_row.zone_id,
                    )
                    session.add(new_alert_row)
                    state_row.active_alert_id = new_alert_id
                    await session.flush()

                    created_rec = _db_row_to_alert_record(new_alert_row)
                    alerts_triggered.append(created_rec)
                    await push_alert_update(user_id, created_rec.model_dump(mode="json"))
                else:
                    # Breach continues: update existing alert location (deduplication across H3 cells)
                    alert_res = await session.execute(
                        select(AlertRecordRow).where(
                            AlertRecordRow.alert_id == state_row.active_alert_id
                        )
                    )
                    existing_alert = alert_res.scalar_one_or_none()
                    if existing_alert and existing_alert.status == AlertStatus.PENDING.value:
                        existing_alert.grid_cell_id = grid_cell_id
                        existing_alert.timestamp = timestamp
                        await session.flush()
                        updated_rec = _db_row_to_alert_record(existing_alert)
                        alerts_updated.append(updated_rec)
                        await push_alert_update(user_id, updated_rec.model_dump(mode="json"))

            # Save state metrics
            state_row.last_distance_m = round(signed_dist, 1)
            state_row.last_evaluated_at = timestamp
            state_row.updated_at = datetime.now(timezone.utc)

            zone_states_summary.append({
                "zone_id": z_row.zone_id,
                "name": z_row.name,
                "state": state_row.state,
                "distance_m": round(signed_dist, 1),
                "consecutive_outside": state_row.consecutive_outside,
                "consecutive_inside": state_row.consecutive_inside,
                "active_alert_id": state_row.active_alert_id,
            })

        await session.commit()

    overall_status = (
        GeofenceEvaluationStatus.BREACH_ACTIVE
        if active_breaches
        else GeofenceEvaluationStatus.NORMAL
    )

    msg = (
        f"Active boundary breach in {len(active_breaches)} zone(s)."
        if active_breaches
        else "All configured safe zones within normal operating boundaries."
    )

    return GeofenceEvaluationResult(
        status=overall_status,
        active_breaches=active_breaches,
        zone_states=zone_states_summary,
        alerts_triggered=alerts_triggered,
        alerts_updated=alerts_updated,
        alerts_resolved=alerts_resolved,
        message=msg,
    )


def _db_row_to_alert_record(row: AlertRecordRow) -> AlertRecord:
    """Helper converting AlertRecordRow to Pydantic AlertRecord."""
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return AlertRecord(
        alert_id=row.alert_id,
        user_id=row.user_id,
        grid_cell_id=row.grid_cell_id,
        timestamp=ts,
        alert_tier=row.alert_tier,
        status=AlertStatus(row.status),
        zone_id=getattr(row, "zone_id", None),
        dismissal=None,
    )

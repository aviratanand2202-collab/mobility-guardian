"""Persistent Trajectory Looping Alert Lifecycle & Session State Machine (Phase 3).

Manages the lifecycle of persistent looping alerts:
1. Session Tracking: Tracks active looping events per patient to prevent alert duplication.
2. Cross-Cell Deduplication: Continued movement across H3 cells while looping updates
   the active alert's location and timestamp rather than generating new alerts.
3. Physiological Escalation: If physiological arousal (elevated resting heart rate) occurs
   during an ongoing loop, the active alert tier is escalated from Tier 2 (SUSPICIOUS)
   to Tier 3 (CRITICAL).
4. Auto-Resolution: When looping ceases (consecutive non-looping observations, return to a safe zone,
   or progressive linear transit), the active alert is auto-resolved cleanly and connected
   clients are notified over WebSockets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import select

from app.db import async_session, init_db, AlertRecordRow, LoopStateRow
from app.models import AlertRecord, AlertStatus
from app.services.geospatial import location_to_h3_cell
from app.websocket.risk_stream import push_alert_update

logger = logging.getLogger(__name__)


@dataclass
class LoopLifecycleResult:
    """Outcome of evaluating trajectory loop evidence against active alert sessions."""

    is_looping: bool
    active_alert_id: Optional[str]
    action: str  # "NONE", "ALERT_TRIGGERED", "ALERT_UPDATED", "ALERT_RESOLVED"
    alert_tier: Optional[int]
    consecutive_non_looping: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def evaluate_loop_alert_lifecycle(
    user_id: str,
    lat: float,
    lng: float,
    timestamp: datetime,
    loop_evidence: Optional[Dict[str, Any]] = None,
    hr_evidence: Optional[Dict[str, Any]] = None,
    is_inside_safe_zone: bool = False,
) -> LoopLifecycleResult:
    """Evaluate trajectory loop evidence and manage persistent AlertRecord lifecycle."""
    try:
        return await _evaluate_loop_lifecycle_inner(
            user_id=user_id,
            lat=lat,
            lng=lng,
            timestamp=timestamp,
            loop_evidence=loop_evidence,
            hr_evidence=hr_evidence,
            is_inside_safe_zone=is_inside_safe_zone,
        )
    except Exception as exc:
        if "no such table" in str(exc).lower():
            logger.info("Database table missing; initializing tables and retrying.")
            await init_db()
            return await _evaluate_loop_lifecycle_inner(
                user_id=user_id,
                lat=lat,
                lng=lng,
                timestamp=timestamp,
                loop_evidence=loop_evidence,
                hr_evidence=hr_evidence,
                is_inside_safe_zone=is_inside_safe_zone,
            )
        raise


async def _evaluate_loop_lifecycle_inner(
    user_id: str,
    lat: float,
    lng: float,
    timestamp: datetime,
    loop_evidence: Optional[Dict[str, Any]] = None,
    hr_evidence: Optional[Dict[str, Any]] = None,
    is_inside_safe_zone: bool = False,
) -> LoopLifecycleResult:
    grid_cell_id = location_to_h3_cell(lat, lng)

    is_looping = bool(loop_evidence and loop_evidence.get("is_looping"))
    is_corroborated = bool(hr_evidence and hr_evidence.get("is_corroborating_risk"))
    target_tier = 3 if is_corroborated else 2
    loop_pattern = loop_evidence.get("pattern", "NONE") if loop_evidence else "NONE"

    action = "NONE"
    message = "No active looping condition."
    active_alert_id = None

    async with async_session() as session:
        # 1. Fetch or create LoopStateRow
        res = await session.execute(
            select(LoopStateRow).where(LoopStateRow.user_id == user_id)
        )
        state_row = res.scalar_one_or_none()
        if state_row is None:
            state_row = LoopStateRow(
                user_id=user_id,
                is_looping=False,
                active_alert_id=None,
                consecutive_non_looping=0,
                last_pattern=loop_pattern,
                alert_tier=target_tier,
                updated_at=datetime.now(timezone.utc),
            )
            session.add(state_row)
            await session.flush()

        active_alert_id = state_row.active_alert_id

        # 2. Case: Confirmed persistent looping
        if is_looping:
            state_row.is_looping = True
            state_row.consecutive_non_looping = 0
            state_row.last_pattern = loop_pattern

            if not state_row.active_alert_id:
                # Brand new persistent loop alert session
                new_alert_id = uuid4().hex
                new_alert_row = AlertRecordRow(
                    alert_id=new_alert_id,
                    user_id=user_id,
                    grid_cell_id=grid_cell_id,
                    timestamp=timestamp,
                    alert_tier=target_tier,
                    status=AlertStatus.PENDING.value,
                )
                session.add(new_alert_row)
                state_row.active_alert_id = new_alert_id
                state_row.alert_tier = target_tier
                state_row.updated_at = datetime.now(timezone.utc)
                await session.commit()

                rec = AlertRecord(
                    alert_id=new_alert_row.alert_id,
                    user_id=new_alert_row.user_id,
                    grid_cell_id=new_alert_row.grid_cell_id,
                    timestamp=new_alert_row.timestamp,
                    alert_tier=new_alert_row.alert_tier,
                    status=AlertStatus.PENDING,
                )
                await push_alert_update(user_id, rec.model_dump(mode="json"))

                active_alert_id = new_alert_id
                action = "ALERT_TRIGGERED"
                message = f"Persistent looping alert triggered (Tier {target_tier})."
            else:
                # Deduplication: update existing active alert session
                alert_res = await session.execute(
                    select(AlertRecordRow).where(
                        AlertRecordRow.alert_id == state_row.active_alert_id
                    )
                )
                existing_alert = alert_res.scalar_one_or_none()
                if existing_alert and existing_alert.status == AlertStatus.PENDING.value:
                    existing_alert.grid_cell_id = grid_cell_id
                    existing_alert.timestamp = timestamp
                    # Escalate alert tier if physiological corroboration newly detected
                    if target_tier > existing_alert.alert_tier:
                        existing_alert.alert_tier = target_tier
                        state_row.alert_tier = target_tier

                    state_row.updated_at = datetime.now(timezone.utc)
                    await session.commit()

                    rec = AlertRecord(
                        alert_id=existing_alert.alert_id,
                        user_id=existing_alert.user_id,
                        grid_cell_id=existing_alert.grid_cell_id,
                        timestamp=existing_alert.timestamp,
                        alert_tier=existing_alert.alert_tier,
                        status=AlertStatus.PENDING,
                    )
                    await push_alert_update(user_id, rec.model_dump(mode="json"))

                    action = "ALERT_UPDATED"
                    message = f"Persistent looping alert updated across grid cell {grid_cell_id}."
                else:
                    # Alert was previously closed externally; start fresh session
                    new_alert_id = uuid4().hex
                    new_alert_row = AlertRecordRow(
                        alert_id=new_alert_id,
                        user_id=user_id,
                        grid_cell_id=grid_cell_id,
                        timestamp=timestamp,
                        alert_tier=target_tier,
                        status=AlertStatus.PENDING.value,
                    )
                    session.add(new_alert_row)
                    state_row.active_alert_id = new_alert_id
                    state_row.alert_tier = target_tier
                    state_row.updated_at = datetime.now(timezone.utc)
                    await session.commit()

                    rec = AlertRecord(
                        alert_id=new_alert_row.alert_id,
                        user_id=new_alert_row.user_id,
                        grid_cell_id=new_alert_row.grid_cell_id,
                        timestamp=new_alert_row.timestamp,
                        alert_tier=new_alert_row.alert_tier,
                        status=AlertStatus.PENDING,
                    )
                    await push_alert_update(user_id, rec.model_dump(mode="json"))

                    active_alert_id = new_alert_id
                    action = "ALERT_TRIGGERED"
                    message = f"New persistent looping alert triggered (Tier {target_tier})."

        # 3. Case: Movement is NOT looping
        else:
            state_row.last_pattern = loop_pattern
            if state_row.active_alert_id:
                state_row.consecutive_non_looping += 1

                # Check auto-resolution triggers:
                # - Entered safe zone
                # - Progressive transit confirmed
                # - Consecutive non-looping readings >= 2 (debouncing resolution)
                should_resolve = (
                    is_inside_safe_zone
                    or loop_pattern == "PROGRESSIVE_TRANSIT"
                    or state_row.consecutive_non_looping >= 2
                )

                if should_resolve:
                    alert_res = await session.execute(
                        select(AlertRecordRow).where(
                            AlertRecordRow.alert_id == state_row.active_alert_id
                        )
                    )
                    existing_alert = alert_res.scalar_one_or_none()
                    if existing_alert and existing_alert.status == AlertStatus.PENDING.value:
                        existing_alert.status = AlertStatus.AUTO_RESOLVED.value
                        existing_alert.timestamp = timestamp
                        state_row.updated_at = datetime.now(timezone.utc)
                        await session.commit()

                        resolved_rec = AlertRecord(
                            alert_id=existing_alert.alert_id,
                            user_id=existing_alert.user_id,
                            grid_cell_id=existing_alert.grid_cell_id,
                            timestamp=existing_alert.timestamp,
                            alert_tier=existing_alert.alert_tier,
                            status=AlertStatus.AUTO_RESOLVED,
                        )
                        await push_alert_update(user_id, resolved_rec.model_dump(mode="json"))

                        action = "ALERT_RESOLVED"
                        message = "Persistent looping condition resolved (safe zone entered or transit resumed)."

                    state_row.active_alert_id = None
                    state_row.is_looping = False
                    state_row.consecutive_non_looping = 0
                    active_alert_id = None
                else:
                    state_row.updated_at = datetime.now(timezone.utc)
                    await session.commit()
                    message = f"Loop resolution pending ({state_row.consecutive_non_looping}/2 non-looping readings)."
            else:
                state_row.is_looping = False
                state_row.updated_at = datetime.now(timezone.utc)
                await session.commit()

    return LoopLifecycleResult(
        is_looping=is_looping,
        active_alert_id=active_alert_id,
        action=action,
        alert_tier=target_tier if is_looping else None,
        consecutive_non_looping=state_row.consecutive_non_looping,
        message=message,
    )

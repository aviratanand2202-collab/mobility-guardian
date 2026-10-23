"""
Async SQLAlchemy database setup for SQLite persistence.

Tables defined here are internal ORM models for the persistence layer.
They are NOT Pydantic models and do not modify app/models.py.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Resolve the data directory relative to this file's location so it works
# regardless of the working directory uvicorn is launched from.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_DB_PATH = Path(os.getenv("DATABASE_PATH", str(_DATA_DIR / "app.db")))
_DB_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{_DB_PATH}")

engine = create_async_engine(_DB_URL, echo=False)
async_session = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


# ---------- Consent table ----------

class ConsentRecordRow(Base):
    """Mirrors the ConsentRecord Pydantic model (flattened nested objects)."""
    __tablename__ = "consent_records"

    user_id = Column(String, primary_key=True)
    guardian_granted = Column(Boolean, nullable=False)
    guardian_granted_at = Column(DateTime(timezone=True), nullable=False)
    guardian_id = Column(String, nullable=False)
    end_user_consent_applicable = Column(Boolean, nullable=False)
    end_user_granted = Column(Boolean, nullable=True)
    end_user_granted_at = Column(DateTime(timezone=True), nullable=True)
    data_retention_ack = Column(Boolean, nullable=False, default=False)


# ---------- Dismissal quarantine table ----------

class CellDismissalStateRow(Base):
    """Mirrors the CellDismissalState dataclass.

    dismissal_timestamps is stored as a JSON-serialized list of ISO 8601
    strings.  The list is capped at 3 entries, so a join table would be
    over-engineering.
    """
    __tablename__ = "cell_dismissal_states"

    grid_cell_id = Column(String, primary_key=True)
    decay_clock_start = Column(DateTime(timezone=True), nullable=True)
    dismissal_timestamps = Column(Text, nullable=False, default="[]")


# ---------- PDR hysteresis table ----------

class PDRStateRow(Base):
    """Mirrors the PDRState dataclass. Only persists the current tier."""
    __tablename__ = "pdr_states"

    user_id = Column(String, primary_key=True)
    tier = Column(String, nullable=False, default="INDOOR_PACING")


# ---------- Alert records table ----------

class AlertRecordRow(Base):
    """Mirrors the AlertRecord Pydantic model (flattened Dismissal)."""
    __tablename__ = "alert_records"

    alert_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    grid_cell_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    alert_tier = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    dismissed_at = Column(DateTime(timezone=True), nullable=True)
    dismissal_decay_clock_start = Column(DateTime(timezone=True), nullable=True)
    dismissal_count_in_window = Column(Integer, nullable=True)
    zone_id = Column(String, nullable=True, index=True)


# ---------- Risk score history table ----------

class RiskScoreHistoryRow(Base):
    """
    Persisted evaluation history for computed RiskScoreOutput instances.

    NOTE on Data Hygiene / Retention (per docs/limitations.md):
    This table currently has no automated retention, TTL, or pruning policy
    and will grow unbounded in long-running production environments.
    Automated background pruning and H3 rollup aggregations are flagged
    as a known system limitation for future development phases.
    """
    __tablename__ = "risk_score_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    risk_tier = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    polling_tier = Column(Integer, nullable=False)
    grid_cell_id = Column(String, nullable=True)
    raw_output_json = Column(Text, nullable=False)

    __table_args__ = (
        Index("ix_risk_score_history_user_timestamp", "user_id", "timestamp"),
    )


# ---------- Trusted Companion Sessions table ----------

class CompanionSessionRow(Base):
    """
    Temporary companion monitoring session.

    Privacy and Security:
    - Companion is an expiring temporary guest, NOT a permanent family member.
    - Personal ID is NEVER stored or exposed to the caregiver dashboard;
      a privacy-preserving alias (e.g. 'Trusted Companion') is used.
    - Requires explicit consent before location updates are accepted.
    """
    __tablename__ = "companion_sessions"

    session_id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    caregiver_id = Column(String, nullable=False, default="caregiver_primary")
    invite_token = Column(String, nullable=False, unique=True, index=True)
    status = Column(String, nullable=False, default="PENDING_INVITE")
    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    consent_granted = Column(Boolean, nullable=False, default=False)
    consent_granted_at = Column(DateTime(timezone=True), nullable=True)
    companion_alias = Column(String, nullable=False, default="Trusted Companion")
    companion_user_id = Column(String, nullable=True, index=True)
    relationship = Column(String, nullable=False, default="trusted_companion")
    last_companion_lat = Column(Float, nullable=True)
    last_companion_lng = Column(Float, nullable=True)
    last_companion_accuracy_m = Column(Float, nullable=True)
    last_companion_timestamp = Column(DateTime(timezone=True), nullable=True)


async def save_risk_history(
    user_id: str,
    timestamp: DateTime,
    risk_tier: str,
    risk_score: float,
    polling_tier: int,
    grid_cell_id: str | None,
    raw_output_json: str,
) -> RiskScoreHistoryRow:
    """Persist a single computed risk output record to the history table."""
    async with async_session() as session:
        row = RiskScoreHistoryRow(
            user_id=user_id,
            timestamp=timestamp,
            risk_tier=risk_tier,
            risk_score=risk_score,
            polling_tier=polling_tier,
            grid_cell_id=grid_cell_id,
            raw_output_json=raw_output_json,
        )
        session.add(row)
        await session.commit()
        return row


# ---------- Wearable Readings table ----------

class WearableReadingRow(Base):
    """
    Persisted wearable biometric and device readings (e.g. heart rate).
    Decoupled from kinematic GPS telemetry, linked by user_id and timestamp.
    """
    __tablename__ = "wearable_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    heart_rate_bpm = Column(Integer, nullable=False)
    wear_state = Column(String, nullable=False, default="UNKNOWN")
    sensor_status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_wearable_readings_user_timestamp", "user_id", "timestamp"),
    )


# ---------- Patient-Specific Safe Zones table ----------

class SafeZoneRow(Base):
    """
    Caregiver-configured patient safe zones (e.g. Home, Company, Medical Center).
    Supports circular boundaries (center + radius) and arbitrary polygons (vertices list).
    """
    __tablename__ = "safe_zones"

    zone_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    geometry_type = Column(String, nullable=False, default="circle")
    center_lat = Column(Float, nullable=True)
    center_lng = Column(Float, nullable=True)
    radius_m = Column(Float, nullable=True)
    polygon_vertices_json = Column(Text, nullable=True)
    is_enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


async def save_wearable_reading(
    user_id: str,
    timestamp: datetime,
    heart_rate_bpm: int,
    wear_state: str = "UNKNOWN",
    sensor_status: str | None = None,
) -> WearableReadingRow:
    """Persist a single valid wearable reading."""
    async with async_session() as session:
        row = WearableReadingRow(
            user_id=user_id,
            timestamp=timestamp,
            heart_rate_bpm=heart_rate_bpm,
            wear_state=wear_state,
            sensor_status=sensor_status,
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        await session.commit()
        return row


# ---------- Geofence Boundary State table ----------

class GeofenceStateRow(Base):
    """
    Persisted state machine state per patient per zone.
    Tracks boundary transitions (INSIDE, EXIT_PENDING, BREACHED, REENTRY_PENDING, RESOLVED).
    """
    __tablename__ = "geofence_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    zone_id = Column(String, nullable=False, index=True)
    state = Column(String, nullable=False, default="OUTSIDE")
    consecutive_outside = Column(Integer, nullable=False, default=0)
    consecutive_inside = Column(Integer, nullable=False, default=0)
    last_distance_m = Column(Float, nullable=True)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)
    active_alert_id = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_geofence_states_user_zone", "user_id", "zone_id", unique=True),
    )


# ---------- Persistent Looping State table ----------

class LoopStateRow(Base):
    """
    Persisted state machine state for trajectory looping sessions.
    Tracks active looping alerts, debouncing, and auto-resolution.
    """
    __tablename__ = "loop_states"

    user_id = Column(String, primary_key=True)
    is_looping = Column(Boolean, nullable=False, default=False)
    active_alert_id = Column(String, nullable=True)
    consecutive_non_looping = Column(Integer, nullable=False, default=0)
    last_pattern = Column(String, nullable=True)
    alert_tier = Column(Integer, nullable=False, default=2)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

# ---------- Group Separation State table ----------

class GroupSeparationStateRow(Base):
    """
    Persisted state machine state for group/companion separation monitoring.
    Tracks accompaniment status, debouncing, physiological corroboration, and auto-resolution.
    """
    __tablename__ = "group_separation_states"

    patient_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default="ACCOMPANIMENT_UNVERIFIED")
    consecutive_separated = Column(Integer, nullable=False, default=0)
    consecutive_united = Column(Integer, nullable=False, default=0)
    last_distance_m = Column(Float, nullable=True)
    active_alert_id = Column(String, nullable=True)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))



# ---------- Lifecycle ----------

async def init_db() -> None:
    """Create the data/ directory and all tables (idempotent)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        def _migrate_columns(connection):
            res = connection.execute(text("PRAGMA table_info(companion_sessions)"))
            cols = [row[1] for row in res.fetchall()]
            if "companion_user_id" not in cols:
                connection.execute(text("ALTER TABLE companion_sessions ADD COLUMN companion_user_id TEXT"))
            if "relationship" not in cols:
                connection.execute(text("ALTER TABLE companion_sessions ADD COLUMN relationship TEXT DEFAULT 'trusted_companion'"))

            res_alerts = connection.execute(text("PRAGMA table_info(alert_records)"))
            alert_cols = [row[1] for row in res_alerts.fetchall()]
            if "zone_id" not in alert_cols:
                connection.execute(text("ALTER TABLE alert_records ADD COLUMN zone_id TEXT"))

        await conn.run_sync(_migrate_columns)


async def dispose_engine() -> None:
    """Dispose the async engine, closing all pooled connections."""
    await engine.dispose()

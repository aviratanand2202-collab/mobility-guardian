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

        await conn.run_sync(_migrate_columns)


async def dispose_engine() -> None:
    """Dispose the async engine, closing all pooled connections."""
    await engine.dispose()

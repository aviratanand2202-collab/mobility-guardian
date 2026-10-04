"""
Async SQLAlchemy database setup for SQLite persistence.

Tables defined here are internal ORM models for the persistence layer.
They are NOT Pydantic models and do not modify app/models.py.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Resolve the data directory relative to this file's location so it works
# regardless of the working directory uvicorn is launched from.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_DB_PATH = _DATA_DIR / "app.db"
_DB_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

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


# ---------- Lifecycle ----------

async def init_db() -> None:
    """Create the data/ directory and all tables (idempotent)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose the async engine, closing all pooled connections."""
    await engine.dispose()

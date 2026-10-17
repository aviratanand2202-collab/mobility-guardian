"""Pytest configuration for ml/tests ensuring complete database isolation from app.db."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import pytest

# Ensure repository root and backend directory are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
_BACKEND_DATA_DIR = _BACKEND_DIR / "app" / "data"
_TEST_DB_PATH = _BACKEND_DATA_DIR / "test_app.db"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Force isolated test database environment variables before any app module imports
os.environ["DATABASE_PATH"] = str(_TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"

# If app.db or backend.app.db was already imported prior to conftest load, rebind engine
for mod_name in ("backend.app.db", "app.db"):
    if mod_name in sys.modules:
        db_mod = sys.modules[mod_name]
        db_mod._DB_PATH = _TEST_DB_PATH
        db_mod._DB_URL = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
        db_mod.engine = create_async_engine(db_mod._DB_URL, echo=False)
        db_mod.async_session = async_sessionmaker(
            db_mod.engine, class_=AsyncSession, expire_on_commit=False
        )


@pytest.fixture(autouse=True, scope="session")
def enforce_test_db_isolation():
    """Verify before and after the test session that app.db is never targeted."""
    # Pre-test verification
    active_path = Path(os.environ.get("DATABASE_PATH", ""))
    assert active_path.name == "test_app.db", (
        f"CRITICAL SAFETY VIOLATION: Test suite is configured to target {active_path} instead of test_app.db!"
    )
    assert "app.db" != active_path.name, "CRITICAL SAFETY VIOLATION: Target is production app.db!"

    yield

    # Post-test verification
    active_path = Path(os.environ.get("DATABASE_PATH", ""))
    assert active_path.name == "test_app.db"


@pytest.fixture(autouse=True)
def clean_test_database_tables():
    """Clean test tables in test_app.db between tests if backend db is loaded."""
    yield
    # Only clean if app.db has been imported and initialized in the test session
    db_mod = sys.modules.get("backend.app.db") or sys.modules.get("app.db")
    if db_mod is not None and hasattr(db_mod, "async_session"):
        from sqlalchemy import delete
        async def _cleanup():
            async with db_mod.async_session() as session:
                for row_cls in (
                    db_mod.ConsentRecordRow,
                    db_mod.CellDismissalStateRow,
                    db_mod.PDRStateRow,
                    db_mod.AlertRecordRow,
                ):
                    try:
                        await session.execute(delete(row_cls))
                    except Exception:
                        pass
                await session.commit()
        try:
            asyncio.run(_cleanup())
        except Exception:
            pass

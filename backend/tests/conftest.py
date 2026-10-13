"""Pytest session configuration ensuring test database isolation from live development database."""
import os
from pathlib import Path

# Set test database path before app.db module is imported anywhere in the test session
_BACKEND_DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
_TEST_DB_PATH = _BACKEND_DATA_DIR / "test_app.db"

os.environ["DATABASE_PATH"] = str(_TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"

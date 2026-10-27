"""Tests for the Model Comparison API endpoint."""

from pathlib import Path
import sys

# Ensure backend directory is in sys.path when run from repo root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_model_comparison_success(client):
    """Verify that GET /api/models/comparison returns 200 with valid schema."""
    response = client.get("/api/models/comparison")
    assert response.status_code == 200

    data = response.json()
    assert data["production_default"] == "xgboost_improved"
    assert "horizons" in data
    assert data["horizons"] == ["120s", "360s", "600s", "840s"]

    # Check models list
    models = data.get("models", [])
    assert len(models) == 3
    model_ids = [m["id"] for m in models]
    assert "xgboost_improved" in model_ids
    assert "random_forest" in model_ids
    assert "lightgbm" in model_ids

    # Production default check
    xgb_entry = next(m for m in models if m["id"] == "xgboost_improved")
    assert xgb_entry["is_production"] is True

    # Test evaluations check
    test_evals = data.get("test_evaluations", {})
    for m_id in model_ids:
        assert m_id in test_evals
        for h in ["120s", "360s", "600s", "840s"]:
            assert h in test_evals[m_id]
            h_data = test_evals[m_id][h]
            assert "baseline" in h_data
            assert "revised" in h_data
            assert "delta" in h_data

            # Check required metrics
            rev = h_data["revised"]
            assert "accuracy" in rev
            assert "balanced_accuracy" in rev
            assert "precision" in rev
            assert "recall" in rev
            assert "macro_f1" in rev
            assert "false_positive_rate" in rev
            assert "specificity" in rev
            assert "confusion_matrix" in rev

            # Verify FPR is reduced in revised
            delta = h_data["delta"]
            assert delta["delta_fp"] < 0
            assert delta["delta_fpr"] < 0
            assert delta["delta_balanced_accuracy"] > 0

    # Dataset info check
    ds_info = data.get("dataset_info", {})
    assert "evaluation_split" in ds_info
    assert "disclaimer" in ds_info
    assert ds_info["feature_count"] == 72

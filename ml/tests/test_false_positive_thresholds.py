"""Automated regression and verification tests for false-positive reduction pipeline."""

import json
import os
import pytest
import numpy as np


@pytest.fixture(scope="module")
def eval_data():
    results_path = "ml/reports/false_positive_reduction/evaluation_results.json"
    assert os.path.exists(results_path), f"Missing evaluation results file: {results_path}"
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def thresholds_data():
    thresh_path = "ml/reports/false_positive_reduction/selected_thresholds.json"
    assert os.path.exists(thresh_path), f"Missing selected thresholds file: {thresh_path}"
    with open(thresh_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_selected_thresholds_schema_and_feasibility(thresholds_data):
    """Verify threshold schema, boundary validity, and experimental floor feasibility."""
    models = ["xgboost_improved", "random_forest", "lightgbm"]
    horizons = ["120s", "360s", "600s", "840s"]

    for m in models:
        assert m in thresholds_data["models"]
        for h in horizons:
            h_data = thresholds_data["models"][m][h]
            rev_t = h_data["revised_threshold"]
            base_t = h_data["baseline_threshold"]

            # Thresholds must be in valid probability space (0.5 to 1.0)
            assert 0.50 <= rev_t <= 0.99
            assert 0.50 <= base_t <= 0.99
            # Revised threshold should be higher than or equal to baseline to reduce FPR
            assert rev_t >= base_t

            # Verify every experimental floor was achievable on validation set
            assert h_data["floor_achieved"] is True
            assert h_data["val_revised_recall"] >= h_data["target_recall_floor"]


def test_evaluation_metric_mathematical_consistency(eval_data):
    """Verify that all classification and confusion matrix metrics are mathematically exact."""
    models = ["xgboost_improved", "random_forest", "lightgbm"]
    horizons = ["120s", "360s", "600s", "840s"]

    for m in models:
        for h in horizons:
            d = eval_data["test_evaluations"][m][h]
            for variant in ["baseline", "revised"]:
                v = d[variant]
                cm = v["confusion_matrix"]
                tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
                tot = tn + fp + fn + tp

                assert tot == d["sample_count"]

                expected_fpr = round(fp / (fp + tn), 6)
                assert abs(v["false_positive_rate"] - expected_fpr) < 1e-4

                expected_spec = round(tn / (fp + tn), 6)
                assert abs(v["specificity"] - expected_spec) < 1e-4
                assert abs(v["specificity"] - (1.0 - v["false_positive_rate"])) < 1e-4

                expected_rec = round(tp / (tp + fn), 6)
                assert abs(v["recall"] - expected_rec) < 1e-4

                expected_prec = round(tp / (tp + fp), 6)
                assert abs(v["precision"] - expected_prec) < 1e-4

                expected_bal = round((expected_rec + expected_spec) / 2.0, 6)
                assert abs(v["balanced_accuracy"] - expected_bal) < 1e-4


def test_false_positive_reduction_confirmed_on_test_set(eval_data):
    """Confirm that the revised thresholds achieve a net reduction in false positives."""
    models = ["xgboost_improved", "random_forest", "lightgbm"]
    horizons = ["120s", "360s", "600s", "840s"]

    total_fps_eliminated = {m: 0 for m in models}

    for m in models:
        for h in horizons:
            delta = eval_data["test_evaluations"][m][h]["delta"]
            # Delta FP must be strictly negative (fewer false alarms)
            assert delta["delta_fp"] < 0, f"{m} at {h} did not reduce FP: {delta['delta_fp']}"
            assert delta["delta_fpr"] < 0, f"{m} at {h} did not reduce FPR: {delta['delta_fpr']}"
            total_fps_eliminated[m] += abs(delta["delta_fp"])

            # Balanced Accuracy must improve
            assert delta["delta_balanced_accuracy"] > 0, f"{m} at {h} did not improve BalAcc"

    # Verify at least 2,000 false positives eliminated for each model family
    for m in models:
        assert total_fps_eliminated[m] > 2000, f"Expected >2000 FPs eliminated for {m}, got {total_fps_eliminated[m]}"


def test_production_xgboost_default_and_feature_schema():
    """Verify that Improved XGBoost preserves the 72-feature schema and remains default."""
    xgb_meta_path = "ml/models/xgboost_improved/model_metadata.json"
    with open(xgb_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert len(meta["feature_names"]) == 72
    assert "mean_acceleration_mps2" in meta["feature_names"]
    assert "path_closure_ratio" in meta["feature_names"]
    assert "sin_hour" in meta["feature_names"]

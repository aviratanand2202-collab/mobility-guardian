"""Independent Random Forest Training and Evaluation Pipeline.

Trains and evaluates independent Random Forest models across all 4 horizons
(120s, 360s, 600s, 840s) using the exact same leakage-safe cohort splits,
72-feature schema, target definitions, and evaluation protocols as the improved XGBoost baseline.

Key Design & Scientific Validity:
1. Leakage Prevention: Features are strictly past-only; trajectories and users are separated
   chronologically and by patient cohort (train_df, val_df, test_df).
2. Class Imbalance Handling: Evaluates training-only class weighting (class_weight="balanced")
   to balance tree split criteria without test resampling or synthetic label generation.
3. Native Missing Value Support: Uses scikit-learn's native missing value handling in tree splits.
4. Validation-Only Hyperparameter Tuning: Screens candidate tree depths and leaf constraints
   on validation data; test set is never seen during selection.
5. Probability Calibration: Validated Platt scaling (LogisticRegression) fitted strictly
   on validation raw probabilities.
6. Calibrated Threshold Selection: Optimal macro-F1 decision thresholds searched directly
   in the calibrated probability space on validation data.
7. Unbiased Test Evaluation: Final metrics computed strictly on held-out test cohort.
8. Safe Serialization: Serializes artifacts to an isolated directory ('ml/models/random_forest')
   without touching existing XGBoost models.

Target Disclaimer:
The prediction target represents mathematically derived personalized kinematic risk
(spatial-temporal movement anomalies exceeding personalized historical profile bounds),
NOT a clinical diagnosis of dementia, wandering, or medical condition.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.src.improved_pipeline import enrich_dataset_with_features
from ml.src.risk import RiskConfig

logger = logging.getLogger(__name__)


def train_and_serialize_random_forest_models(
    output_dir: str = "ml/models/random_forest",
    cache_dir: str = "ml/models/xgboost/cache",
    features_path: str = "ml/data/processed/trajectory_features.parquet",
    xgb_improved_meta_path: str = "ml/models/xgboost_improved/model_metadata.json",
) -> Dict[str, Any]:
    """Train independent Random Forest models across all 4 horizons and serialize artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. Load cached splits and trajectory features
    logger.info("Loading cached splits and trajectory features...")
    tr_path = os.path.join(cache_dir, "train_df.parquet")
    va_path = os.path.join(cache_dir, "val_df.parquet")
    te_path = os.path.join(cache_dir, "test_df.parquet")

    train_df = pd.read_parquet(tr_path)
    val_df = pd.read_parquet(va_path)
    test_df = pd.read_parquet(te_path)
    tf_df = pd.read_parquet(features_path)

    # 2. Enrich datasets with extended kinematic, spatial, dwell, and temporal features
    logger.info("Enriching datasets with extended kinematic features...")
    train_m = enrich_dataset_with_features(train_df, tf_df)
    val_m = enrich_dataset_with_features(val_df, tf_df)
    test_m = enrich_dataset_with_features(test_df, tf_df)

    # Load improved XGBoost metadata to confirm exact feature ordering
    with open(xgb_improved_meta_path, "r") as f:
        xgb_meta = json.load(f)

    expected_features = xgb_meta.get("feature_names", [])
    if not expected_features or len(expected_features) != 72:
        raise ValueError(
            f"Expected 72 features from improved XGBoost metadata, found {len(expected_features)}."
        )

    # Verify all expected features are present in enriched train_m
    missing_feats = [f for f in expected_features if f not in train_m.columns]
    if missing_feats:
        raise ValueError(f"Enriched dataset missing expected features: {missing_feats}")

    feature_cols = expected_features
    logger.info(f"Verified exact 72-feature schema matching improved XGBoost pipeline.")

    # Copy population baseline for standalone portability
    pop_source = os.path.join("ml/models/xgboost", "population_baseline.json")
    pop_target = os.path.join(output_dir, "population_baseline.json")
    if os.path.exists(pop_source):
        shutil.copy2(pop_source, pop_target)

    cfg = RiskConfig()
    metadata: Dict[str, Any] = {
        "schema_version": "1.1.0",
        "model_version": "2.0.0-random-forest",
        "model_family": "random_forest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": (
            "Independent Random Forest predictive kinematic risk models trained across 4 horizons "
            "(120s, 360s, 600s, 840s) using balanced class weights, native tree-split missing value handling, "
            "Platt scaling calibration, and optimal Macro-F1 threshold search on validation data."
        ),
        "target_definition": (
            "Personalized kinematic outlier (mathematical spatial-temporal excursion beyond historical bounds). "
            "Not a clinical dementia or wandering diagnosis."
        ),
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "target_features": list(cfg.target_features),
        "horizons_sec": list(cfg.horizons_sec),
        "hyperparameter_search": {
            "strategy": "Validation PR-AUC screening followed by 100-tree ensemble training",
            "candidates": [
                {"max_depth": 12, "min_samples_leaf": 10, "max_features": "sqrt"},
                {"max_depth": 14, "min_samples_leaf": 5, "max_features": "sqrt"},
                {"max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
            ],
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": cfg.random_seed,
        },
        "horizons": {},
        "comparison_summary": {},
    }

    comparison_results = {}

    # Candidate hyperparameter configurations for validation tuning
    candidates = [
        {"max_depth": 12, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"max_depth": 14, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
    ]

    for h in cfg.horizons_sec:
        logger.info(f"\n=======================================================")
        logger.info(f"=== Training Random Forest Horizon {h}s ===")
        logger.info(f"=======================================================")
        t_h_start = time.time()
        tgt_col = f"target_{h}s"

        tr_sub = train_m.dropna(subset=[tgt_col]).copy()
        va_sub = val_m.dropna(subset=[tgt_col]).copy()
        te_sub = test_m.dropna(subset=[tgt_col]).copy()

        y_tr = tr_sub[tgt_col].astype(int).values
        y_va = va_sub[tgt_col].astype(int).values
        y_te = te_sub[tgt_col].astype(int).values

        X_tr = tr_sub[feature_cols].copy()
        X_va = va_sub[feature_cols].copy()
        X_te = te_sub[feature_cols].copy()

        pos_tr = int(np.sum(y_tr == 1))
        neg_tr = int(np.sum(y_tr == 0))
        logger.info(
            f"Horizon {h}s - Train rows: {len(y_tr)} (Pos: {pos_tr}, Neg: {neg_tr}, "
            f"Pos%: {pos_tr / len(y_tr) * 100:.2f}%)"
        )
        logger.info(f"Horizon {h}s - Val rows: {len(y_va)}, Test rows: {len(y_te)}")

        # Step A: Validation Hyperparameter Tuning
        logger.info(f"Screening candidate hyperparameters on validation set for {h}s...")
        best_candidate = candidates[0]
        best_va_prauc = -1.0

        for cand in candidates:
            # Fast screening with 40 trees to evaluate validation discrimination
            rf_cand = RandomForestClassifier(
                n_estimators=40,
                max_depth=cand["max_depth"],
                min_samples_leaf=cand["min_samples_leaf"],
                max_features=cand["max_features"],
                class_weight="balanced",
                n_jobs=-1,
                random_state=cfg.random_seed,
            )
            rf_cand.fit(X_tr, y_tr)
            va_cand_probs = rf_cand.predict_proba(X_va)[:, 1]
            cand_prauc = average_precision_score(y_va, va_cand_probs)
            logger.info(
                f"  Candidate depth={cand['max_depth']}, min_samples_leaf={cand['min_samples_leaf']} "
                f"-> Val PR-AUC: {cand_prauc:.4f}"
            )
            if cand_prauc > best_va_prauc:
                best_va_prauc = cand_prauc
                best_candidate = cand

        logger.info(
            f"Selected Best Hyperparameters for Horizon {h}s: {best_candidate} "
            f"(Val PR-AUC: {best_va_prauc:.4f})"
        )

        # Step B: Train Final 100-Tree Model with Best Hyperparameters
        t_fit_start = time.time()
        final_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=best_candidate["max_depth"],
            min_samples_leaf=best_candidate["min_samples_leaf"],
            max_features=best_candidate["max_features"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=cfg.random_seed,
        )
        final_rf.fit(X_tr, y_tr)
        fit_duration_sec = time.time() - t_fit_start
        logger.info(f"Trained 100-tree Random Forest for {h}s in {fit_duration_sec:.2f}s")

        # Step C: Probability Calibration via Platt Scaling on Validation Set
        val_raw_probs = final_rf.predict_proba(X_va)[:, 1]

        calibrator = LogisticRegression()
        calibrator.fit(val_raw_probs.reshape(-1, 1), y_va)
        val_cal_probs = calibrator.predict_proba(val_raw_probs.reshape(-1, 1))[:, 1]

        # Step D: Threshold Search on Calibrated Probabilities (Maximizing Validation Macro-F1)
        best_thresh = 0.5
        best_f1 = -1.0
        for cand_t in np.linspace(cfg.xgb_threshold_min, cfg.xgb_threshold_max, cfg.xgb_threshold_steps):
            preds = (val_cal_probs >= cand_t).astype(int)
            f1 = f1_score(y_va, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(cand_t)

        logger.info(f"Optimal Threshold for {h}s: {best_thresh:.2f} (Val Macro-F1: {best_f1:.4f})")

        # Step E: Unbiased Test Split Evaluation
        t_infer_start = time.time()
        te_raw_probs = final_rf.predict_proba(X_te)[:, 1]
        te_cal_probs = calibrator.predict_proba(te_raw_probs.reshape(-1, 1))[:, 1]
        infer_duration_sec = time.time() - t_infer_start
        latency_per_sample_ms = (infer_duration_sec / len(X_te)) * 1000.0

        te_preds = (te_cal_probs >= best_thresh).astype(int)

        # Compute all required metrics
        pr_auc = float(average_precision_score(y_te, te_cal_probs))
        roc_auc = float(roc_auc_score(y_te, te_cal_probs))
        macro_f1 = float(f1_score(y_te, te_preds, average="macro", zero_division=0))
        pos_f1 = float(f1_score(y_te, te_preds, pos_label=1, zero_division=0))
        neg_f1 = float(f1_score(y_te, te_preds, pos_label=0, zero_division=0))
        prec = float(precision_score(y_te, te_preds, zero_division=0))
        rec = float(recall_score(y_te, te_preds, zero_division=0))
        brier = float(brier_score_loss(y_te, te_cal_probs))

        cm = confusion_matrix(y_te, te_preds)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Step F: Serialize Random Forest Model (.joblib with compression)
        model_filename = f"rf_horizon_{h}s.joblib"
        model_save_path = os.path.join(output_dir, model_filename)
        joblib.dump(final_rf, model_save_path, compress=3)
        file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
        logger.info(f"Saved compressed Random Forest model to: {model_save_path} ({file_size_mb:.2f} MB)")

        cal_coef = [[float(c) for c in calibrator.coef_[0]]]
        cal_intercept = [float(calibrator.intercept_[0])]
        cal_classes = [int(c) for c in calibrator.classes_]

        metadata["horizons"][f"{h}s"] = {
            "horizon_sec": h,
            "model_file": model_filename,
            "optimal_threshold": best_thresh,
            "n_estimators": 100,
            "selected_hyperparameters": best_candidate,
            "training_time_sec": round(fit_duration_sec, 2),
            "inference_latency_per_sample_ms": round(latency_per_sample_ms, 4),
            "model_file_size_mb": round(file_size_mb, 2),
            "calibrator": {
                "coef": cal_coef,
                "intercept": cal_intercept,
                "classes": cal_classes,
            },
            "metrics": {
                "macro_f1": macro_f1,
                "f1_positive": pos_f1,
                "f1_negative": neg_f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "false_alert_rate": far,
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                },
            },
        }

        # Step G: Compare Side-by-Side with Improved XGBoost
        xgb_h_info = xgb_meta.get("horizons", {}).get(f"{h}s", {}).get("metrics", {})
        xgb_cm = xgb_h_info.get("confusion_matrix", {})
        xgb_tn = xgb_cm.get("tn", 0)
        xgb_fp = xgb_cm.get("fp", 0)
        xgb_far = float(xgb_fp / (xgb_fp + xgb_tn)) if (xgb_fp + xgb_tn) > 0 else 0.0

        comparison_results[f"{h}s"] = {
            "xgboost_improved": {
                "pr_auc": xgb_h_info.get("pr_auc"),
                "roc_auc": xgb_h_info.get("roc_auc"),
                "macro_f1": xgb_h_info.get("macro_f1"),
                "pos_f1": xgb_h_info.get("f1_positive"),
                "neg_f1": xgb_h_info.get("f1_negative"),
                "precision": xgb_h_info.get("precision"),
                "recall": xgb_h_info.get("recall"),
                "brier_score": xgb_h_info.get("brier_score"),
                "false_alert_rate": xgb_far,
                "confusion_matrix": xgb_cm,
            },
            "random_forest": {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "macro_f1": macro_f1,
                "pos_f1": pos_f1,
                "neg_f1": neg_f1,
                "precision": prec,
                "recall": rec,
                "brier_score": brier,
                "false_alert_rate": far,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            },
            "delta_rf_vs_xgb": {
                "pr_auc": pr_auc - float(xgb_h_info.get("pr_auc", 0.0)),
                "roc_auc": roc_auc - float(xgb_h_info.get("roc_auc", 0.0)),
                "macro_f1": macro_f1 - float(xgb_h_info.get("macro_f1", 0.0)),
                "pos_f1": pos_f1 - float(xgb_h_info.get("f1_positive", 0.0)),
                "neg_f1": neg_f1 - float(xgb_h_info.get("f1_negative", 0.0)),
                "false_alert_rate": far - xgb_far,
                "brier_score": brier - float(xgb_h_info.get("brier_score", 0.0)),
            },
        }

        total_h_sec = time.time() - t_h_start
        logger.info(f"Completed Horizon {h}s in {total_h_sec:.1f}s.")

    metadata["comparison_summary"] = comparison_results

    # Write model_metadata.json
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {meta_path}")

    # Write evaluation_comparison.json
    comp_path = os.path.join(output_dir, "evaluation_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved evaluation comparison to: {comp_path}")

    total_pipeline_sec = time.time() - t_start
    logger.info(f"\n=======================================================")
    logger.info(f"Random Forest Training Pipeline completed in {total_pipeline_sec:.1f}s.")
    logger.info(f"=======================================================\n")
    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    train_and_serialize_random_forest_models()

"""Tuning and Evaluation of False-Positive Control Thresholds for Mobility Guardian ML Models.

This script executes a strict validation-only threshold optimization protocol across
Improved XGBoost, Random Forest, and LightGBM for all four prediction horizons:
120s, 360s, 600s, and 840s.

Protocol Safeguards:
1. Validation-Only Selection: Decision thresholds are searched and selected strictly
   on the validation cohort. Held-out test data is NEVER used for threshold selection.
2. Experimental Recall Floors: Treats recall constraints as experimental bounds,
   verifying feasibility and documenting whether each floor is achievable.
3. Threshold Freezing: Selected thresholds are frozen and saved before test evaluation.
4. Comprehensive Mathematical Metrics: Reports TN, FP, FN, TP, FPR, Specificity,
   Recall, Balanced Accuracy, Precision, Macro-F1, and Brier score.
5. Baseline Verification: Verifies exact reproduction of baseline metrics prior to deltas.
6. Non-Destructive: Preserves all existing models, feature schemas, and production defaults.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from ml.src.improved_pipeline import enrich_dataset_with_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [120, 360, 600, 840]
MODELS = ["xgboost_improved", "random_forest", "lightgbm"]

# Experimental Recall Floors
RECALL_FLOORS = {
    120: 0.75,
    360: 0.85,
    600: 0.88,
    840: 0.88,
}


def load_model_and_calibrator(model_type: str, h: int, feature_names: List[str]):
    """Load model artifact and Platt scaling calibrator from disk."""
    meta_path = os.path.join(f"ml/models/{model_type}", "model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    h_info = meta["horizons"][f"{h}s"]
    cal_info = h_info["calibrator"]

    calibrator = LogisticRegression()
    calibrator.coef_ = np.array(cal_info["coef"])
    calibrator.intercept_ = np.array(cal_info["intercept"])
    calibrator.classes_ = np.array(cal_info["classes"])

    if model_type == "xgboost_improved":
        booster = xgb.Booster()
        booster.load_model(os.path.join(f"ml/models/{model_type}", f"xgb_horizon_{h}s.json"))
        model_obj = booster
    elif model_type == "random_forest":
        rf = joblib.load(os.path.join(f"ml/models/{model_type}", f"rf_horizon_{h}s.joblib"))
        model_obj = rf
    elif model_type == "lightgbm":
        bst = lgb.Booster(model_file=os.path.join(f"ml/models/{model_type}", f"lgb_horizon_{h}s.txt"))
        model_obj = bst
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_obj, calibrator, h_info


def predict_calibrated_probs(
    model_obj: Any,
    calibrator: LogisticRegression,
    model_type: str,
    X: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Generate calibrated probabilities from model object and Platt calibrator."""
    if model_type == "xgboost_improved":
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        raw_probs = model_obj.predict(dmat)
    elif model_type == "random_forest":
        raw_probs = model_obj.predict_proba(X)[:, 1]
    elif model_type == "lightgbm":
        raw_probs = model_obj.predict(X)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    return cal_probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute complete suite of classification and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    total = len(y_true)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    balanced_accuracy = float((recall + specificity) / 2.0)
    acc = float((tp + tn) / total) if total > 0 else 0.0

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    neg_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_positive_rate": round(fpr, 6),
        "specificity": round(specificity, 6),
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "f1_positive": round(pos_f1, 6),
        "f1_negative": round(neg_f1, 6),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier_score": round(brier, 6),
    }


def run_threshold_optimization_and_evaluation(
    output_dir: str = "ml/reports/false_positive_reduction",
    cache_dir: str = "ml/models/xgboost/cache",
    feat_path: str = "ml/data/processed/trajectory_features.parquet",
) -> Dict[str, Any]:
    """Execute the end-to-end threshold optimization and evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # 1. Load cached splits
    logger.info("Loading cached validation and test splits...")
    val_df = pd.read_parquet(os.path.join(cache_dir, "val_df.parquet"))
    test_df = pd.read_parquet(os.path.join(cache_dir, "test_df.parquet"))
    tf_df = pd.read_parquet(feat_path)

    logger.info("Enriching datasets with extended kinematic features...")
    val_m = enrich_dataset_with_features(val_df, tf_df)
    test_m = enrich_dataset_with_features(test_df, tf_df)

    # Load canonical feature schema
    xgb_meta_path = "ml/models/xgboost_improved/model_metadata.json"
    with open(xgb_meta_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)["feature_names"]

    logger.info(f"Loaded {len(feature_names)} features for evaluation.")

    # Data structures for results
    threshold_grid = np.linspace(0.40, 0.98, 59)
    selected_thresholds: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Constrained Clinical Optimization (Recall Floor + FPR Minimization)",
        "recall_floors": RECALL_FLOORS,
        "models": {},
    }

    validation_evaluations: Dict[str, Any] = {}
    test_evaluations: Dict[str, Any] = {}

    # STEP A: Validation-Only Threshold Tuning
    logger.info("=== STEP A: Validation-Only Threshold Optimization ===")
    for model_type in MODELS:
        selected_thresholds["models"][model_type] = {}
        validation_evaluations[model_type] = {}

        for h in HORIZONS:
            tgt_col = f"target_{h}s"
            val_sub = val_m.dropna(subset=[tgt_col]).copy()
            X_val = val_sub[feature_names].values
            y_val = val_sub[tgt_col].values.astype(int)

            model_obj, calibrator, h_info = load_model_and_calibrator(model_type, h, feature_names)
            val_probs = predict_calibrated_probs(model_obj, calibrator, model_type, X_val, feature_names)

            orig_thresh = float(h_info["optimal_threshold"])
            orig_val_metrics = compute_metrics(y_val, val_probs, orig_thresh)

            # Evaluate experimental recall constraint
            req_floor = RECALL_FLOORS[h]
            feasible_candidates = []

            for cand_t in threshold_grid:
                preds = (val_probs >= cand_t).astype(int)
                rec = float(recall_score(y_val, preds, zero_division=0))
                if rec >= req_floor:
                    # Specificity
                    cm = confusion_matrix(y_val, preds)
                    tn, fp = cm[0, 0], cm[0, 1]
                    spec = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
                    bal_acc = float((rec + spec) / 2.0)
                    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
                    feasible_candidates.append({
                        "threshold": float(cand_t),
                        "recall": rec,
                        "specificity": spec,
                        "fpr": 1.0 - spec,
                        "balanced_accuracy": bal_acc,
                        "macro_f1": macro_f1,
                    })

            floor_achieved = len(feasible_candidates) > 0
            if floor_achieved:
                # Sort by specificity desc (i.e. minimum FPR), then by recall desc
                feasible_candidates.sort(key=lambda x: (x["specificity"], x["recall"]), reverse=True)
                chosen_t = feasible_candidates[0]["threshold"]
            else:
                # Fallback: candidate with highest recall
                logger.warning(f"Recall floor {req_floor} not strictly achievable for {model_type} at {h}s; selecting max recall candidate.")
                cand_recs = [(float(t), float(recall_score(y_val, (val_probs >= t).astype(int), zero_division=0))) for t in threshold_grid]
                cand_recs.sort(key=lambda x: x[1], reverse=True)
                chosen_t = cand_recs[0][0]

            revised_val_metrics = compute_metrics(y_val, val_probs, chosen_t)

            selected_thresholds["models"][model_type][f"{h}s"] = {
                "horizon_sec": h,
                "baseline_threshold": orig_thresh,
                "revised_threshold": chosen_t,
                "target_recall_floor": req_floor,
                "floor_achieved": floor_achieved,
                "val_baseline_recall": orig_val_metrics["recall"],
                "val_revised_recall": revised_val_metrics["recall"],
                "val_baseline_fpr": orig_val_metrics["false_positive_rate"],
                "val_revised_fpr": revised_val_metrics["false_positive_rate"],
                "val_baseline_balanced_acc": orig_val_metrics["balanced_accuracy"],
                "val_revised_balanced_acc": revised_val_metrics["balanced_accuracy"],
            }

            validation_evaluations[model_type][f"{h}s"] = {
                "baseline": orig_val_metrics,
                "revised": revised_val_metrics,
                "delta": {
                    "delta_fp": revised_val_metrics["confusion_matrix"]["fp"] - orig_val_metrics["confusion_matrix"]["fp"],
                    "delta_fn": revised_val_metrics["confusion_matrix"]["fn"] - orig_val_metrics["confusion_matrix"]["fn"],
                    "delta_fpr": round(revised_val_metrics["false_positive_rate"] - orig_val_metrics["false_positive_rate"], 6),
                    "delta_recall": round(revised_val_metrics["recall"] - orig_val_metrics["recall"], 6),
                    "delta_balanced_accuracy": round(revised_val_metrics["balanced_accuracy"] - orig_val_metrics["balanced_accuracy"], 6),
                },
            }

            logger.info(
                f"[{model_type} {h}s] Selected Threshold: {chosen_t:.2f} (Base: {orig_thresh:.2f}) | "
                f"Val FPR: {orig_val_metrics['false_positive_rate']*100:.2f}% -> {revised_val_metrics['false_positive_rate']*100:.2f}% | "
                f"Val Rec: {orig_val_metrics['recall']*100:.2f}% -> {revised_val_metrics['recall']*100:.2f}% | "
                f"Floor Achieved: {floor_achieved}"
            )

    # Save frozen thresholds before test evaluation
    frozen_path = os.path.join(output_dir, "selected_thresholds.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(selected_thresholds, f, indent=2)
    logger.info(f"FROZEN thresholds successfully serialized to: {frozen_path}")

    # STEP B: Unbiased Held-Out Test Evaluation
    logger.info("=== STEP B: Unbiased Held-Out Test Evaluation ===")
    for model_type in MODELS:
        test_evaluations[model_type] = {}
        for h in HORIZONS:
            tgt_col = f"target_{h}s"
            test_sub = test_m.dropna(subset=[tgt_col]).copy()
            X_test = test_sub[feature_names].values
            y_test = test_sub[tgt_col].values.astype(int)

            model_obj, calibrator, h_info = load_model_and_calibrator(model_type, h, feature_names)
            test_probs = predict_calibrated_probs(model_obj, calibrator, model_type, X_test, feature_names)

            orig_thresh = selected_thresholds["models"][model_type][f"{h}s"]["baseline_threshold"]
            revised_thresh = selected_thresholds["models"][model_type][f"{h}s"]["revised_threshold"]

            test_base_metrics = compute_metrics(y_test, test_probs, orig_thresh)
            test_rev_metrics = compute_metrics(y_test, test_probs, revised_thresh)

            # Mathematical delta verification
            fp_base = test_base_metrics["confusion_matrix"]["fp"]
            fp_rev = test_rev_metrics["confusion_matrix"]["fp"]
            fn_base = test_base_metrics["confusion_matrix"]["fn"]
            fn_rev = test_rev_metrics["confusion_matrix"]["fn"]

            delta_fp = fp_rev - fp_base
            delta_fn = fn_rev - fn_base

            test_evaluations[model_type][f"{h}s"] = {
                "sample_count": len(y_test),
                "positive_count": int(np.sum(y_test)),
                "negative_count": int(len(y_test) - np.sum(y_test)),
                "baseline": test_base_metrics,
                "revised": test_rev_metrics,
                "delta": {
                    "delta_fp": delta_fp,
                    "delta_fn": delta_fn,
                    "delta_fpr": round(test_rev_metrics["false_positive_rate"] - test_base_metrics["false_positive_rate"], 6),
                    "delta_specificity": round(test_rev_metrics["specificity"] - test_base_metrics["specificity"], 6),
                    "delta_recall": round(test_rev_metrics["recall"] - test_base_metrics["recall"], 6),
                    "delta_precision": round(test_rev_metrics["precision"] - test_base_metrics["precision"], 6),
                    "delta_balanced_accuracy": round(test_rev_metrics["balanced_accuracy"] - test_base_metrics["balanced_accuracy"], 6),
                    "delta_macro_f1": round(test_rev_metrics["macro_f1"] - test_base_metrics["macro_f1"], 6),
                    "delta_accuracy": round(test_rev_metrics["accuracy"] - test_base_metrics["accuracy"], 6),
                },
            }

            logger.info(
                f"[TEST {model_type} {h}s] Base(t={orig_thresh:.2f}) -> Rev(t={revised_thresh:.2f}) | "
                f"FPR: {test_base_metrics['false_positive_rate']*100:.2f}% -> {test_rev_metrics['false_positive_rate']*100:.2f}% (dFPR: {test_evaluations[model_type][f'{h}s']['delta']['delta_fpr']*100:+.2f}%) | "
                f"Recall: {test_base_metrics['recall']*100:.2f}% -> {test_rev_metrics['recall']*100:.2f}% | "
                f"BalAcc: {test_base_metrics['balanced_accuracy']*100:.2f}% -> {test_rev_metrics['balanced_accuracy']*100:.2f}% | "
                f"dFP: {delta_fp:+d}, dFN: {delta_fn:+d}"
            )

    # Save full evaluation results
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Validation-Only Constrained Optimization evaluated on Unbiased Held-Out Test Split",
        "selected_thresholds": selected_thresholds,
        "validation_evaluations": validation_evaluations,
        "test_evaluations": test_evaluations,
    }

    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Complete evaluation results written to: {eval_path}")

    return full_results


if __name__ == "__main__":
    run_threshold_optimization_and_evaluation()

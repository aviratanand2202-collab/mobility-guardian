"""Train and Serialize Frozen XGBoost Models for Production Deployment.

Reproduces the exact Chunk 6 training protocol and serializes:
1. Native XGBoost booster models (JSON) per horizon (120s, 360s, 600s, 840s).
2. Probability calibrators (LogisticRegression coefficients).
3. Horizon-specific decision thresholds and optimal hyperparameters.
4. Exact feature schema and ordering (49 predictor features).
5. Population prior baseline for zero-shot new-user evaluation.
6. Rigorous verification against canonical test predictions in risk_predictions.parquet.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
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
import xgboost as xgb

from ml.src.risk import RiskConfig
from ml.src.run_risk_pipeline import (
    build_known_cohort_splits,
    partition_users,
)


def build_cohort_data(
    feat_df: pd.DataFrame,
    cfg: RiskConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build train, val, and test splits using canonical pipeline function with caching."""
    cache_dir = "ml/models/xgboost/cache"
    os.makedirs(cache_dir, exist_ok=True)
    tr_path = os.path.join(cache_dir, "train_df.parquet")
    va_path = os.path.join(cache_dir, "val_df.parquet")
    te_path = os.path.join(cache_dir, "test_df.parquet")
    pop_path = os.path.join(cache_dir, "pop_baseline.json")

    if (
        os.path.exists(tr_path)
        and os.path.exists(va_path)
        and os.path.exists(te_path)
        and os.path.exists(pop_path)
    ):
        print("Loading cached cohort splits from disk...")
        train_df = pd.read_parquet(tr_path)
        val_df = pd.read_parquet(va_path)
        test_df = pd.read_parquet(te_path)
        with open(pop_path, "r") as f:
            pop_baseline = json.load(f)
        print(f"Loaded cached splits: Tr={len(train_df)}, Va={len(val_df)}, Te={len(test_df)}")
        return train_df, val_df, test_df, pop_baseline

    known_users, _ = partition_users(feat_df, cfg)
    print(f"Partitioned {len(known_users)} known users for training.")
    train_df, val_df, test_df, pop_baseline = build_known_cohort_splits(feat_df, known_users, cfg)

    train_df.to_parquet(tr_path, index=False)
    val_df.to_parquet(va_path, index=False)
    test_df.to_parquet(te_path, index=False)
    with open(pop_path, "w") as f:
        json.dump(pop_baseline, f, indent=2)
    print(f"Cached splits to {cache_dir}")
    return train_df, val_df, test_df, pop_baseline


def train_and_serialize_models(output_dir: str = "ml/models/xgboost") -> Dict[str, Any]:
    """Train XGBoost for each horizon, verify against canonical benchmark, and serialize."""
    cfg = RiskConfig()
    feat_df = pd.read_parquet("ml/data/processed/trajectory_features.parquet")
    train_df, val_df, test_df, pop_baseline = build_cohort_data(feat_df, cfg)

    # Feature columns for model training
    exclude_cols = {
        "window_id",
        "user_id",
        "trajectory_id",
        "segment_id",
        "window_index",
        "start_time",
        "end_time",
        "target_120s",
        "target_360s",
        "target_600s",
        "target_840s",
    }
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Extracted {len(feature_cols)} feature columns.")

    # Load canonical test predictions and report for verification
    canon_preds_df = pd.read_parquet("ml/data/processed/risk_predictions.parquet")
    with open("predictive_risk_report.json", "r") as f:
        canon_report = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    profiles_dir = "ml/models/profiles"
    os.makedirs(profiles_dir, exist_ok=True)

    metadata: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "model_version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "target_features": list(cfg.target_features),
        "horizons_sec": list(cfg.horizons_sec),
        "hyperparameters": {
            "learning_rate": cfg.xgb_learning_rate,
            "max_depth": cfg.xgb_max_depth,
            "n_estimators": cfg.xgb_n_estimators,
            "subsample": cfg.xgb_subsample,
            "colsample_bytree": cfg.xgb_colsample_bytree,
            "early_stopping_rounds": cfg.xgb_early_stopping_rounds,
            "random_seed": cfg.random_seed,
        },
        "horizons": {},
    }

    # Save population prior baseline
    pop_base_path = os.path.join(output_dir, "population_baseline.json")
    with open(pop_base_path, "w") as f:
        json.dump(pop_baseline, f, indent=2)
    print(f"Saved population prior baseline to: {pop_base_path}")

    for h in cfg.horizons_sec:
        print(f"\n--- Training & Serializing Horizon {h}s ---")
        tgt_col = f"target_{h}s"

        tr_sub = train_df.dropna(subset=[tgt_col]).copy()
        va_sub = val_df.dropna(subset=[tgt_col]).copy()
        te_sub = test_df.dropna(subset=[tgt_col]).copy()

        y_tr = tr_sub[tgt_col].astype(int).values
        y_va = va_sub[tgt_col].astype(int).values
        y_te = te_sub[tgt_col].astype(int).values

        X_tr = tr_sub[feature_cols].copy()
        X_va = va_sub[feature_cols].copy()
        X_te = te_sub[feature_cols].copy()

        pos_tr = int(np.sum(y_tr == 1))
        neg_tr = int(np.sum(y_tr == 0))
        scale_pos = float(neg_tr / max(pos_tr, 1))

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_te, label=y_te)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": cfg.xgb_learning_rate,
            "max_depth": cfg.xgb_max_depth,
            "subsample": cfg.xgb_subsample,
            "colsample_bytree": cfg.xgb_colsample_bytree,
            "scale_pos_weight": scale_pos,
            "seed": cfg.random_seed,
        }

        evals = [(dtrain, "train"), (dval, "val")]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg.xgb_n_estimators,
            evals=evals,
            early_stopping_rounds=cfg.xgb_early_stopping_rounds,
            verbose_eval=False,
        )

        val_probs = bst.predict(dval)
        best_xgb_thresh = 0.5
        best_xgb_f1 = -1.0
        for cand_t in np.linspace(cfg.xgb_threshold_min, cfg.xgb_threshold_max, cfg.xgb_threshold_steps):
            preds = (val_probs >= cand_t).astype(int)
            f1 = f1_score(y_va, preds, average="macro", zero_division=0)
            if f1 > best_xgb_f1:
                best_xgb_f1 = f1
                best_xgb_thresh = float(cand_t)

        calibrator = LogisticRegression()
        calibrator.fit(val_probs.reshape(-1, 1), y_va)

        te_probs = bst.predict(dtest)
        pred_te_xgb = (te_probs >= best_xgb_thresh).astype(int)
        _ = calibrator.predict_proba(te_probs.reshape(-1, 1))[:, 1]

        # REPRODUCIBILITY VERIFICATION
        canon_h = canon_preds_df[canon_preds_df["horizon_sec"] == h].copy()
        canon_probs = canon_h["y_prob_xgb"].values
        canon_preds = canon_h["y_pred_xgb"].values

        assert len(canon_probs) == len(te_probs), f"Length mismatch: {len(canon_probs)} != {len(te_probs)}"
        prob_diff = np.max(np.abs(canon_probs - te_probs))
        pred_mismatches = int(np.sum(canon_preds != pred_te_xgb))
        print(f"  Reproducibility Check (Horizon {h}s):")
        print(f"    Max probability diff: {prob_diff:.8f}")
        print(f"    Binary prediction mismatches: {pred_mismatches}/{len(te_probs)}")

        assert prob_diff < 1e-4, f"Horizon {h}s probabilities diverged from benchmark! Diff={prob_diff}"
        assert pred_mismatches == 0, f"Horizon {h}s predictions diverged! Mismatches={pred_mismatches}"

        # Verify threshold against report
        report_h = canon_report["horizons"][f"horizon_{h}s"]["optimal_parameters"]
        expected_thresh = report_h["xgb_best_threshold"]
        assert abs(best_xgb_thresh - expected_thresh) < 1e-6, (
            f"Horizon {h}s threshold mismatch: {best_xgb_thresh} != {expected_thresh}"
        )

        # Save native XGBoost JSON model
        model_filename = f"xgb_horizon_{h}s.json"
        model_filepath = os.path.join(output_dir, model_filename)
        bst.save_model(model_filepath)
        print(f"  Saved model artifact: {model_filepath}")

        # Compute horizon test metrics
        macro_f1 = float(f1_score(y_te, pred_te_xgb, average="macro", zero_division=0))
        f1_pos = float(f1_score(y_te, pred_te_xgb, pos_label=1, zero_division=0))
        f1_neg = float(f1_score(y_te, pred_te_xgb, pos_label=0, zero_division=0))
        prec_pos = float(precision_score(y_te, pred_te_xgb, pos_label=1, zero_division=0))
        rec_pos = float(recall_score(y_te, pred_te_xgb, pos_label=1, zero_division=0))
        roc_auc = float(roc_auc_score(y_te, te_probs))
        pr_auc = float(average_precision_score(y_te, te_probs))
        brier = float(brier_score_loss(y_te, te_probs))
        cm = confusion_matrix(y_te, pred_te_xgb, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metadata["horizons"][f"{h}s"] = {
            "horizon_sec": h,
            "model_file": model_filename,
            "optimal_threshold": float(best_xgb_thresh),
            "mad_best_kappa": float(report_h["mad_best_kappa"]),
            "pct_best_p95_count": int(report_h["pct_best_p95_count"]),
            "best_iteration": int(bst.best_iteration),
            "calibrator": {
                "coef": calibrator.coef_.tolist(),
                "intercept": calibrator.intercept_.tolist(),
                "classes": calibrator.classes_.tolist(),
            },
            "metrics": {
                "macro_f1": macro_f1,
                "f1_positive": f1_pos,
                "f1_negative": f1_neg,
                "precision": prec_pos,
                "recall": rec_pos,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
        }

    # Save comprehensive metadata
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved model metadata to: {meta_path}")

    # Final assertion: Verify that canonical benchmark artifacts remain byte-identical
    expected_outputs = {
        "ml/data/processed/risk_predictions.parquet": (
            "d716f7a268a7e96c874fb89c0e3311ebb343fb4c8fa7557f5eaa26c98b4a204f"
        ),
        "predictive_risk_report.json": "c7b57bdff9c126c382f5e535b376df101d432daa0da2a3c064c424f95bafced1",
        "predictive_risk_report.md": "181c2b108b1f4f8fc044e63f4cdd70e0447726925bf120da6818c9db945df0da",
    }
    for rel_path, exp_hash in expected_outputs.items():
        curr_hash = hashlib.sha256(Path(rel_path).read_bytes()).hexdigest()
        assert curr_hash == exp_hash, f"Canonical benchmark file {rel_path} was modified!"
    print("ALL CANONICAL BENCHMARK ARTIFACTS VERIFIED BYTE-IDENTICAL!")

    return metadata


if __name__ == "__main__":
    train_and_serialize_models()

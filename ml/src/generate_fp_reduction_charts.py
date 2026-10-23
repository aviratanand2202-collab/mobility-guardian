"""Generate publication-quality comparison charts for false-positive reduction evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RESULTS_PATH = "ml/reports/false_positive_reduction/evaluation_results.json"
OUTPUT_DIR = "ml/reports/false_positive_reduction/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

test_evals = data["test_evaluations"]
horizons = ["120s", "360s", "600s", "840s"]
models = ["xgboost_improved", "random_forest", "lightgbm"]
model_labels = {
    "xgboost_improved": "Improved XGBoost (Prod)",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}

colors_base = {
    "xgboost_improved": "#64b5f6",
    "random_forest": "#81c784",
    "lightgbm": "#ffb74d",
}
colors_rev = {
    "xgboost_improved": "#1565c0",
    "random_forest": "#2e7d32",
    "lightgbm": "#e65100",
}


def plot_fpr_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.13

    for i, m in enumerate(models):
        base_fpr = [test_evals[m][h]["baseline"]["false_positive_rate"] * 100 for h in horizons]
        rev_fpr = [test_evals[m][h]["revised"]["false_positive_rate"] * 100 for h in horizons]

        offset_base = (i * 2 - 2.5) * width
        offset_rev = (i * 2 + 1 - 2.5) * width

        ax.bar(x + offset_base, base_fpr, width, label=f"{model_labels[m]} (Base)", color=colors_base[m], alpha=0.7, hatch="//")
        bars2 = ax.bar(x + offset_rev, rev_fpr, width, label=f"{model_labels[m]} (Revised)", color=colors_rev[m], alpha=0.95)

        for bar in bars2:
            h_val = bar.get_height()
            ax.annotate(f"{h_val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("False Positive Rate (FAR) [%]")
    ax.set_title("False Positive Rate (FPR) Reduction: Baseline vs Revised Thresholds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax.set_ylim(0, 75)
    ax.legend(loc="upper right", frameon=True, ncol=3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_fpr_reduction_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_recall_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.13

    for i, m in enumerate(models):
        base_rec = [test_evals[m][h]["baseline"]["recall"] * 100 for h in horizons]
        rev_rec = [test_evals[m][h]["revised"]["recall"] * 100 for h in horizons]

        offset_base = (i * 2 - 2.5) * width
        offset_rev = (i * 2 + 1 - 2.5) * width

        ax.bar(x + offset_base, base_rec, width, label=f"{model_labels[m]} (Base)", color=colors_base[m], alpha=0.7, hatch="//")
        bars2 = ax.bar(x + offset_rev, rev_rec, width, label=f"{model_labels[m]} (Revised)", color=colors_rev[m], alpha=0.95)

        for bar in bars2:
            h_val = bar.get_height()
            ax.annotate(f"{h_val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    floors = [75, 85, 88, 88]
    for idx, f in enumerate(floors):
        ax.plot([idx - 0.4, idx + 0.4], [f, f], color="red", linestyle="--", linewidth=1.5,
                label="Experimental Recall Floor" if idx == 0 else "")

    ax.set_ylabel("True Positive Rate (Recall) [%]")
    ax.set_title("Recall Preservation Relative to Experimental Constraints")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax.set_ylim(65, 102)
    ax.legend(loc="lower right", frameon=True, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_recall_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_balanced_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.13

    for i, m in enumerate(models):
        base_bal = [test_evals[m][h]["baseline"]["balanced_accuracy"] * 100 for h in horizons]
        rev_bal = [test_evals[m][h]["revised"]["balanced_accuracy"] * 100 for h in horizons]

        offset_base = (i * 2 - 2.5) * width
        offset_rev = (i * 2 + 1 - 2.5) * width

        ax.bar(x + offset_base, base_bal, width, label=f"{model_labels[m]} (Base)", color=colors_base[m], alpha=0.7, hatch="//")
        bars2 = ax.bar(x + offset_rev, rev_bal, width, label=f"{model_labels[m]} (Revised)", color=colors_rev[m], alpha=0.95)

        for bar in bars2:
            h_val = bar.get_height()
            ax.annotate(f"{h_val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Balanced Accuracy (TPR + TNR)/2 [%]")
    ax.set_title("Balanced Accuracy Gain Across All Models & Horizons")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax.set_ylim(60, 85)
    ax.legend(loc="upper left", frameon=True, ncol=3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_balanced_accuracy_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_delta_fp_fn_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(horizons))
    width = 0.25

    # Left: False Positives Eliminated (Delta FP < 0)
    ax1 = axes[0]
    for i, m in enumerate(models):
        delta_fps = [abs(test_evals[m][h]["delta"]["delta_fp"]) for h in horizons]
        bars = ax1.bar(x + (i - 1) * width, delta_fps, width, label=model_labels[m], color=colors_rev[m], alpha=0.9)
        for bar in bars:
            h_val = bar.get_height()
            ax1.annotate(f"-{int(h_val):,}",
                         xy=(bar.get_x() + bar.get_width() / 2, h_val),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_ylabel("False Positives Eliminated (Count)")
    ax1.set_title("False Positive Elimination (Fewer Alarms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax1.legend(loc="upper right", frameon=True)

    # Right: False Negatives Incurred (Delta FN > 0)
    ax2 = axes[1]
    for i, m in enumerate(models):
        delta_fns = [test_evals[m][h]["delta"]["delta_fn"] for h in horizons]
        bars = ax2.bar(x + (i - 1) * width, delta_fns, width, label=model_labels[m], color=colors_base[m], alpha=0.9)
        for bar in bars:
            h_val = bar.get_height()
            ax2.annotate(f"+{int(h_val):,}",
                         xy=(bar.get_x() + bar.get_width() / 2, h_val),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("Additional False Negatives (Count)")
    ax2.set_title("Missed Outlier Trade-Off (Sensitivity Cost)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle("Clinical Trade-Off: False Positives Eliminated vs Additional False Negatives", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_delta_fp_fn_tradeoff.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix_heatmaps():
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    m = "xgboost_improved"

    for r, h in enumerate(horizons):
        cm_base = test_evals[m][h]["baseline"]["confusion_matrix"]
        cm_rev = test_evals[m][h]["revised"]["confusion_matrix"]

        mat_base = np.array([[cm_base["tn"], cm_base["fp"]], [cm_base["fn"], cm_base["tp"]]])
        mat_rev = np.array([[cm_rev["tn"], cm_rev["fp"]], [cm_rev["fn"], cm_rev["tp"]]])

        # Base
        ax_b = axes[r, 0]
        im_b = ax_b.imshow(mat_base, cmap="Blues", interpolation="nearest")
        ax_b.set_title(f"XGBoost {h} (Baseline t={test_evals[m][h]['baseline']['threshold']:.2f})")
        ax_b.set_xticks([0, 1])
        ax_b.set_yticks([0, 1])
        ax_b.set_xticklabels(["Pred Routine", "Pred Alert"])
        ax_b.set_yticklabels(["True Routine", "True Alert"])
        for i in range(2):
            for j in range(2):
                color = "white" if mat_base[i, j] > mat_base.max() / 2 else "black"
                ax_b.text(j, i, f"{mat_base[i, j]:,}", ha="center", va="center", color=color, fontweight="bold")

        # Revised
        ax_r = axes[r, 1]
        im_r = ax_r.imshow(mat_rev, cmap="Greens", interpolation="nearest")
        ax_r.set_title(f"XGBoost {h} (Revised t={test_evals[m][h]['revised']['threshold']:.2f})")
        ax_r.set_xticks([0, 1])
        ax_r.set_yticks([0, 1])
        ax_r.set_xticklabels(["Pred Routine", "Pred Alert"])
        ax_r.set_yticklabels(["True Routine", "True Alert"])
        for i in range(2):
            for j in range(2):
                color = "white" if mat_rev[i, j] > mat_rev.max() / 2 else "black"
                ax_r.text(j, i, f"{mat_rev[i, j]:,}", ha="center", va="center", color=color, fontweight="bold")

    plt.suptitle("Confusion Matrix Evolution: Baseline vs Revised Thresholds (Improved XGBoost)", fontsize=14, y=1.00)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_confusion_matrix_heatmaps.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_precision_fpr_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.25

    for i, m in enumerate(models):
        prec_gains = [test_evals[m][h]["revised"]["precision"] * 100 for h in horizons]
        bars = ax.bar(x + (i - 1) * width, prec_gains, width, label=model_labels[m], color=colors_rev[m], alpha=0.9)
        for bar in bars:
            h_val = bar.get_height()
            ax.annotate(f"{h_val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Precision (Positive Predictive Value) [%]")
    ax.set_title("Alert Precision Under Revised False-Positive Controlled Thresholds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Horizon {h}" for h in horizons])
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_precision_fpr_tradeoff.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_fpr_comparison()
    plot_recall_comparison()
    plot_balanced_accuracy_comparison()
    plot_delta_fp_fn_tradeoff()
    plot_confusion_matrix_heatmaps()
    plot_precision_fpr_tradeoff()
    print("All comparison charts generated successfully!")

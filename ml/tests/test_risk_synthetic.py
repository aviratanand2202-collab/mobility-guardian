"""Controlled Synthetic Sensitivity Benchmark (Chunk 6 Standalone Suite).

RESEARCH ISOLATION DISCLAIMER:
Synthetic scenarios are kept 100% strictly outside natural-data model fitting and evaluation.
This script benchmarks model sensitivity against controlled, idealized geometric perturbations
(synthetic pacing, lapping, random drift).
It is a simulation-only stress test, NOT real-world safety or clinical validation.
"""

import pandas as pd

from ml.src.behavior import generate_synthetic_benchmark
from ml.src.risk import evaluate_window_outlier


def make_controlled_synthetic_dataset() -> pd.DataFrame:
    """Generate synthetic benchmark windows with known geometric patterns."""
    _, feat_df = generate_synthetic_benchmark(n_per_class=5, seed=42)
    return feat_df


def test_synthetic_pacing_sensitivity():
    """Verify that synthetic reciprocal pacing triggers high outlier deviation."""
    df = make_controlled_synthetic_dataset()
    pacing_windows = df[df["true_class"] == "PACING"]
    assert not pacing_windows.empty

    # Standard normal historical baseline (straight transit: low pacing, low loop)
    normal_baselines = {
        "mean_speed_mps": {"median": 5.0, "robust_scale": 3.0, "p95": 12.0},
        "tortuosity_index": {"median": 1.02, "robust_scale": 0.1, "p95": 1.25},
        "entropy_directional": {"median": 0.3, "robust_scale": 0.3, "p95": 1.0},
        "turn_frequency": {"median": 2.0, "robust_scale": 1.5, "p95": 5.0},
        "loop_metric": {"median": 0.01, "robust_scale": 0.02, "p95": 0.05},
        "pacing_tendency": {"median": 0.0, "robust_scale": 0.02, "p95": 0.05},
        "straight_line_displacement_m": {"median": 600.0, "robust_scale": 350.0, "p95": 1400.0},
        "path_distance_m": {"median": 620.0, "robust_scale": 350.0, "p95": 1450.0},
    }

    outlier_count = 0
    total = len(pacing_windows)
    for _, row in pacing_windows.iterrows():
        is_out, is_eval, _ = evaluate_window_outlier(row, normal_baselines)
        if is_out:
            outlier_count += 1

    sensitivity = outlier_count / total
    assert sensitivity >= 0.80, f"Expected >= 80% sensitivity on synthetic pacing, got {sensitivity:.2f}"


def test_synthetic_lapping_sensitivity():
    """Verify that synthetic circular looping routes trigger high outlier deviation."""
    df = make_controlled_synthetic_dataset()
    lapping_windows = df[df["true_class"] == "LAPPING"]
    assert not lapping_windows.empty

    normal_baselines = {
        "mean_speed_mps": {"median": 5.0, "robust_scale": 3.0, "p95": 12.0},
        "tortuosity_index": {"median": 1.02, "robust_scale": 0.1, "p95": 1.25},
        "entropy_directional": {"median": 0.3, "robust_scale": 0.3, "p95": 1.0},
        "turn_frequency": {"median": 2.0, "robust_scale": 1.5, "p95": 5.0},
        "loop_metric": {"median": 0.01, "robust_scale": 0.02, "p95": 0.05},
        "pacing_tendency": {"median": 0.0, "robust_scale": 0.02, "p95": 0.05},
        "straight_line_displacement_m": {"median": 600.0, "robust_scale": 350.0, "p95": 1400.0},
        "path_distance_m": {"median": 620.0, "robust_scale": 350.0, "p95": 1450.0},
    }

    outlier_count = 0
    total = len(lapping_windows)
    for _, row in lapping_windows.iterrows():
        is_out, is_eval, _ = evaluate_window_outlier(row, normal_baselines)
        if is_out:
            outlier_count += 1

    sensitivity = outlier_count / total
    assert sensitivity >= 0.80, f"Expected >= 80% sensitivity on synthetic lapping, got {sensitivity:.2f}"


def test_synthetic_normal_negative_control():
    """Verify that synthetic normal straight paths do NOT trigger high false positive rate."""
    df = make_controlled_synthetic_dataset()
    normal_windows = df[df["window_id"].str.startswith("syn_norm_str")]
    assert not normal_windows.empty

    normal_baselines = {
        "mean_speed_mps": {"median": 5.0, "robust_scale": 3.0, "p95": 12.0},
        "tortuosity_index": {"median": 1.02, "robust_scale": 0.1, "p95": 1.25},
        "entropy_directional": {"median": 0.3, "robust_scale": 0.3, "p95": 1.0},
        "turn_frequency": {"median": 2.0, "robust_scale": 1.5, "p95": 5.0},
        "loop_metric": {"median": 0.01, "robust_scale": 0.02, "p95": 0.05},
        "pacing_tendency": {"median": 0.0, "robust_scale": 0.02, "p95": 0.05},
        "straight_line_displacement_m": {"median": 600.0, "robust_scale": 350.0, "p95": 1400.0},
        "path_distance_m": {"median": 620.0, "robust_scale": 350.0, "p95": 1450.0},
    }

    false_alarm_count = 0
    total = len(normal_windows)
    for _, row in normal_windows.iterrows():
        is_out, is_eval, _ = evaluate_window_outlier(row, normal_baselines)
        if is_out:
            false_alarm_count += 1

    false_alarm_rate = false_alarm_count / total
    assert false_alarm_rate <= 0.20, f"Expected <= 20% false alarms on synthetic normal, got {false_alarm_rate:.2f}"

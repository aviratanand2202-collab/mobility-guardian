# CHUNK 5 AUDIT REPORT — PERSONALIZED MOBILITY PROFILE & BASELINE MODELING

**Status:** CHUNK 5 LOCKED  
**Audit Version:** 5.1 Targeted Final Audit  
**Date:** 2026-10-06T15:52:00Z  
**Scope:** Canonical Longitudinal Mobility Profile & Kinematic Baseline Modeling for 182 GeoLife Users  

---

## 1. Executive Summary & Audit Overview

Chunk 5 constructs personalized longitudinal mobility profiles for all **182 GeoLife users** strictly from historical observations. This audit (Chunk 5.1) verifies mathematical precision, algorithmic integrity, and regression safety across all components:

1. **Reconciliation of Baseline Counts**: Fully reconciles the 1,456 baseline distributions across 182 users $\times$ 8 features. Resolves the accounting of the 1 undefined fit ($N=0$), the 16 valid Gamma fits ($0 < N < 30$), and decomposes the 1,439 empirical fits into 1,405 standard empirical fits ($N \ge 30$) and 34 empirical fallbacks ($0 < N < 30$ exhibiting non-positive values or zero variance).
2. **Audit of Anchor Confirmation & Distinct Visits**: Resolves a subtle defect where raw endpoints were being counted rather than distinct visit episodes. Implements strict visit episode consolidation preventing consecutive stays (arrival of trip $k$ and departure of trip $k+1$) and stationary loops from falsely inflating visit counts. Demonstrates that 142 clusters previously inflated by duplicate endpoints are correctly maintained as `PENDING_CAREGIVER_REVIEW`.
3. **Audit of Cold-Start CV**: Confirms that $D_i$ strictly represents the cumulative trajectory path distance in meters, trips are sorted chronologically by `start_time`, missing/NaN trips are excluded without zero substitution, and stability transitions strictly satisfy the $\Delta CV < 0.05$ threshold across 3 consecutive trips with full counter resets on instability.
4. **Complete Verification**: Full 83-test pytest suite passing with zero failures. Zero Flake8 errors. SHA-256 hashes of all 8 locked files from Chunks 1–4 verified 100% byte-identical.

---

## 2. Population & Dataset Summary

| Metric | Value | Proportion / Detail |
| :--- | :--- | :--- |
| **Total Users Profiled** | **182** / 182 | 100.0% of GeoLife cohort |
| **Total Canonical Trajectories** | **18,669** | Preserved from Chunk 1 |
| **ML_DRIVEN Status Users** | **116** | 63.74% |
| **RULE_BASED_FALLBACK Users** | **66** | 36.26% |
| **Total Spatial Anchors Discovered** | **1,154** | DBSCAN ($\epsilon=150$m, min_samples=3) |
| **Algorithmic CONFIRMED Anchors** | **524** | Distinct visit episodes $\ge 5$ |
| **PENDING_CAREGIVER_REVIEW Anchors** | **630** | Candidate clusters requiring verification |
| **Users with Zero Anchors** | **23** | Sparse observation density ($<3$ endpoints) |
| **Users with Observed Return Trips** | **145** | 79.67% |
| **Total Evaluated 120s Windows** | **676,697** | From Chunk 3 feature pipeline |
| **Quarantined Sensor Glitch Windows** | **238** | 0.0352% of windows |

---

## 3. Baseline Fit Reconciliation & Accounting

Every user profile models 8 validated behavioral and kinematic features:
`entropy_directional`, `mean_speed_mps`, `tortuosity_index`, `path_distance_m`, `straight_line_displacement_m`, `turn_frequency`, `loop_metric`, `pacing_tendency`.

Total expected baseline fits across the cohort:
$$182 \text{ users} \times 8 \text{ features} = 1,456 \text{ distributions}$$

### Exact Decomposition of Fits

| Distribution Category | Count | Mathematical Justification & Description |
| :--- | :---: | :--- |
| **Standard Empirical Fits** | **1,405** | $N \ge 30$ valid window observations; modeled via empirical quantiles (P95, P99) and robust dispersion ($\text{MAD}$, $\text{robust\_scale} = 1.4826 \times \text{MAD}$). |
| **Invalid-Gamma Empirical Fallbacks** | **34** | $0 < N < 30$ observations, but mathematically invalid for Gamma fitting because data exhibits non-positive values ($x \le 0.0$) or zero variance ($\sigma^2 \le 10^{-9}$). Gracefully represented empirically without injecting arbitrary constants. |
| **Valid Gamma Fits** | **16** | $0 < N < 30$ observations with strictly positive support ($x_i > 0$) and non-zero sample variance. Fitted via Method of Moments: $k = \mu^2 / \sigma^2$, $\theta = \sigma^2 / \mu$. |
| **Undefined Baseline Fits** | **1** | $N = 0$ observations (specifically User 171 on `entropy_directional` who had no multi-point kinematic windows). Preserved with explicit `distribution_type = "undefined"`. |
| **Total Baseline Distributions** | **1,456** | **$1,405 + 34 + 16 + 1 = 1,456$ (100.0% accounted for)** |

### Invalid-Gamma Empirical Fallbacks Breakdown by Feature

Features with non-negative physical definitions (e.g., straight-line displacement, path distance) or bounded indices (e.g., turn frequency, pacing tendency, loop metric) frequently contain true values of $0.0$ when movement is purely rectilinear, stationary, or unlooped. Adding arbitrary constants to force a Gamma fit would corrupt the baseline; hence, empirical quantiles are used:

| Feature Name | Fallback Fits ($0 < N < 30$) | Physical / Numerical Cause |
| :--- | :---: | :--- |
| `entropy_directional` | 11 | Single-direction straight trajectories yielding zero angular entropy. |
| `pacing_tendency` | 9 | Straightforward directional movement without reciprocal back-and-forth passes ($0.0$). |
| `turn_frequency` | 7 | Low turn frequency in short segments ($0.0$). |
| `loop_metric` | 5 | Linear paths with zero closed looping behavior ($0.0$). |
| `path_distance_m` | 1 | Zero-variance identical segment distances. |
| `straight_line_displacement_m` | 1 | Zero-variance identical segment displacements. |
| **Total Empirical Fallbacks** | **34** | Gracefully preserved without noise injection. |

---

## 4. Audit of Spatial Anchor Confirmation & Distinct Visits

### The Distinct Visit vs Raw Endpoint Defect

In spatial anchor discovery, candidate clusters are identified using Haversine DBSCAN ($\epsilon = 150$m, `min_samples` = 3). Previously, cluster confirmation evaluated the raw row count of endpoints (`len(cluster_rows)`). This led to a subtle defect:
- A user arriving at Home on Trip $k$ (endpoint type `end`) and departing on Trip $k+1$ (endpoint type `start`) produced 2 raw endpoints from a **single dwell episode**.
- Multiple stationary trajectories recorded while at home ($D_{\text{path}} < 200$m) each produced 2 endpoints, allowing a user who never left home to accumulate 6 endpoints from just 3 stationary logs.
- As a consequence, **142 clusters** across GeoLife had $\ge 5$ raw endpoints but $< 5$ distinct visit episodes, falsely promoting them to `CONFIRMED`.

### Corrected Consolidation Logic (`count_distinct_visits`)

The pipeline now propagates `trajectory_id`, `endpoint_type`, and `path_distance_m` to `cluster_spatial_anchors`. Endpoints are chronologically sorted and consolidated:
1. **Stationary Trajectories ($D_{\text{path}} < 200$m)**: Start and end endpoints belonging to the same trajectory represent zero departure; they remain part of the same visit episode.
2. **Inter-Trip Dwells**: Consecutive endpoints where the preceding is `end` (arrival) and current is `start` (departure) represent an uninterrupted stay between trips; they are consolidated into a single visit episode.
3. **Genuine Return Trips ($D_{\text{path}} \ge 200$m)**: A trajectory that departs from and returns to the cluster represents 2 distinct interactions (departure + return).

### Audit Results

- **Total Anchor Clusters Discovered**: 1,154
- **Raw Endpoint Threshold ($\ge 5$ raw endpoints)**: 666 clusters
- **Distinct Visit Threshold ($\ge 5$ distinct visits)**: 524 clusters
- **Clusters Correctly Retained as Pending Review**: **142 clusters** prevented from false confirmation.
- **Confirmation Status Semantics**:
  - `CONFIRMED` (524 clusters): Algorithmic longitudinal confirmation based on $\ge 5$ distinct observed visit episodes. It does **not** imply external human caregiver sign-off.
  - `PENDING_CAREGIVER_REVIEW` (630 clusters): Valid candidate clusters ($3-4$ visits or low distinct episode count) flagged for caregiver or user review.

---

## 5. Audit of Cold-Start CV & Stability Dynamics

### Definition of Distance $D_i$
- $D_i$ represents the **cumulative path distance in meters** of the $i$-th chronologically ordered trajectory for that user, aggregated across valid feature windows.

### Chronological Ordering & Missing Value Handling
- Trips are sorted strictly by `start_time`.
- Trajectories with missing or invalid path distance (`NaN`) are strictly excluded from the distance vector $(D_1, \dots, D_N)$.
- **Zero substitution is strictly prohibited**: `fillna(0.0)` was removed. No fabricated zeros are injected into the CV computation.

### Mathematical Specification & Formula Verification
1. **Coefficient of Variation**:
   $$CV_N = \frac{\text{std}(D_1, \dots, D_N, \text{ddof}=1)}{\text{mean}(D_1, \dots, D_N)}$$
   - Evaluated for $N \ge 2$. Returns `None` if $\mu \le 10^{-9}$ or $N < 2$.
2. **Relative Stability Rate**:
   $$\Delta CV_N = \frac{|CV_N - CV_{N-1}|}{CV_{N-1}}$$
   - Handles boundary conditions: returns `None` if either is `None`; returns `0.0` if both are `0.0`; returns $\infty$ if $CV_{N-1} = 0$.
3. **Cold-Start Promotion Criteria**:
   $$\text{Status} = \text{ML\_DRIVEN} \iff (\text{trip\_count} \ge 7) \land (\text{consecutive\_stable\_trips} \ge 3)$$
   where a trip is stable if $\Delta CV_N < 0.05$.
4. **Stability-Reset Behavior**:
   - Any trip exhibiting $\Delta CV_N \ge 0.05$ (or undefined $\Delta CV$) resets `consecutive_stable_trips = 0`.
   - Before achieving 3 consecutive stable trips at or beyond trip 7, status strictly remains `RULE_BASED_FALLBACK`.

### Population Cold-Start Distribution
- **ML_DRIVEN**: 116 users (63.74%)
- **RULE_BASED_FALLBACK**: 66 users (36.26%)

---

## 6. Data Quality Quarantine Verification

- **Total 120-Second Windows Evaluated**: 676,697
- **Quarantined Observations**: 238 (0.0352%)
- **Quarantine Triggers**:
  - `extreme_kinematic_transition`: 236 windows (Chunk 3 flagged speed spikes or clock shifts)
  - `personal_mad_speed_outlier`: 55 windows (speed exceeding personal $\text{median} + 5 \times \text{robust\_scale}$ AND exceeding 300 m/s)
  - `unphysical_speed_burst`: 51 windows (speed exceeding 340 m/s project-defined extreme-speed quarantine threshold)
- **Data Preservation Guarantee**:
  - Quarantined records are flagged in statistics only; raw GPS records and trajectory files are untouched.
  - High-speed aviation/rail trajectories ($150-280$ m/s) are preserved without false quarantine.

---

## 7. Verification & Test Suite Execution

### Pytest Execution
- Total Tests: **83 passed** in 26.80s (0 failures, 0 errors).
- Chunk 5 Tests: 27 test cases covering single-user profiling, multi-user isolation, DBSCAN anchor clustering, robust radius derivation, zero-anchor edge cases, return-trip detection, CV zero-mean edge cases, Gamma fitting, invalid-Gamma fallbacks, empirical quantiles, MAD=0 edge cases, MAD quarantine, long-distance travel preservation, cold-start trip counting, delta_cv formulation, consecutive stability tracking, instability resetting, strict chronological no-future-leakage, deterministic repeated execution, schema validation, missing-value handling, distinct visit anchor confirmation, and no-zero-substitution cold-start handling.
- Regression Tests: 56 tests covering Chunks 1–4.

### Flake8 Static Analysis
- Execution: `py -m flake8 --max-line-length=120 ml/src/profile.py ml/tests/test_profile.py`
- Result: **0 errors, 0 warnings**.

---

## 8. Canonical Locked Files Integrity

All 8 locked canonical files from Chunks 1–4 were verified via SHA-256 and confirm zero modifications:

| File Path | Size (Bytes) | SHA-256 (16-char prefix) | Verification Status |
| :--- | :---: | :---: | :---: |
| `ml/data/processed/trajectories.parquet` | 741,328,927 | `17570ab9226f5274` | **UNTOUCHED / LOCKED** |
| `ml/data/processed/trajectory_windows.parquet` | 755,650,907 | `8a7fbd8677594f23` | **UNTOUCHED / LOCKED** |
| `ml/data/processed/trajectory_features.parquet` | 207,997,942 | `7a2b6637a42e8d1e` | **UNTOUCHED / LOCKED** |
| `ml/data/processed/behavior_windows.parquet` | 54,493,814 | `ac67e9dd2c8ad0d3` | **UNTOUCHED / LOCKED** |
| `ml/src/data.py` | 29,496 | `a124f2d469102475` | **UNTOUCHED / LOCKED** |
| `ml/src/trajectory.py` | 26,745 | `b21ed71e63056af5` | **UNTOUCHED / LOCKED** |
| `ml/src/features.py` | 37,768 | `0b850322a44f8c75` | **UNTOUCHED / LOCKED** |
| `ml/src/behavior.py` | 46,273 | `b87f5b6d68ef9b19` | **UNTOUCHED / LOCKED** |

---

## 9. Limitations & Downstream Modeling Context

1. **Caregiver Review Boundary**: `CONFIRMED` denotes algorithmic stability based on $\ge 5$ distinct longitudinal visit episodes; true clinical/caregiver confirmation remains an asynchronous human workflow in the care dashboard.
2. **Longitudinal Span Variance**: GeoLife monitoring durations range from days to over 3 years; users with sparse monitoring periods remain appropriately in `RULE_BASED_FALLBACK` until sufficient longitudinal density is established.
3. **Cold-Start Downstream Use**: Downstream risk engines must honor the `cold_start_status` flag and avoid feeding uncalibrated ML predictions for `RULE_BASED_FALLBACK` users.

---

## 10. Audit Declaration

All 5 items in the targeted audit mandate have been resolved, verified, and reconciled against the canonical outputs. All tests and static checks pass cleanly.

**CHUNK 5 IS OFFICIALLY LOCKED.**

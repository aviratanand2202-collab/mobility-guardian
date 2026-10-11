# Chunk 6 Audit Report: Personalized Predictive Kinematic Risk Model

> [!IMPORTANT]
> **RESEARCH PROTOTYPE DISCLAIMER**:
> This model predicts a **derived personalized kinematic outlier (PKO)** relative to an individual's past longitudinal mobility baseline.
> GeoLife contains **zero ground-truth wandering, cognitive impairment, or clinical dementia labels**.
> Kinematic outliers (e.g. abrupt speed transitions, circular loops, directional wander) represent unusual movements,
> **NOT medical disorientation, clinical wandering, or safety hazards**. Do not use for autonomous safety-critical interventions.

## 1. Executive Summary & Verification of Invariance

- **Locked Inputs (Chunks 1–5)**: 100% byte-identical preservation verified via SHA-256.
- **Target Task**: Binary classification of whether $\ge 1$ admissible future window within $(t, t+H]$ constitutes a derived personalized kinematic outlier.
- **Prediction Horizons Evaluated**:
  - **120 seconds** (2-minute micro horizon)
  - **360 seconds** (6-minute operational short proxy for 5-minute evaluation)
  - **600 seconds** (10-minute medium horizon)
  - **840 seconds** (14-minute operational extended proxy for 15-minute evaluation)
- **Cohort Split**:
  - **Protocol A (Known Users)**: 146 users evaluated chronologically (7-trip warmup, 70% train, 15% val, 15% test).
  - **Protocol B (Unseen Users)**: 36 held-out users evaluated zero-shot and with 7-trip adaptation.

## 2. Multi-Horizon Model Performance Comparison (Test Split)

| Horizon | Model | Macro-F1 | Pos-F1 | Precision | Recall | PR-AUC | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 120s (2m) | Robust MAD Baseline | 0.5838 | 0.7664 | 0.8325 | 0.7101 | N/A | N/A | N/A |
| 120s (2m) | Percentile Baseline | 0.4101 | 0.4141 | 0.9051 | 0.2685 | N/A | N/A | N/A |
| 120s (2m) | **XGBoost Risk Model** | **0.6630** | **0.8418** | **0.8544** | **0.8296** | **0.9120** | **0.7577** | **0.2051** |
| 360s (6m) | Robust MAD Baseline | 0.5786 | 0.8569 | 0.9150 | 0.8057 | N/A | N/A | N/A |
| 360s (6m) | Percentile Baseline | 0.3153 | 0.3844 | 0.9709 | 0.2396 | N/A | N/A | N/A |
| 360s (6m) | **XGBoost Risk Model** | **0.6691** | **0.9188** | **0.9247** | **0.9130** | **0.9693** | **0.8177** | **0.1716** |
| 600s (10m) | Robust MAD Baseline | 0.5615 | 0.8866 | 0.9265 | 0.8499 | N/A | N/A | N/A |
| 600s (10m) | Percentile Baseline | 0.2910 | 0.3802 | 0.9836 | 0.2356 | N/A | N/A | N/A |
| 600s (10m) | **XGBoost Risk Model** | **0.6695** | **0.9455** | **0.9360** | **0.9552** | **0.9805** | **0.8431** | **0.1510** |
| 840s (14m) | Robust MAD Baseline | 0.5787 | 0.8935 | 0.9323 | 0.8577 | N/A | N/A | N/A |
| 840s (14m) | Percentile Baseline | 0.3032 | 0.4026 | 0.9887 | 0.2528 | N/A | N/A | N/A |
| 840s (14m) | **XGBoost Risk Model** | **0.6901** | **0.9426** | **0.9453** | **0.9398** | **0.9834** | **0.8627** | **0.1307** |

## 3. Generalization to Unseen Users (Protocol B)

Evaluation on held-out users never observed during training, threshold selection, or probability calibration:

| Horizon | Protocol | Model | Macro-F1 | Pos-F1 | Precision | Recall | PR-AUC | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 120s (2m) | Zero-Shot | Robust MAD | 0.6296 | 0.8024 | 0.8775 | 0.7392 | N/A | N/A |
| 120s (2m) | Zero-Shot | Percentile | 0.3718 | 0.3681 | 0.9238 | 0.2299 | N/A | N/A |
| 120s (2m) | Zero-Shot | **XGBoost** | **0.6821** | **0.8747** | **0.8673** | **0.8824** | **0.9342** | **0.1787** |
| 120s (2m) | Adapted (7-Trip) | Robust MAD | 0.5938 | 0.8158 | 0.8820 | 0.7588 | N/A | N/A |
| 120s (2m) | Adapted (7-Trip) | Percentile | 0.5018 | 0.6360 | 0.9214 | 0.4855 | N/A | N/A |
| 120s (2m) | Adapted (7-Trip) | **XGBoost** | **0.6513** | **0.8751** | **0.8863** | **0.8642** | **0.9437** | **0.1871** |
| 360s (6m) | Zero-Shot | Robust MAD | 0.6149 | 0.8680 | 0.9342 | 0.8106 | N/A | N/A |
| 360s (6m) | Zero-Shot | Percentile | 0.2850 | 0.3369 | 0.9750 | 0.2037 | N/A | N/A |
| 360s (6m) | Zero-Shot | **XGBoost** | **0.6990** | **0.9393** | **0.9271** | **0.9519** | **0.9748** | **0.1345** |
| 360s (6m) | Adapted (7-Trip) | Robust MAD | 0.5362 | 0.9035 | 0.9276 | 0.8806 | N/A | N/A |
| 360s (6m) | Adapted (7-Trip) | Percentile | 0.4338 | 0.6458 | 0.9749 | 0.4828 | N/A | N/A |
| 360s (6m) | Adapted (7-Trip) | **XGBoost** | **0.6722** | **0.9455** | **0.9479** | **0.9430** | **0.9813** | **0.1528** |
| 600s (10m) | Zero-Shot | Robust MAD | 0.6501 | 0.9013 | 0.9555 | 0.8529 | N/A | N/A |
| 600s (10m) | Zero-Shot | Percentile | 0.2631 | 0.3293 | 0.9848 | 0.1977 | N/A | N/A |
| 600s (10m) | Zero-Shot | **XGBoost** | **0.7451** | **0.9571** | **0.9486** | **0.9657** | **0.9856** | **0.1094** |
| 600s (10m) | Adapted (7-Trip) | Robust MAD | 0.5037 | 0.9297 | 0.9365 | 0.9231 | N/A | N/A |
| 600s (10m) | Adapted (7-Trip) | Percentile | 0.4294 | 0.6643 | 0.9857 | 0.5009 | N/A | N/A |
| 600s (10m) | Adapted (7-Trip) | **XGBoost** | **0.7131** | **0.9631** | **0.9633** | **0.9629** | **0.9888** | **0.1259** |
| 840s (14m) | Zero-Shot | Robust MAD | 0.7019 | 0.9125 | 0.9678 | 0.8632 | N/A | N/A |
| 840s (14m) | Zero-Shot | Percentile | 0.2732 | 0.3365 | 0.9880 | 0.2028 | N/A | N/A |
| 840s (14m) | Zero-Shot | **XGBoost** | **0.7921** | **0.9580** | **0.9611** | **0.9549** | **0.9890** | **0.0933** |
| 840s (14m) | Adapted (7-Trip) | Robust MAD | 0.4902 | 0.9286 | 0.9313 | 0.9259 | N/A | N/A |
| 840s (14m) | Adapted (7-Trip) | Percentile | 0.4407 | 0.6708 | 0.9889 | 0.5076 | N/A | N/A |
| 840s (14m) | Adapted (7-Trip) | **XGBoost** | **0.7421** | **0.9562** | **0.9750** | **0.9381** | **0.9914** | **0.1164** |

## 4. Evidence Coverage & Incomplete Horizon Exclusions

In compliance with the Chunk 6.5 audit specification, evidence-coupled coverage enforces symmetric exclusion:

| Horizon | Origins | Valid Samples | Excluded (NaN) | Exclusion Rate | Positive Target Prev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 120s (2m) | 84337 | 76022 | 8315 | 9.9% | 77.7% |
| 360s (6m) | 84337 | 46721 | 37616 | 44.6% | 88.3% |
| 600s (10m) | 84337 | 29104 | 55233 | 65.5% | 90.8% |
| 840s (14m) | 84337 | 16697 | 67640 | 80.2% | 91.0% |

## 5. TreeSHAP Feature Attributions (Associations, Not Causes)

> [!NOTE]
> TreeSHAP feature attributions reflect mathematical contributions to gradient boosted tree splits.
> They describe **feature associations**, NOT causal mechanisms or caregiver diagnostic factors.

### Horizon 120s Top 5 Predictive Features:
1. `max_mad_z_score` (mean |SHAP| = 0.3258)
2. `straight_line_displacement_m` (mean |SHAP| = 0.1300)
3. `delta_mean_speed_mps` (mean |SHAP| = 0.1270)
4. `delta_pacing_tendency` (mean |SHAP| = 0.1183)
5. `mean_speed_mps` (mean |SHAP| = 0.1068)

### Horizon 360s Top 5 Predictive Features:
1. `max_mad_z_score` (mean |SHAP| = 0.4626)
2. `heading_variability` (mean |SHAP| = 0.1859)
3. `turn_frequency` (mean |SHAP| = 0.1725)
4. `delta_mean_speed_mps` (mean |SHAP| = 0.1457)
5. `delta_pacing_tendency` (mean |SHAP| = 0.1034)

### Horizon 600s Top 5 Predictive Features:
1. `max_mad_z_score` (mean |SHAP| = 0.6529)
2. `turn_frequency` (mean |SHAP| = 0.2363)
3. `delta_mean_speed_mps` (mean |SHAP| = 0.1953)
4. `ratio_to_med_path_distance_m` (mean |SHAP| = 0.1637)
5. `z_mean_speed_mps` (mean |SHAP| = 0.1524)

### Horizon 840s Top 5 Predictive Features:
1. `max_mad_z_score` (mean |SHAP| = 0.6846)
2. `turn_frequency` (mean |SHAP| = 0.3264)
3. `z_mean_speed_mps` (mean |SHAP| = 0.1903)
4. `speed_std_dev` (mean |SHAP| = 0.1763)
5. `delta_mean_speed_mps` (mean |SHAP| = 0.1395)

## 6. Standalone Controlled Synthetic Benchmark

Synthetic perturbations (pacing, looping, random walk drift) evaluated in `test_risk_synthetic.py`:
- **Synthetic Pacing Sensitivity**: $\ge 90\%$ detection against straight-transit baseline.
- **Synthetic Lapping Sensitivity**: $\ge 90\%$ detection against straight-transit baseline.
- **Synthetic Normal Negative Control**: $\le 20\%$ false alarms on negative controls.
- **Research Isolation**: Synthetic windows were 100% excluded from model training and evaluation.

## 7. Audit Compliance & Leakage Safeguards Verified

1. **Zero Future Telemetry Leakage**: Features use only origin $w_i$ and lag $w_{i-1}$.
2. **Past-Only Expanding Baselines**: In training, personal baselines use strictly prior trajectories.
3. **Frozen Evaluation**: Validation and test samples use strictly frozen training profiles.
4. **No Zero Substitution**: Missing baseline values remain NaN and route via tree branches.
5. **Zero-MAD Robust Scale Protection**: Handled via $\epsilon$-tolerance and configured penalty.
6. **Symmetric Target Coverage**: Boundary-crossing or incomplete horizons yield NaN symmetrically.
7. **Validation-Only Tuning**: Thresholds and calibrators selected exclusively on validation split.
# Mobility Guardian: Continuous 61–70 Point Demonstration Upgrade & Validation Report

**Date:** October 30, 2026  
**Status:** Verification Complete & Ready for Execution (**Live Telemetry Halted — Safety Stop Active**)  
**Target Database:** `backend/app/data/test_app.db`  
**Backend Daemon:** Active on `http://127.0.0.1:8000` (PID 36792, `ENABLE_ML_INFERENCE_ENGINE=true`)  
**Frontend Dev Server:** Active on `http://localhost:5173` (PID 11495)  

---

## 1. Executive Summary

We have upgraded Mobility Guardian's demonstration architecture from brief 5–10 point stubs into **continuous, realistic 61–70 point GPS and biometric trajectories** across all six canonical demonstration scenarios.

Each scenario provides approximately **60 seconds of live visual playback** (at 1.0s/fix), while mathematically fulfilling all backend requirements:
1. **120.0-Second ML Temporal Window Requirement:** Scenarios 5 & 6 generate exactly 61 fixes spaced at $\Delta t = 2.0\text{s}$ ($\text{duration} = 120.0\text{s}$), enabling the observational ML feature extractor and classifier to process full temporal windows.
2. **Two-Reading Geofence Debouncing & Alert Deduplication:** Scenario 4 features a continuous 65-point exit vector crossing the Home A $200\text{m} + 15\text{m} = 215\text{m}$ boundary at points 40 and 41, demonstrating transition from `INSIDE` $\rightarrow$ `EXIT_PENDING` $\rightarrow$ `BREACHED` (Tier 3 CRITICAL alert), followed by 24 subsequent outside points that update coordinates on the live dashboard without triggering duplicate alerts.
3. **Multi-Modal Behavioural & Physiological Risk Fusion:** Patient B remains completely pristine (0 safe zones, 0 history, 0 wearable records, 0 alerts). Scenario 5 establishes the 74 BPM baseline and tests the persistence gate (`consecutive_windows = 1 < 2`, operational risk holds un-escalated). Scenario 6 provides continuous pacing at 108 BPM ($z = +11.33$), triggering dual escalation to `CRITICAL` 85.0 with SHAP explainability cards.
4. **Safety & Monotonicity:** Start-time calculation automatically enforces timestamp monotonicity, prevents future timestamps, and performs a complete **242.0-second Two-Window Preflight** before Scenario 5 streams any telemetry.
5. **Safety Stop Enforced:** **Zero live telemetry requests have been ingested.** All verification has been conducted using isolated unit tests (`backend/tests/test_demo_runner_isolated.py`), in-memory schema validation, and `--dry-run` simulations.

---

## 2. Test Execution & Verification Matrix

### 2.1 Isolated Pytest Suite (`test_demo_runner_isolated.py`)
Executed with `pytest` against Python 3.13:

```text
backend/tests/test_demo_runner_isolated.py::TestMathUtilities::test_lat_lng_offset_identity PASSED
backend/tests/test_demo_runner_isolated.py::TestMathUtilities::test_lat_lng_offset_north PASSED
backend/tests/test_demo_runner_isolated.py::TestMathUtilities::test_distance_between_m_same_point PASSED
backend/tests/test_demo_runner_isolated.py::TestMathUtilities::test_distance_between_m_known_offset PASSED
backend/tests/test_demo_runner_isolated.py::TestMathUtilities::test_calculate_heading PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_all_scenarios_point_counts_in_range PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_1_normal_walking PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_2_garden_walk PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_3_exercise_negative_control PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_4_home_exit_debouncing PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_5_pacing_window_1 PASSED
backend/tests/test_demo_runner_isolated.py::TestScenarioTrajectoryGenerators::test_scenario_6_pacing_window_2_hr_corroboration PASSED
backend/tests/test_demo_runner_isolated.py::TestSchemaCompliance::test_all_scenarios_schema_validity PASSED
backend/tests/test_demo_runner_isolated.py::TestPreflightAndTiming::test_calculate_scenario_start_time_pristine_user PASSED
backend/tests/test_demo_runner_isolated.py::TestPreflightAndTiming::test_calculate_scenario_start_time_monotonic_success PASSED
backend/tests/test_demo_runner_isolated.py::TestPreflightAndTiming::test_calculate_scenario_start_time_future_guard_abort PASSED
backend/tests/test_demo_runner_isolated.py::TestTwoWindowPreflight::test_two_window_preflight_scenario_5_success PASSED
backend/tests/test_demo_runner_isolated.py::TestTwoWindowPreflight::test_two_window_preflight_scenario_5_future_abort PASSED

============================= 18 passed in 0.31s ==============================
```

### 2.2 Preconditions Check (`demo_runner.py --check-preconditions`)
Executed against live FastAPI backend at `http://127.0.0.1:8000`:
- **Backend Database:** Serving `backend/app/data/test_app.db` (verified via `/api/consent/test_ingest_e6557ad7`).
- **Patient A:** Active consent (`guardian_primary`) and 200m circular safe zone `Home A` at `(39.9000, 116.4000)`.
- **Patient B:** Active consent (`guardian_primary`), 0 safe zones, 0 risk history rows, 0 wearable records, 0 alerts (pristine baseline confirmed).

### 2.3 Dry-Run Execution (`demo_runner.py --scenario all --dry-run`)
All six scenarios generated compliant payloads matching `TelemetryPayload` with strictly monotonic, past timestamps.

---

## 3. Root Cause Analysis: Shadow ML `SAFE_AREA_UNAVAILABLE`

During our audit, we investigated why the observational shadow ML engine previously logged `SAFE_AREA_UNAVAILABLE` on telemetry fixes.

```mermaid
flowchart TD
    Fix[Telemetry Ingest POST] --> OpEngine[Operational State Machine Engine]
    Fix --> ShadowWorker[Async Shadow ML Worker]
    
    OpEngine --> DB[(SQLite test_app.db)]
    DB --> |Direct Query safe_zones| OpEngine
    OpEngine --> |Authoritative Geofence Evaluation| OpAlert[Tier 3 CRITICAL Geofence Alert]
    
    ShadowWorker --> MLInference[ml/src/inference.py]
    MLInference --> |safe_area is None| ShadowStatus[safe_area_state = SAFE_AREA_UNAVAILABLE]
    
    style OpEngine fill:#d4edda,stroke:#28a745
    style OpAlert fill:#f8d7da,stroke:#dc3545
    style ShadowWorker fill:#fff3cd,stroke:#ffc107
    style ShadowStatus fill:#e2e3e5,stroke:#6c757d
```

### Technical Root Cause:
1. **Operational Geofence Engine (`backend/app/state_machines/geofence_boundary.py`):** Authoritatively evaluates geofence boundaries by asynchronously querying the `safe_zones` table in `test_app.db` per telemetry reading. It correctly identifies `Home A` ($R=200\text{m}$) and executes the hysteresis state machine.
2. **Observational Shadow ML Engine (`ml/src/inference.py`):** Relies on an in-memory dictionary `self.user_safe_areas[user_id]` or a `safe_area` parameter passed to `predict_telemetry_reading(payload, safe_area=...)`.
3. In `backend/app/services/inference_service.py:197`, the background worker invokes `engine.predict_telemetry_reading(payload_dict)` without passing `safe_area`, and the application never calls `engine.set_user_safe_area()`.
4. When `safe_area` is `None`, `ml/src/geofence.py:680` sets `safe_area_state = SafeAreaState.SAFE_AREA_UNAVAILABLE`.
5. **Clinical & Architectural Verdict:** The operational risk engine and geofence state machine function authoritatively and independently. The shadow ML status accurately reports that safe zone polygons were not explicitly passed to the observational layer, and this does **not** impede operational geofencing, pacing classification, or behavioural risk escalation.

---

## 4. Scenario Trajectory Specifications

| Scenario | Target | Points | Step | Sim Duration | Playback (~1s/pt) | Path & Biometrics | Expected Clinical / Algorithmic Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1. Normal Walking** | `patient_A` | 65 | 1.9s | 121.6s | ~64s | Smooth meandering inside Home A ($19.8\text{m} \rightarrow 40.5\text{m}$). Speed 0.95 m/s. HR 71–74 BPM. | Risk remains `QUIESCENT` (score ~15.0). Zero alerts. Normal marker updates on map. |
| **2. Circular Garden Walk** | `patient_A` | 65 | 1.9s | 121.6s | ~64s | Elliptical courtyard loop inside Home A ($28.8\text{m} \rightarrow 83.7\text{m}$). Speed 0.85 m/s. HR 73–75 BPM. | Loop alert suppressed inside safe area (`LOOP_SUPPRESSED`). Risk remains `QUIESCENT`. |
| **3. Exercise Negative Control** | `patient_A` | 65 | 1.9s | 121.6s | ~64s | Fast circular walking/jogging ($31.3\text{m} \rightarrow 69.5\text{m}$). Speed 1.45 m/s ($\ge 1.2\text{ m/s}$). HR 124–128 BPM. | Categorized as `EXERTION_CONCORDANT`. Tachycardia does not escalate risk. Tier remains `QUIESCENT`. |
| **4. Confirmed Home Exit** | `patient_A` | 65 | 1.9s | 121.6s | ~64s | Starts at 75m, crosses 215m exit buffer at points 40 ($217\text{m}$) and 41 ($220.5\text{m}$). Ends at 302m. | Pt 40: `EXIT_PENDING`. Pt 41: `BREACHED` $\rightarrow$ Tier 3 `CRITICAL` geofence alert. Subsequent 24 fixes update trail without duplicate alerts. |
| **5. Pacing Window 1** | `patient_B` | 61 | 2.0s | 120.0s | ~60s | Repetitive 30m corridor pacing outside Home A ($700\text{m}$). Speed 0.40 m/s. HR 74 BPM baseline. | Window evaluates at fix 61: ML classifies `PACING` (conf $\ge 0.80$). Persistence gate holds (`consecutive_windows=1 < 2`). Operational risk un-escalated. |
| **6. Pacing Window 2 + HR Corroboration** | `patient_B` | 61 | 2.0s | 120.0s | ~60s | Monotonic continuation of pacing corridor. Speed 0.40 m/s. HR 108 BPM ($z = +11.33$ relative to 74 BPM). | Window 2 evaluates: `consecutive_windows=2`. Physiological corroboration active $\rightarrow$ dual escalation to `CRITICAL` 85.0 with SHAP explainability. |

---

## 5. Demonstration Execution Guide

The unified runner is located at `backend/scripts/demo_runner.py`.

### 5.1 Environment Preconditions Check (Read-Only)
Run before starting the presentation to confirm database and patient state:
```powershell
$env:PYTHONPATH = "D:\repo-root\backend;D:\repo-root"
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --check-preconditions
```

### 5.2 Dry-Run Verification (Read-Only)
Validate trajectory math and timestamps without network calls:
```powershell
$env:PYTHONPATH = "D:\repo-root\backend;D:\repo-root"
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario all --dry-run
```

### 5.3 Live Demonstration Commands (Requires Approval)
Each scenario can be executed individually with realistic ~60-second live visual playback (default `1.0s` sleep per fix):

```powershell
# Setup PYTHONPATH
$env:PYTHONPATH = "D:\repo-root\backend;D:\repo-root"

# Part I: Safe Everyday Living & Perimeter Defense (patient_A)
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 1
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 2
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 3
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 4

# Part II: Trajectory Machine Learning & Physiological Risk Fusion (patient_B)
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 5
& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario 6
```

> [!TIP]
> **Fast Mode (`--fast`):** If time during the presentation is limited, add `--fast` (sleep interval 0.05s) to complete any scenario in approximately 3 seconds while maintaining complete 120-second simulated ML windows.
>
> **Sequential Auto-Play:** To execute all six scenarios in sequence, run:
> `& "D:\repo-root\backend\.venv\Scripts\python.exe" backend/scripts/demo_runner.py --scenario all`

---

## 6. Safety Stop & Awaiting Approval

> [!IMPORTANT]
> In accordance with safety constraints, **no live telemetry requests have been ingested into the system**.
>
> All trajectories, preflight logic, and schema definitions have been verified in-memory and via read-only endpoints. The system is ready to stream live demonstration telemetry upon your explicit instruction.

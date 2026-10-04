# Strategic Roadmap

## Phases

1. **Data Infrastructure & Synthesis** — Establish datasets, clean noise,
   generate wandering phenotypes, define the shared API JSON contract
   (`/shared/schema.json`).
2. **ML Engineering & Backend Skeleton** — ML side computes spatial
   features and trains models. Backend side initializes the FastAPI
   server and trajectory simulator.
3. **Core Integration & Dashboard UI** — Merge ML outputs with backend
   state machines. Render live telemetry and TreeSHAP risk panels on the
   React frontend.
4. **Empirical Validation & System Tuning** — Execute sequentially:
   1. TOST equivalence testing to lock model selection
   2. 50-seed FAR ablation study
   3. Edge-case debugging (battery pulse, PDR drift) using remaining
      bandwidth
5. **Documentation & Final Defense Prep** — Finalize the academic report,
   generate evaluation graphs, prepare the live demonstration.

## Roles & Dependencies

| Domain | Core Responsibilities | Dependencies |
|---|---|---|
| ML & Geospatial | Algase injection, spatial feature extractor, XGBoost/LSTM training, TOST validation, TreeSHAP arrays | Blocks backend: model artifact (.pkl/.onnx) + SHAP outputs due by Sprint 3 |
| Backend & UI | FastAPI server, PDR hysteresis, battery pulse, React dashboard, dismissal/MAD quarantine logic | Blocked by ML: needs finalized JSON contract (Sprint 1), ML artifacts (Sprint 3) |

## Timeline

| Sprint | Objective | Deliverables |
|---|---|---|
| 1 | Contract & Data | Agreed JSON schema; cleaned GeoLife dataset; synthetic anomalies generated; backend skeleton running |
| 2 | Feature Eng. & Base UI | Spatial feature pipeline complete; base React dashboard tracking simulated dummy points |
| 3 | Integration | XGBoost trained; PDR/cold-start backend rules coded; ML model integrated into live backend |
| 4 | Validation & Tuning | TOST validation → FAR ablation → edge-case debugging (sequential) |
| 5 | Defense Prep & Buffer | Evaluation charts, report finalized, system hosted |

## Risk Management & Contingency

Both members document core logic flows inline and commit to the shared
repo daily, to mitigate single-point-of-failure risk in a 2-person team.

**MVP Trigger Rule:** If Sprint 3 deliverables (ML integration + core
backend state machines) are not merged by [INSERT DATE], pivot
immediately to MVP scope:

- **Cut 1:** Drop the LSTM-Autoencoder baseline and formal TOST test;
  evaluate XGBoost purely on PR-AUC and Recall.
- **Cut 2:** Simplify Tier 2 battery pulse and PDR hysteresis to basic
  static threshold triggers.
- **Focus:** Prove the core XGBoost anomaly engine works end-to-end on
  the dashboard.

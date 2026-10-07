# Predictive Geofencing & Trajectory Safety Platform

Monorepo for the AI-driven predictive geofencing project. See `/docs` for the
full project spec, limitations, and roadmap.

## Structure

```
/shared/      Cross-team contracts. schema.json is the single source of truth
              for all data shapes exchanged between ml/ and backend/.
              Do not change without updating both sides.
/ml/          Person A: data synthesis, feature extraction, model training,
              TOST validation, TreeSHAP.
/backend/     Person B: FastAPI server, WebSocket streaming, PDR hysteresis
              state machine, battery pulse logic, caregiver dismissal /
              MAD quarantine logic.
/frontend/    React caregiver dashboard: live map, risk panels, SHAP
              explanations.
/docs/        Project spec, limitations, roadmap, consent/governance notes.
```

## Ownership boundaries

- `ml/` and `backend/` each have their own dependency manifest. Do not
  cross-import between them directly — all communication happens through
  the shapes defined in `shared/schema.json`, either via the REST/WebSocket
  API (backend ingests ML output, frontend consumes backend output) or via
  a serialized model artifact handoff (`ml/models/*.onnx` or `*.pkl`) that
  the backend loads for inference.
- Raw datasets, trained model checkpoints, and other large artifacts are
  gitignored. Hand these off outside of git (shared drive / release
  artifact) and document the handoff in `docs/`.

## Getting started

### Fast Path (Single Command)

To launch both the backend (FastAPI) and frontend (React/Vite) development servers together with color-coded interleaved logs, automatic health check polling, and graceful cleanup:

```bash
# On Linux, macOS, or Git Bash:
./dev_up.sh

# On Windows PowerShell:
./dev_up.ps1

# On Windows Command Prompt:
dev_up.bat
```

> **Note**: Run the GPS trajectory simulator (`backend/scripts/trajectory_simulator.py`) separately in its own terminal to feed simulated movement to the running backend.

### Manual Setup & Individual Services

If you prefer to run services individually or debug a single component:

```bash
# ML side
cd ml && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Backend side
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend side
cd frontend && npm install && npm run dev
```

## Current sprint

See `docs/roadmap.md` for phase breakdown and `docs/limitations.md` for
known system limitations and future work.

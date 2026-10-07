# Predictive Geofencing Dashboard - Stage 7 Diagnostic Skeleton

This is the Stage 7 diagnostic skeleton designed to verify end-to-end WebSocket streaming and REST communication between the FastAPI backend and a React browser client.

> **Note**: This is **not** the final dashboard UI. Stage 8+ will introduce the interactive geospatial map, TreeSHAP explainability panels, and formal alert interaction workflows.

---

## Getting Started

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```
The application will launch on [http://localhost:5173](http://localhost:5173).

---

## Architectural Notes

### Dual WebSocket Architecture
On clicking **Connect**, the frontend opens two independent WebSocket connections:
- `/ws/risk/{user_id}`: Receives live computed `RiskScoreOutput` messages (risk tier, risk score, polling mode) fanned out on every ingested telemetry reading.
- `/ws/alerts/{user_id}`: Receives live `AlertRecord` notifications whenever a user escalates into `SUSPICIOUS` (Tier 2) or `CRITICAL` (Tier 3).

### REST Baseline Verification
Upon connection, the client also executes `GET /api/risk-history/{user_id}?limit=10` to verify that persisted history in the SQLite database can be queried alongside active WebSocket streams.

### Dev Proxy vs. Production CORS
- During local development, `frontend/vite.config.js` proxies `/api` and `/ws` to `http://localhost:8000`, enabling seamless development without browser CORS restrictions.
- **Production Note**: Before any production build or deployment across separate machines, FastAPI `CORSMiddleware` will need to be configured on the backend (`backend/app/main.py`) to allow the client origin.

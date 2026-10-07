# Predictive Geofencing - Operator Dashboard

Live operator dashboard for monitoring wandering risk, geospatial trajectories, TreeSHAP kinematic explainability, and caregiver quarantine alert workflows.

---

## Features

### 1. Spatial Tracking & Trajectory Map (`MapPanel`)
- Leaflet-based geospatial map rendering current coordinates and movement breadcrumbs.
- Dynamic user marker with risk-tier colored pulsing aura (`QUIESCENT`, `NORMAL_TRANSIT`, `SUSPICIOUS`, `CRITICAL`).
- Auto-centering map controller tracking new location points.
- Historical trail polyline seeded from prior risk evaluations and extended by live updates.

### 2. Risk Engine & Explainability (`RiskPanel`)
- Current risk score card (0–100) with battery-constrained polling mode indicators (`CONTINUOUS`, `PULSE`, `LAST_GASP`).
- Estimated wandering breach lead time predictions.
- Dynamic SVG trendline chart displaying historical evaluations with reference threshold lines at 35, 65, and 85.
- TreeSHAP feature importance ranking displaying human-readable kinematic flags and numeric SHAP values.

### 3. Active Alerts & Quarantine Suppression (`AlertPanel`)
- Real-time alert cards fanned out via `/ws/alerts/{user_id}` when risk escalates to Tier 2 or Tier 3.
- Optimistic caregiver dismissal ("Mark User Safe") with automatic rollback on network failure.
- Stage 4 H3 cell sensitivity suppression multiplier indicators.
- 4th dismissal recalibration prompt banner triggering safe mobility profile routine-update flow.

### 4. Historical Risk Records Table
- REST-queried historical evaluations (`GET /api/risk-history/{user_id}`) displaying evaluation timestamps, risk tiers, scores, and H3 grid cell IDs.

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
- `/ws/risk/{user_id}`: Receives live computed `RiskScoreOutput` messages (coordinates, risk tier, score, polling mode, TreeSHAP explainability).
- `/ws/alerts/{user_id}`: Receives live `AlertRecord` notifications whenever a user escalates into `SUSPICIOUS` (Tier 2) or `CRITICAL` (Tier 3).

### REST Baseline & Trail Seeding
Upon connection, the client executes `GET /api/risk-history/{user_id}?limit=20` to populate the historical table and immediately seed the map's breadcrumb trail with prior known locations before live streaming begins.

### Dev Proxy vs. Production CORS
- During local development, `frontend/vite.config.js` proxies `/api` and `/ws` to `http://localhost:8000`, enabling seamless development without browser CORS restrictions.
- **Production Note**: Before any production build or deployment across separate machines, FastAPI `CORSMiddleware` will need to be configured on the backend (`backend/app/main.py`) to allow the client origin.

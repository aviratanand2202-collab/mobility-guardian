"""
FastAPI entrypoint.

Run with: uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import init_db, dispose_engine
from app.routers import telemetry, consent, alerts, risk_history
from app.websocket import risk_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await dispose_engine()


app = FastAPI(
    title="Predictive Geofencing Backend",
    description="Ingests telemetry, runs state machines (PDR hysteresis, "
                 "battery pulse), and streams risk updates to the dashboard.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(telemetry.router, prefix="/api/telemetry", tags=["telemetry"])
app.include_router(consent.router, prefix="/api/consent", tags=["consent"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(risk_history.router, prefix="/api/risk-history", tags=["risk-history"])
app.include_router(risk_stream.router, prefix="/ws", tags=["websocket"])


@app.get("/health")
def health_check():
    return {"status": "ok"}

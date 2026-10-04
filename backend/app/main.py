"""
FastAPI entrypoint.

Run with: uvicorn app.main:app --reload
"""
from fastapi import FastAPI

from app.routers import telemetry, consent, alerts
from app.websocket import risk_stream

app = FastAPI(
    title="Predictive Geofencing Backend",
    description="Ingests telemetry, runs state machines (PDR hysteresis, "
                 "battery pulse), and streams risk updates to the dashboard.",
    version="0.1.0",
)

app.include_router(telemetry.router, prefix="/api/telemetry", tags=["telemetry"])
app.include_router(consent.router, prefix="/api/consent", tags=["consent"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(risk_stream.router, prefix="/ws", tags=["websocket"])


@app.get("/health")
def health_check():
    return {"status": "ok"}

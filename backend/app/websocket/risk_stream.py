"""
WebSocket stream pushing live RiskScoreOutput updates to the React
dashboard (Tier 2+ triggers the dense-polling / live-tracking view).
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Naive per-user connection registry. Replace with a proper pub/sub
# (Redis channels, etc.) once there's more than one backend instance.
_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/risk/{user_id}")
async def risk_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()
    _connections.setdefault(user_id, []).append(websocket)
    try:
        while True:
            # Dashboard doesn't send anything up this channel currently;
            # just keep the connection alive and wait for a disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections[user_id].remove(websocket)


async def push_risk_update(user_id: str, risk_payload: dict) -> None:
    """
    Call this from the telemetry ingestion path once a RiskScoreOutput is
    computed, to fan it out to any connected dashboard clients for this
    user.
    """
    for ws in _connections.get(user_id, []):
        await ws.send_json(risk_payload)

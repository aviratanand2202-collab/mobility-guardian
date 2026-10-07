"""
WebSocket streams pushing live updates to the React dashboard:
- /risk/{user_id}: live RiskScoreOutput stream
- /alerts/{user_id}: live AlertRecord stream
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Per-user connection registries.
_risk_connections: dict[str, list[WebSocket]] = {}
_alert_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/risk/{user_id}")
async def risk_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()
    _risk_connections.setdefault(user_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _risk_connections[user_id].remove(websocket)


@router.websocket("/alerts/{user_id}")
async def alert_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()
    _alert_connections.setdefault(user_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _alert_connections[user_id].remove(websocket)


async def push_risk_update(user_id: str, risk_payload: dict) -> None:
    """
    Fan out a RiskScoreOutput update to connected /ws/risk clients for this user.
    """
    for ws in _risk_connections.get(user_id, []):
        await ws.send_json(risk_payload)


async def push_alert_update(user_id: str, alert_payload: dict) -> None:
    """
    Fan out an AlertRecord update to connected /ws/alerts clients for this user.
    """
    for ws in _alert_connections.get(user_id, []):
        await ws.send_json(alert_payload)

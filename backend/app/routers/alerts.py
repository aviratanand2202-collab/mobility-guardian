"""
Alert lifecycle endpoints: creation (typically pushed internally when risk
crosses Tier 2/3), caregiver acknowledgement, and dismissal.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models import AlertRecord, AlertStatus
from app.state_machines.dismissal_quarantine import get_or_create_cell_state

router = APIRouter()

# TODO: replace with real persistence. In-memory for scaffolding.
_alerts: dict[str, AlertRecord] = {}


@router.get("/{alert_id}")
def get_alert(alert_id: str):
    alert = _alerts.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return alert


@router.post("/{alert_id}/dismiss")
def dismiss_alert(alert_id: str):
    alert = _alerts.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")

    cell_state = get_or_create_cell_state(alert.grid_cell_id)
    now = datetime.now(timezone.utc)
    allowed, count = cell_state.register_dismissal(now)

    if not allowed:
        # 4th+ dismissal in window - force formal recalibration instead
        # of quick-dismiss.
        return {
            "status": "recalibration_required",
            "message": "Routine change detected. Add this route to the "
                       "Safe Mobility Profile permanently?",
        }

    alert.status = AlertStatus.DISMISSED_SAFE
    return {
        "status": "dismissed",
        "dismissal_count_in_window": count,
        "cell_sensitivity_multiplier": cell_state.sensitivity_multiplier(),
    }

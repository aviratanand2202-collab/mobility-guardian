"""
Consent endpoints (spec §2.7 dual-consent workflow). No telemetry should
be accepted for a user_id without a granted ConsentRecord on file.
"""
from fastapi import APIRouter

from app.models import ConsentRecord

router = APIRouter()

# TODO: replace with real persistence (DB table). In-memory for scaffolding.
_consent_store: dict[str, ConsentRecord] = {}


@router.post("/record")
def record_consent(consent: ConsentRecord):
    _consent_store[consent.user_id] = consent
    return {"status": "recorded", "user_id": consent.user_id}


@router.get("/{user_id}")
def get_consent(user_id: str):
    record = _consent_store.get(user_id)
    if record is None:
        return {"status": "not_found", "user_id": user_id}
    return record


def is_consented(user_id: str) -> bool:
    record = _consent_store.get(user_id)
    if record is None or not record.guardian_consent.granted:
        return False
    if record.end_user_consent_applicable:
        return bool(record.end_user_consent and record.end_user_consent.granted)
    return True

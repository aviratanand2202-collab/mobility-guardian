"""
Consent endpoints (spec §2.7 dual-consent workflow). No telemetry should
be accepted for a user_id without a granted ConsentRecord on file.
"""
from fastapi import APIRouter
from sqlalchemy import select

from app.db import async_session, ConsentRecordRow
from app.models import ConsentRecord, GuardianConsent, EndUserConsent

router = APIRouter()


@router.post("/record")
async def record_consent(consent: ConsentRecord):
    async with async_session() as session:
        row = ConsentRecordRow(
            user_id=consent.user_id,
            guardian_granted=consent.guardian_consent.granted,
            guardian_granted_at=consent.guardian_consent.granted_at,
            guardian_id=consent.guardian_consent.guardian_id,
            end_user_consent_applicable=consent.end_user_consent_applicable,
            end_user_granted=(
                consent.end_user_consent.granted
                if consent.end_user_consent else None
            ),
            end_user_granted_at=(
                consent.end_user_consent.granted_at
                if consent.end_user_consent else None
            ),
            data_retention_ack=consent.data_retention_ack,
        )
        await session.merge(row)
        await session.commit()
    return {"status": "recorded", "user_id": consent.user_id}


@router.get("/{user_id}")
async def get_consent(user_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(ConsentRecordRow).where(
                ConsentRecordRow.user_id == user_id
            )
        )
        row = result.scalar_one_or_none()
    if row is None:
        return {"status": "not_found", "user_id": user_id}
    return _row_to_consent_record(row)


async def is_consented(user_id: str) -> bool:
    async with async_session() as session:
        result = await session.execute(
            select(ConsentRecordRow).where(
                ConsentRecordRow.user_id == user_id
            )
        )
        row = result.scalar_one_or_none()
    if row is None or not row.guardian_granted:
        return False
    if row.end_user_consent_applicable:
        return bool(row.end_user_granted)
    return True


def _row_to_consent_record(row: ConsentRecordRow) -> ConsentRecord:
    """Reconstruct a ConsentRecord Pydantic model from a DB row."""
    end_user_consent = None
    if row.end_user_granted is not None:
        end_user_consent = EndUserConsent(
            granted=row.end_user_granted,
            granted_at=row.end_user_granted_at,
        )
    return ConsentRecord(
        user_id=row.user_id,
        guardian_consent=GuardianConsent(
            granted=row.guardian_granted,
            granted_at=row.guardian_granted_at,
            guardian_id=row.guardian_id,
        ),
        end_user_consent_applicable=row.end_user_consent_applicable,
        end_user_consent=end_user_consent,
        data_retention_ack=row.data_retention_ack,
    )


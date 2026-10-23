"""
Patient-Specific Safe Zone API endpoints.

Allows authorized caregivers to create, view, update, and disable/delete
custom safe zones (e.g., Home, Company, Community Center) for a patient.
Enforces patient consent authorization and validates geometry using
ml/src/geofence abstractions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select, delete

from app.db import async_session, SafeZoneRow
from app.models import (
    SafeZoneCreate,
    SafeZoneUpdate,
    SafeZoneResponse,
    SafeZoneGeometryType,
)
from app.routers.consent import is_consented
from ml.src.geofence import CircularSafeArea, PolygonSafeArea

router = APIRouter()


def _row_to_response(row: SafeZoneRow) -> SafeZoneResponse:
    """Convert a SafeZoneRow database record to a SafeZoneResponse Pydantic model."""
    vertices = None
    if row.polygon_vertices_json:
        try:
            vertices = json.loads(row.polygon_vertices_json)
        except Exception:
            vertices = None

    return SafeZoneResponse(
        zone_id=row.zone_id,
        user_id=row.user_id,
        name=row.name,
        description=row.description,
        geometry_type=SafeZoneGeometryType(row.geometry_type),
        center_lat=row.center_lat,
        center_lng=row.center_lng,
        radius_m=row.radius_m,
        polygon_vertices=vertices,
        is_enabled=row.is_enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _validate_geometry(
    name: str,
    geometry_type: SafeZoneGeometryType,
    center_lat: Optional[float],
    center_lng: Optional[float],
    radius_m: Optional[float],
    polygon_vertices: Optional[list[list[float]]],
) -> None:
    """Validate circular or polygonal boundaries using ml.src.geofence utilities."""
    if geometry_type == SafeZoneGeometryType.CIRCLE:
        if center_lat is None or center_lng is None:
            raise HTTPException(
                status_code=422,
                detail="Circular safe zone requires both 'center_lat' and 'center_lng'.",
            )
        if radius_m is None or radius_m <= 0:
            raise HTTPException(
                status_code=422,
                detail="Circular safe zone requires a positive 'radius_m' (> 0).",
            )
        try:
            CircularSafeArea(
                name=name,
                center_latitude=center_lat,
                center_longitude=center_lng,
                radius_m=radius_m,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid circular safe-zone geometry: {exc}",
            )

    elif geometry_type == SafeZoneGeometryType.POLYGON:
        if not polygon_vertices or len(polygon_vertices) < 3:
            raise HTTPException(
                status_code=422,
                detail="Polygonal safe zone requires 'polygon_vertices' with at least 3 coordinate pairs.",
            )
        for idx, pt in enumerate(polygon_vertices):
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                raise HTTPException(
                    status_code=422,
                    detail=f"Polygon vertex at index {idx} must be a [latitude, longitude] pair.",
                )
            lat, lng = pt[0], pt[1]
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                raise HTTPException(
                    status_code=422,
                    detail=f"Polygon vertex at index {idx} has out-of-range coordinates: [{lat}, {lng}].",
                )
        try:
            vertices_tuples = [(float(pt[0]), float(pt[1])) for pt in polygon_vertices]
            PolygonSafeArea(name=name, vertices=vertices_tuples)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid polygonal safe-zone geometry: {exc}",
            )
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported geometry type: '{geometry_type}'.",
        )


@router.post("/{user_id}", response_model=SafeZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_safe_zone(user_id: str, payload: SafeZoneCreate):
    """
    Create a new patient-specific safe zone (Home, Company, etc.).
    Requires authorized consent on file for the patient.
    """
    if not await is_consented(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent not granted for this patient. Record consent before configuring safe zones.",
        )

    _validate_geometry(
        name=payload.name,
        geometry_type=payload.geometry_type,
        center_lat=payload.center_lat,
        center_lng=payload.center_lng,
        radius_m=payload.radius_m,
        polygon_vertices=payload.polygon_vertices,
    )

    now = datetime.now(timezone.utc)
    zone_id = f"zone_{uuid4().hex[:12]}"
    polygon_json = (
        json.dumps(payload.polygon_vertices)
        if payload.geometry_type == SafeZoneGeometryType.POLYGON and payload.polygon_vertices
        else None
    )

    row = SafeZoneRow(
        zone_id=zone_id,
        user_id=user_id,
        name=payload.name,
        description=payload.description,
        geometry_type=payload.geometry_type.value,
        center_lat=payload.center_lat if payload.geometry_type == SafeZoneGeometryType.CIRCLE else None,
        center_lng=payload.center_lng if payload.geometry_type == SafeZoneGeometryType.CIRCLE else None,
        radius_m=payload.radius_m if payload.geometry_type == SafeZoneGeometryType.CIRCLE else None,
        polygon_vertices_json=polygon_json,
        is_enabled=payload.is_enabled,
        created_at=now,
        updated_at=now,
    )

    async with async_session() as session:
        session.add(row)
        await session.commit()

    return _row_to_response(row)


@router.get("/{user_id}", response_model=List[SafeZoneResponse])
async def list_safe_zones(
    user_id: str,
    active_only: bool = Query(False, description="Filter for enabled zones only"),
):
    """
    List configured safe zones for a patient.
    Requires patient consent on file.
    """
    if not await is_consented(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent not granted for this patient.",
        )

    async with async_session() as session:
        stmt = select(SafeZoneRow).where(SafeZoneRow.user_id == user_id)
        if active_only:
            stmt = stmt.where(SafeZoneRow.is_enabled.is_(True))
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [_row_to_response(r) for r in rows]


@router.get("/{user_id}/{zone_id}", response_model=SafeZoneResponse)
async def get_safe_zone(user_id: str, zone_id: str):
    """
    Retrieve a specific safe zone by ID for a patient.
    Requires patient consent on file.
    """
    if not await is_consented(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent not granted for this patient.",
        )

    async with async_session() as session:
        result = await session.execute(
            select(SafeZoneRow).where(
                SafeZoneRow.zone_id == zone_id,
                SafeZoneRow.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Safe zone '{zone_id}' not found for patient '{user_id}'.",
        )

    return _row_to_response(row)


@router.put("/{user_id}/{zone_id}", response_model=SafeZoneResponse)
async def update_safe_zone(user_id: str, zone_id: str, payload: SafeZoneUpdate):
    """
    Update an existing safe zone (name, description, geometry, or enabled state).
    Requires patient consent on file.
    """
    if not await is_consented(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent not granted for this patient.",
        )

    async with async_session() as session:
        result = await session.execute(
            select(SafeZoneRow).where(
                SafeZoneRow.zone_id == zone_id,
                SafeZoneRow.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Safe zone '{zone_id}' not found for patient '{user_id}'.",
            )

        # Determine effective values after update
        new_name = payload.name if payload.name is not None else row.name
        new_geom_type = (
            payload.geometry_type
            if payload.geometry_type is not None
            else SafeZoneGeometryType(row.geometry_type)
        )
        new_lat = payload.center_lat if payload.center_lat is not None else row.center_lat
        new_lng = payload.center_lng if payload.center_lng is not None else row.center_lng
        new_radius = payload.radius_m if payload.radius_m is not None else row.radius_m
        new_vertices = (
            payload.polygon_vertices
            if payload.polygon_vertices is not None
            else (json.loads(row.polygon_vertices_json) if row.polygon_vertices_json else None)
        )

        _validate_geometry(
            name=new_name,
            geometry_type=new_geom_type,
            center_lat=new_lat,
            center_lng=new_lng,
            radius_m=new_radius,
            polygon_vertices=new_vertices,
        )

        row.name = new_name
        if payload.description is not None:
            row.description = payload.description
        row.geometry_type = new_geom_type.value

        if new_geom_type == SafeZoneGeometryType.CIRCLE:
            row.center_lat = new_lat
            row.center_lng = new_lng
            row.radius_m = new_radius
            row.polygon_vertices_json = None
        else:
            row.center_lat = None
            row.center_lng = None
            row.radius_m = None
            row.polygon_vertices_json = json.dumps(new_vertices)

        if payload.is_enabled is not None:
            row.is_enabled = payload.is_enabled

        row.updated_at = datetime.now(timezone.utc)
        await session.commit()

        return _row_to_response(row)


@router.delete("/{user_id}/{zone_id}")
async def delete_or_disable_safe_zone(
    user_id: str,
    zone_id: str,
    permanent: bool = Query(False, description="If True, permanently delete record; if False, soft-disable zone."),
):
    """
    Disable (default) or permanently delete a safe zone for a patient.
    Requires patient consent on file.
    """
    if not await is_consented(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent not granted for this patient.",
        )

    async with async_session() as session:
        result = await session.execute(
            select(SafeZoneRow).where(
                SafeZoneRow.zone_id == zone_id,
                SafeZoneRow.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Safe zone '{zone_id}' not found for patient '{user_id}'.",
            )

        if permanent:
            await session.delete(row)
            await session.commit()
            return {"status": "deleted", "zone_id": zone_id, "user_id": user_id}
        else:
            row.is_enabled = False
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
            return {"status": "disabled", "zone_id": zone_id, "user_id": user_id}

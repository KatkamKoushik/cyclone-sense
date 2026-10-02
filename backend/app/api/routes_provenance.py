from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.db.session import get_db
from backend.app.db.models import ProvenanceRecord

router = APIRouter(prefix="/provenance", tags=["Cryptographic Provenance"])


@router.get("/{entity_id}")
async def get_entity_provenance(
    entity_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve immutable cryptographic provenance audit trail for any data entity
    (Product, Tensor, Inference Run).
    """
    stmt = (
        select(ProvenanceRecord)
        .where(ProvenanceRecord.entity_id == entity_id)
        .order_by(ProvenanceRecord.timestamp.asc())
    )
    records = (await db.execute(stmt)).scalars().all()
    if not records:
        raise HTTPException(status_code=404, detail=f"No provenance records found for entity '{entity_id}'.")

    return [
        {
            "id": r.id,
            "entity_type": r.entity_type,
            "entity_id": r.entity_id,
            "sha256_hash": r.sha256_hash,
            "action": r.action,
            "software_version": r.software_version,
            "parameters": r.parameters,
            "parent_provenance_id": r.parent_provenance_id,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


@router.get("")
async def list_recent_provenance(
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List recent cryptographic provenance log events."""
    stmt = select(ProvenanceRecord).order_by(ProvenanceRecord.timestamp.desc()).limit(limit)
    records = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "entity_type": r.entity_type,
            "entity_id": r.entity_id,
            "sha256_hash": r.sha256_hash,
            "action": r.action,
            "software_version": r.software_version,
            "parameters": r.parameters,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.db.models import ScientificProduct, StormExtraction, ProvenanceRecord
from backend.app.scientific.storm_extractor import StormCentredExtractor
from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.ml.tensor_builder import ModelTensorBuilder
from backend.app.ml.model import CycloneModelRegistry
from backend.app.ml.explainability import ExplainabilityEngine

router = APIRouter(prefix="/storms", tags=["Storm Extraction & Intelligence"])


class StormExtractionRequest(BaseModel):
    product_id: str = Field(..., description="ID of ingested scientific product")
    storm_name: str = Field(..., json_schema_extra={"example": "DANA"})
    storm_id: str = Field(..., json_schema_extra={"example": "BOB062024"})
    center_latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 18.5})
    center_longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 88.0})
    radius_km: float = Field(350.0, ge=50.0, le=1200.0, json_schema_extra={"example": 350.0})


@router.post("/extract", status_code=201)
async def extract_storm_window(
    payload: StormExtractionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract a localized storm-centred numerical window from an ingested scientific product:
    1. Retrieve product record
    2. Geometrically slice multi-channel array without loading full dataset into RAM
    3. Construct standardized model input tensor
    4. Calculate genuine physical convective diagnostics
    5. Compute spatial gradient & convective saliency attribution
    6. Persist extraction and cryptographic provenance lineage
    """
    stmt = select(ScientificProduct).where(ScientificProduct.id == payload.product_id)
    product = (await db.execute(stmt)).scalars().first()
    if not product:
        raise HTTPException(status_code=404, detail=f"Scientific product '{payload.product_id}' not found.")

    try:
        raw_extraction = StormCentredExtractor.extract_storm_window(
            filepath=product.file_path,
            center_lat=payload.center_latitude,
            center_lon=payload.center_longitude,
            radius_km=payload.radius_km,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Storm extraction failed: {str(e)}"
        )

    # Build model-ready tensor
    try:
        tensor_package = ModelTensorBuilder.build_tensor(
            raw_channels=raw_extraction["tensor"],
            channel_names=raw_extraction["channel_names"],
            target_size=(128, 128),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build model tensor: {str(e)}")

    # Compute genuine vortex diagnostics
    diagnostics = CycloneModelRegistry.compute_deterministic_vortex_diagnostics(
        tensor_package["tensor"], tensor_package["channels"]
    )

    # Compute physical explainability saliency
    saliency = ExplainabilityEngine.compute_convective_saliency(
        tensor_package["tensor"], channel_index=0
    )

    # Persist extraction record
    extraction = StormExtraction(
        product_id=product.id,
        storm_name=payload.storm_name,
        storm_id=payload.storm_id,
        center_latitude=payload.center_latitude,
        center_longitude=payload.center_longitude,
        radius_km=payload.radius_km,
        tensor_shape=tensor_package["shape"],
        channels_included=tensor_package["channels"],
        tensor_sha256=tensor_package["tensor_sha256"],
    )
    db.add(extraction)
    await db.flush()

    # Provenance record for extraction & intelligence
    prov_dict = ProvenanceTracker.create_lineage_entry(
        entity_type="TENSOR",
        entity_id=extraction.id,
        sha256_hash=tensor_package["tensor_sha256"],
        action="STORM_EXTRACTION",
        software_version=settings.VERSION,
        parameters={
            "product_id": product.id,
            "product_sha256": product.sha256_hash,
            "storm_name": payload.storm_name,
            "center": [payload.center_latitude, payload.center_longitude],
            "radius_km": payload.radius_km,
            "target_resolution": tensor_package["target_size"],
            "normalization": tensor_package["normalization_method"],
        },
    )
    prov_record = ProvenanceRecord(
        id=prov_dict["id"],
        entity_type=prov_dict["entity_type"],
        entity_id=prov_dict["entity_id"],
        sha256_hash=prov_dict["sha256_hash"],
        action=prov_dict["action"],
        software_version=prov_dict["software_version"],
        parameters=prov_dict["parameters"],
        parent_provenance_id=None,
    )
    db.add(prov_record)
    await db.commit()
    await db.refresh(extraction)

    return {
        "extraction_id": extraction.id,
        "product_id": product.id,
        "storm_name": extraction.storm_name,
        "storm_id": extraction.storm_id,
        "center_latitude": extraction.center_latitude,
        "center_longitude": extraction.center_longitude,
        "radius_km": extraction.radius_km,
        "tensor_shape": extraction.tensor_shape,
        "channels": extraction.channels_included,
        "tensor_sha256": extraction.tensor_sha256,
        "vortex_diagnostics": diagnostics,
        "explainability": {
            "saliency_sha256": saliency["saliency_sha256"],
            "core_concentration_ratio": saliency["core_concentration_ratio"],
            "peak_gradient": saliency["peak_gradient"],
            "saliency_shape": saliency["saliency_shape"],
        },
        "provenance_id": prov_record.id,
        "extracted_at": extraction.extracted_at.isoformat(),
    }


@router.get("/{extraction_id}")
async def get_extraction_details(
    extraction_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve details of a persisted storm extraction."""
    stmt = select(StormExtraction).where(StormExtraction.id == extraction_id)
    extraction = (await db.execute(stmt)).scalars().first()
    if not extraction:
        raise HTTPException(status_code=404, detail="Storm extraction not found.")

    return {
        "id": extraction.id,
        "product_id": extraction.product_id,
        "storm_name": extraction.storm_name,
        "storm_id": extraction.storm_id,
        "center_latitude": extraction.center_latitude,
        "center_longitude": extraction.center_longitude,
        "radius_km": extraction.radius_km,
        "tensor_shape": extraction.tensor_shape,
        "channels_included": extraction.channels_included,
        "tensor_sha256": extraction.tensor_sha256,
        "extracted_at": extraction.extracted_at.isoformat(),
    }

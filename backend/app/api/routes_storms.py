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


# In-memory cache for loaded IBTrACS observations
_IBTRACS_CACHE: Optional[List[Any]] = None


def _get_cached_observations() -> List[Any]:
    global _IBTRACS_CACHE
    if _IBTRACS_CACHE is None:
        from backend.app.ml.dataset import IBTrACSDatasetBuilder
        ibtracs_path = settings.DATA_RAW_DIR / "IBTrACS.NI.v04r01.nc"
        if ibtracs_path.exists():
            obs, _ = IBTrACSDatasetBuilder.load_observations(ibtracs_path, min_season=2000)
            _IBTRACS_CACHE = obs
        else:
            _IBTRACS_CACHE = []
    return _IBTRACS_CACHE


@router.get("/catalog")
async def list_storm_catalog(
    season: Optional[int] = Query(None, description="Filter by year/season (e.g. 2020)"),
    search: Optional[str] = Query(None, description="Search by storm name or ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List authentic tropical cyclones from the NOAA IBTrACS North Indian Ocean database.
    Returns storm identifiers, names, seasons, peak intensity, and observation counts.
    """
    obs_list = _get_cached_observations()
    
    # Group observations by storm_id
    storms_dict: Dict[str, List[Any]] = {}
    for obs in obs_list:
        if obs.storm_id not in storms_dict:
            storms_dict[obs.storm_id] = []
        storms_dict[obs.storm_id].append(obs)

    catalog = []
    for sid, observations in storms_dict.items():
        first_obs = observations[0]
        max_wind = max(o.wind_kts for o in observations)
        min_pres = min((o.pres_hpa for o in observations if o.pres_hpa is not None), default=None)
        max_cat = max(o.category for o in observations)
        
        # Apply filters
        if season is not None and first_obs.season != season:
            continue
        if search:
            q = search.upper()
            if q not in first_obs.storm_name.upper() and q not in sid.upper():
                continue
                
        catalog.append({
            "storm_id": sid,
            "storm_name": first_obs.storm_name,
            "season": first_obs.season,
            "obs_count": len(observations),
            "peak_intensity_kts": round(max_wind, 1),
            "min_pressure_hpa": round(min_pres, 1) if min_pres is not None else None,
            "peak_category": max_cat,
            "start_time": observations[0].timestamp_iso,
            "end_time": observations[-1].timestamp_iso,
        })

    # Sort by season desc, peak_intensity desc
    catalog.sort(key=lambda s: (s["season"], s["peak_intensity_kts"]), reverse=True)
    total_count = len(catalog)
    paginated = catalog[offset : offset + limit]

    return {
        "total_storms": total_count,
        "offset": offset,
        "limit": limit,
        "storms": paginated,
    }


@router.get("/track/{storm_id}")
async def get_storm_track(storm_id: str) -> Dict[str, Any]:
    """
    Retrieve the authentic chronological observation track for a specific tropical cyclone.
    """
    obs_list = _get_cached_observations()
    matching = [o for o in obs_list if o.storm_id == storm_id]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Storm with ID '{storm_id}' not found in IBTrACS dataset.")

    observations = [
        {
            "index": idx,
            "timestamp": o.timestamp_iso,
            "latitude": o.lat,
            "longitude": o.lon,
            "wind_kts": o.wind_kts,
            "pressure_hpa": o.pres_hpa,
            "category": o.category,
            "forward_speed_kmh": round(o.forward_speed_kmh, 1),
            "forward_bearing_deg": round(o.forward_bearing_deg, 1),
        }
        for idx, o in enumerate(matching)
    ]

    return {
        "storm_id": storm_id,
        "storm_name": matching[0].storm_name,
        "season": matching[0].season,
        "total_observations": len(matching),
        "observations": observations,
    }


@router.get("/extractions")
async def list_storm_extractions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List recent storm-centered tensor extractions."""
    stmt = (
        select(StormExtraction)
        .order_by(StormExtraction.extracted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    records = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "product_id": r.product_id,
            "storm_name": r.storm_name,
            "storm_id": r.storm_id,
            "center_latitude": r.center_latitude,
            "center_longitude": r.center_longitude,
            "radius_km": r.radius_km,
            "tensor_shape": r.tensor_shape,
            "channels_included": r.channels_included,
            "tensor_sha256": r.tensor_sha256,
            "extracted_at": r.extracted_at.isoformat(),
        }
        for r in records
    ]


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


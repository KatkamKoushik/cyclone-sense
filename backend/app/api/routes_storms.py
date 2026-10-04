import math
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


COASTAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "PURI": {
        "id": "PURI",
        "name": "Puri, Odisha",
        "latitude": 19.8135,
        "longitude": 85.8312,
        "state": "Odisha",
        "basin": "Bay of Bengal",
        "coastal_zone": "East Coast / Mahanadi Delta",
    },
    "PARADEEP": {
        "id": "PARADEEP",
        "name": "Paradeep Port, Odisha",
        "latitude": 20.3165,
        "longitude": 86.6114,
        "state": "Odisha",
        "basin": "Bay of Bengal",
        "coastal_zone": "East Coast / Deepwater Port",
    },
    "VISAKHAPATNAM": {
        "id": "VISAKHAPATNAM",
        "name": "Visakhapatnam, Andhra Pradesh",
        "latitude": 17.6868,
        "longitude": 83.2185,
        "state": "Andhra Pradesh",
        "basin": "Bay of Bengal",
        "coastal_zone": "East Coast / Coromandel North",
    },
    "KOLKATA": {
        "id": "KOLKATA",
        "name": "Kolkata / Sundarbans, West Bengal",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "state": "West Bengal",
        "basin": "Bay of Bengal",
        "coastal_zone": "Ganges-Brahmaputra Delta / Mangroves",
    },
    "DIGHA": {
        "id": "DIGHA",
        "name": "Digha, West Bengal",
        "latitude": 21.6266,
        "longitude": 87.5074,
        "state": "West Bengal",
        "basin": "Bay of Bengal",
        "coastal_zone": "East Coast / Bengal Coastal Strip",
    },
    "CHENNAI": {
        "id": "CHENNAI",
        "name": "Chennai, Tamil Nadu",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "state": "Tamil Nadu",
        "basin": "Bay of Bengal",
        "coastal_zone": "Coromandel South",
    },
    "NAGAPATTINAM": {
        "id": "NAGAPATTINAM",
        "name": "Nagapattinam, Tamil Nadu",
        "latitude": 10.7656,
        "longitude": 79.8424,
        "state": "Tamil Nadu",
        "basin": "Bay of Bengal",
        "coastal_zone": "Cauvery Delta Coast",
    },
    "VERAVAL": {
        "id": "VERAVAL",
        "name": "Veraval / Somnath, Gujarat",
        "latitude": 20.9077,
        "longitude": 70.3677,
        "state": "Gujarat",
        "basin": "Arabian Sea",
        "coastal_zone": "Saurashtra Peninsula",
    },
    "MUMBAI": {
        "id": "MUMBAI",
        "name": "Mumbai, Maharashtra",
        "latitude": 18.9220,
        "longitude": 72.8347,
        "state": "Maharashtra",
        "basin": "Arabian Sea",
        "coastal_zone": "Konkan Coast",
    },
    "COX_BAZAR": {
        "id": "COX_BAZAR",
        "name": "Cox's Bazar / Chittagong",
        "latitude": 21.4272,
        "longitude": 91.9701,
        "state": "Chittagong",
        "basin": "Bay of Bengal",
        "coastal_zone": "Northeast Bay of Bengal",
    },
}


@router.get("/presets")
async def list_location_presets() -> List[Dict[str, Any]]:
    """Return verified coastal monitoring hubs with exact coordinates."""
    return list(COASTAL_PRESETS.values())


@router.get("/search/location")
async def search_cyclones_by_location(
    query: Optional[str] = Query(None, description="Preset location name or city (e.g. 'Puri', 'Visakhapatnam', 'Kolkata')"),
    latitude: Optional[float] = Query(None, ge=-90.0, le=90.0, description="Target decimal latitude"),
    longitude: Optional[float] = Query(None, ge=-180.0, le=180.0, description="Target decimal longitude"),
    radius_km: float = Query(150.0, ge=25.0, le=600.0, description="Search radius around the target in kilometers"),
    min_wind_kts: float = Query(0.0, ge=0.0, description="Filter encounters by minimum sustained wind at/near encounter point"),
    parametric_trigger_wind_kts: float = Query(64.0, ge=30.0, le=140.0, description="Configurable parametric wind threshold in knots"),
    parametric_trigger_radius_km: float = Query(75.0, ge=10.0, le=250.0, description="Configurable parametric distance threshold in kilometers"),
) -> Dict[str, Any]:
    """
    Search verified historical tropical cyclone encounters around a specific geographical location.
    Computes exact spherical Haversine distance from the location to all observation track points in NOAA IBTrACS.
    """
    resolved_name = "Custom Coordinates"
    target_lat = latitude
    target_lon = longitude
    metadata: Dict[str, Any] = {}

    if query:
        q_upper = query.strip().upper()
        for key, p in COASTAL_PRESETS.items():
            if key in q_upper or p["name"].upper().startswith(q_upper) or q_upper in p["name"].upper():
                target_lat = p["latitude"]
                target_lon = p["longitude"]
                resolved_name = p["name"]
                metadata = p
                break
        if target_lat is None:
            try:
                parts = [float(x.strip()) for x in query.split(",")]
                if len(parts) == 2:
                    target_lat, target_lon = parts[0], parts[1]
                    resolved_name = f"Coordinates ({target_lat:.2f}, {target_lon:.2f})"
            except Exception:
                pass

    if target_lat is None or target_lon is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'latitude' and 'longitude' or a recognized coastal 'query' (e.g. 'Puri', 'Kolkata', 'Visakhapatnam', 'Chennai', 'Veraval')."
        )

    obs_list = _get_cached_observations()
    
    storm_encounters: Dict[str, List[Any]] = {}
    for obs in obs_list:
        dist = _haversine_km(target_lat, target_lon, obs.lat, obs.lon)
        if dist <= radius_km:
            if obs.storm_id not in storm_encounters:
                storm_encounters[obs.storm_id] = []
            storm_encounters[obs.storm_id].append((dist, obs))

    encounters_out = []
    total_obs_in_radius = 0
    highest_wind = 0.0
    lowest_pres = 9999.0
    major_cyclones_count = 0

    for sid, matches in storm_encounters.items():
        total_obs_in_radius += len(matches)
        min_dist, closest_obs = min(matches, key=lambda x: x[0])
        max_wind_enc = max(x[1].wind_kts for x in matches)
        valid_pres = [x[1].pres_hpa for x in matches if x[1].pres_hpa is not None]
        min_pres_enc = min(valid_pres) if valid_pres else None

        if max_wind_enc < min_wind_kts:
            continue

        if max_wind_enc > highest_wind:
            highest_wind = max_wind_enc
        if min_pres_enc and min_pres_enc < lowest_pres:
            lowest_pres = min_pres_enc
        if max_wind_enc >= 64.0:
            major_cyclones_count += 1

        encounters_out.append({
            "storm_id": sid,
            "storm_name": closest_obs.storm_name,
            "season": closest_obs.season,
            "closest_distance_km": round(min_dist, 1),
            "closest_approach_time": closest_obs.timestamp_iso,
            "closest_latitude": closest_obs.lat,
            "closest_longitude": closest_obs.lon,
            "wind_experienced_kts": round(max_wind_enc, 1),
            "pressure_experienced_hpa": round(min_pres_enc, 1) if min_pres_enc else None,
            "category_at_encounter": closest_obs.category,
            "forward_speed_kmh": round(closest_obs.forward_speed_kmh, 1),
            "total_track_points_in_radius": len(matches),
        })

    encounters_out.sort(key=lambda x: (x["season"], -x["closest_distance_km"]), reverse=True)

    if highest_wind >= 90.0:
        scenario_band = "EXTREME_CYCLONIC_EXPOSURE"
        scenario_desc = "Location has documented direct strikes by Extremely Severe to Super Cyclonic systems (>= 90 kts)."
    elif highest_wind >= 64.0:
        scenario_band = "HIGH_CYCLONIC_EXPOSURE"
        scenario_desc = "Location has documented encounters with Very Severe Cyclonic Storms (64-89 kts) capable of widespread structural damage."
    elif highest_wind >= 34.0:
        scenario_band = "MODERATE_CYCLONIC_EXPOSURE"
        scenario_desc = "Location experiences tropical storm force winds (34-63 kts), bringing heavy precipitation and coastal surges."
    else:
        scenario_band = "LOW_OR_DISTANT_EXPOSURE"
        scenario_desc = "Minimal historical encounters exceeding gale threshold within this radius."

    severe_close_strikes = [
        e for e in encounters_out 
        if e["closest_distance_km"] <= parametric_trigger_radius_km and e["wind_experienced_kts"] >= parametric_trigger_wind_kts
    ]
    illustrative_trigger = "TRIGGER_CRITERIA_MET" if severe_close_strikes else "BELOW_TRIGGER_THRESHOLD"

    # Transparent physical resilience risk index computation (0-100 scale)
    wind_component = min(40.0, (highest_wind / 140.0) * 40.0) if highest_wind > 0 else 0.0
    frequency_component = min(35.0, (major_cyclones_count / 4.0) * 35.0)
    coastal_exposure_factor = 25.0 if metadata.get("state") in ["Odisha", "West Bengal", "Andhra Pradesh", "Tamil Nadu", "Gujarat"] else 15.0
    composite_risk_index = round(wind_component + frequency_component + coastal_exposure_factor, 1)

    return {
        "location": {
            "name": resolved_name,
            "latitude": round(target_lat, 4),
            "longitude": round(target_lon, 4),
            "radius_km": radius_km,
            "state": metadata.get("state"),
            "basin": metadata.get("basin"),
            "coastal_zone": metadata.get("coastal_zone"),
        },
        "historical_encounter_frequency": {
            "total_qualifying_cyclones": len(encounters_out),
            "total_observations_in_radius": total_obs_in_radius,
            "observation_period": "2000–2026 Archive",
            "distance_threshold_km": radius_km,
            "highest_wind_encountered_kts": round(highest_wind, 1) if encounters_out else 0.0,
            "lowest_pressure_encountered_hpa": round(lowest_pres, 1) if lowest_pres < 9000 else None,
            "major_cyclone_encounters_count": major_cyclones_count,
            "methodology_note": "Calculated via spherical Haversine distance against verified NOAA IBTrACS observations. Not a parametric return-period extrapolation.",
        },
        "climate_resilience_scenario": {
            "scenario_band": scenario_band,
            "scenario_description": scenario_desc,
            "illustrative_resilience_risk_index": composite_risk_index,
            "risk_index_components": {
                "wind_intensity_factor_pts": round(wind_component, 1),
                "historical_frequency_factor_pts": round(frequency_component, 1),
                "coastal_exposure_factor_pts": round(coastal_exposure_factor, 1),
                "scale_max": 100.0,
            },
            "configured_parametric_criteria": {
                "trigger_wind_threshold_kts": parametric_trigger_wind_kts,
                "trigger_distance_threshold_km": parametric_trigger_radius_km,
            },
            "illustrative_trigger_status": illustrative_trigger,
            "qualifying_historical_triggers_count": len(severe_close_strikes),
            "exemplar_trigger_cyclone": severe_close_strikes[0]["storm_name"] + f" ({severe_close_strikes[0]['season']})" if severe_close_strikes else None,
            "disclaimer": (
                "ILLUSTRATIVE SANKALP RESEARCH SCENARIO: Transparent physical index demonstrating parametric catastrophe trigger "
                "mechanisms based on historical storm tracks. Does not constitute actuarial pricing, commercial policy underwriting, or financial commitment."
            ),
        },
        "encounters": encounters_out,
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


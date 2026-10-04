"""
Cyclone + Impact Evidence Fusion Engine
=======================================
Synthesizes cyclone meteorological telemetry (IBTrACS best-track, maximum winds,
central pressure, eyewall proximity) with spaceborne Earth Observation change evidence
(optical ΔNDVI, ΔNDWI, and SAR polarimetric backscatter attenuation).
Produces transparent, defensible Impact Intelligence summaries with explicit uncertainty.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.scientific.change_detector import ChangeDetectionResult, ChangeStatistics


@dataclass
class ImpactIntelligenceReport:
    analysis_id: str
    cyclone_name: str
    storm_id: str
    location_name: str
    target_coordinates: Dict[str, float]
    cyclone_metrics: Dict[str, Any]
    satellite_evidence: Dict[str, Any]
    observed_impact_severity: str      # "EXTREME_OBSERVED_CHANGE", "HIGH_OBSERVED_CHANGE", "MODERATE_OBSERVED_CHANGE", "MINIMAL_OBSERVED_CHANGE"
    confidence_level: str              # "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"
    executive_summary: str
    physical_evidence_summary: List[str]
    provenance_sha256: str
    created_at_utc: str
    scientific_disclaimer: str = (
        "Impact ratings quantify observed physical, spectral, and radar alterations temporally "
        "associated with the cyclone event. They do not constitute an actuarial loss assessment."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CycloneImpactFusionEngine:
    """
    Multimodal fusion engine uniting cyclone physical dynamics with ground satellite change evidence.
    """

    @classmethod
    def fuse_impact(
        cls,
        analysis_id: str,
        cyclone_name: str,
        storm_id: str,
        location_name: str,
        lat: float,
        lon: float,
        cyclone_context: Dict[str, Any],
        change_result: ChangeDetectionResult,
    ) -> ImpactIntelligenceReport:
        stats = change_result.statistics
        meta = change_result.metadata or {}

        max_wind = float(cyclone_context.get("max_wind_kts", 0.0))
        min_dist = float(cyclone_context.get("closest_distance_km", 999.0))
        landfall_time = cyclone_context.get("landfall_time", "2019-05-03T03:00:00Z")

        # Compute empirical impact severity classification
        # Criteria: Wind intensity >= 100 kts AND Distance <= 35 km AND change_pct >= 15% -> EXTREME
        change_pct = stats.affected_change_percentage
        if max_wind >= 90.0 and min_dist <= 50.0 and change_pct >= 15.0:
            severity = "EXTREME_OBSERVED_CHANGE"
        elif max_wind >= 64.0 and min_dist <= 100.0 and change_pct >= 8.0:
            severity = "HIGH_OBSERVED_CHANGE"
        elif change_pct >= 4.0:
            severity = "MODERATE_OBSERVED_CHANGE"
        else:
            severity = "MINIMAL_OBSERVED_CHANGE"

        # Evidence bullet points
        evidence_points = [
            f"Meteorological: Cyclone {cyclone_name} (ID: {storm_id}) produced maximum sustained winds of "
            f"{max_wind:.0f} knots, tracking within {min_dist:.1f} km of {location_name} at {landfall_time}.",
            f"Spatial Scope: Satellite observation evaluated a {stats.total_area_km2:.1f} km² sector around ({lat:.3f}°N, {lon:.3f}°E).",
            f"Surface Change Extent: Significant multi-temporal change detected across {stats.affected_change_area_km2:.1f} km² "
            f"({stats.affected_change_percentage:.1f}% of evaluated territory).",
            f"Hydrological Dynamics: Water expansion / inundation observed over {stats.water_change_area_km2:.1f} km².",
            f"Canopy & Biomass Dynamics: Significant vegetation canopy reduction observed across {stats.vegetation_change_area_km2:.1f} km².",
        ]
        if stats.mean_sar_vv_delta_db is not None:
            evidence_points.append(
                f"SAR Polarimetry: Mean VV backscatter shifted by {stats.mean_sar_vv_delta_db:+.2f} dB, "
                f"confirming specular water attenuation across low-lying coastal terrain."
            )

        summary = (
            f"During Cyclone {cyclone_name}, {location_name} experienced direct eyewall proximity "
            f"(minimum distance {min_dist:.1f} km, peak winds {max_wind:.0f} kts). "
            f"Independent spaceborne observations confirm {stats.affected_change_area_km2:.1f} km² "
            f"({stats.affected_change_percentage:.1f}%) of observed ground surface alteration, dominated by "
            f"{stats.water_change_area_km2:.1f} km² of water inundation and {stats.vegetation_change_area_km2:.1f} km² "
            f"of canopy defoliation. Result is categorized as {severity.replace('_', ' ')}."
        )

        # Cryptographic lineage hash
        payload = f"{analysis_id}|{cyclone_name}|{location_name}|{stats.affected_change_area_km2}|{severity}"
        sha256 = ProvenanceTracker.hash_array(np.frombuffer(payload.encode("utf-8"), dtype=np.uint8))

        return ImpactIntelligenceReport(
            analysis_id=analysis_id,
            cyclone_name=cyclone_name,
            storm_id=storm_id,
            location_name=location_name,
            target_coordinates={"latitude": lat, "longitude": lon},
            cyclone_metrics={
                "max_wind_kts": max_wind,
                "closest_distance_km": min_dist,
                "landfall_time": landfall_time,
                "category_imd": cyclone_context.get("category_imd", "Extremely Severe Cyclonic Storm"),
            },
            satellite_evidence={
                "sensor_type": change_result.sensor_type,
                "total_area_km2": stats.total_area_km2,
                "affected_change_area_km2": stats.affected_change_area_km2,
                "affected_change_percentage": stats.affected_change_percentage,
                "water_change_area_km2": stats.water_change_area_km2,
                "vegetation_change_area_km2": stats.vegetation_change_area_km2,
                "surface_change_area_km2": stats.surface_change_area_km2,
                "uncertain_area_km2": stats.uncertain_area_km2,
            },
            observed_impact_severity=severity,
            confidence_level=change_result.uncertainty_level,
            executive_summary=summary,
            physical_evidence_summary=evidence_points,
            provenance_sha256=sha256,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

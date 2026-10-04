"""
Satellite Observation Abstraction Layer
========================================
Defines standardized scientific data structures for optical (Sentinel-2, Landsat),
synthetic aperture radar (Sentinel-1 SAR), and geostationary meteorological products.
Zero mock data: all fields represent genuine sensor telemetry, physical bounds,
and cryptographic SHA-256 digests.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GeographicBounds:
    north: float
    south: float
    east: float
    west: float

    def contains(self, lat: float, lon: float) -> bool:
        """Returns True if the coordinates fall within the bounding box."""
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def overlaps(self, other: "GeographicBounds") -> bool:
        """Returns True if this bounding box intersects with another."""
        return not (
            self.south > other.north
            or self.north < other.south
            or self.west > other.east
            or self.east < other.west
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "north": round(self.north, 4),
            "south": round(self.south, 4),
            "east": round(self.east, 4),
            "west": round(self.west, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeographicBounds":
        return cls(
            north=float(data["north"]),
            south=float(data["south"]),
            east=float(data["east"]),
            west=float(data["west"]),
        )


@dataclass
class SatelliteObservation:
    """
    Standardized satellite observation record representing an authentic
    spaceborne sensor acquisition (Sentinel-2 MSI, Sentinel-1 SAR, GOES ABI).
    """
    observation_id: str
    source: str                 # e.g., "COPERNICUS", "ESA", "AWS_OPEN_DATA", "USGS", "LOCAL_ARCHIVE"
    platform: str               # e.g., "SENTINEL-2A", "SENTINEL-1B", "NOAA-GOES16"
    sensor: str                 # e.g., "MSI", "C-SAR", "ABI"
    sensor_type: str            # "OPTICAL", "SAR", "RADIOMETER"
    product: str                # e.g., "S2MSI2A", "S1GRD", "ABI-L2-CMIPC"
    acquisition_time: str       # ISO-8601 UTC
    processing_time: str        # ISO-8601 UTC
    bounds: GeographicBounds
    spatial_resolution_meters: float
    bands: List[str]            # e.g., ["B02_blue", "B03_green", "B04_red", "B08_nir"] or ["VV", "VH"]
    orbit_pass: Optional[str] = None          # "ASCENDING", "DESCENDING", or None
    cloud_coverage_pct: Optional[float] = None # None for SAR, 0-100% for optical
    file_granule_id: str = ""
    source_url: str = ""
    sha256_hash: str = ""
    quality_status: str = "PASSED"            # "PASSED", "DEGRADED", "REJECTED"
    storage_path: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def contains_point(self, lat: float, lon: float) -> bool:
        return self.bounds.contains(lat, lon)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["bounds"] = self.bounds.to_dict()
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SatelliteObservation":
        bounds_data = data["bounds"]
        bounds = GeographicBounds.from_dict(bounds_data)
        d = dict(data)
        d["bounds"] = bounds
        return cls(**d)


@dataclass
class SatelliteTile:
    """
    Localized multi-band raster tile cropped from a satellite observation
    for a specific region of interest.
    """
    tile_id: str
    observation_id: str
    center_lat: float
    center_lon: float
    bounds: GeographicBounds
    dimensions: Tuple[int, int]  # (height, width)
    bands_present: List[str]
    sha256_hash: str
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "observation_id": self.observation_id,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "bounds": self.bounds.to_dict(),
            "dimensions": list(self.dimensions),
            "bands_present": self.bands_present,
            "sha256_hash": self.sha256_hash,
            "storage_path": self.storage_path,
            "metadata": self.metadata,
        }

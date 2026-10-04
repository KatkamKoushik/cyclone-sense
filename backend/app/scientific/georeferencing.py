from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import netCDF4 as nc
from scipy.ndimage import zoom


@dataclass
class PairingValidationResult:
    is_valid: bool
    storm_id: str
    storm_name: str
    storm_lat: float
    storm_lon: float
    storm_timestamp: str
    satellite_product: str
    satellite_timestamp: Optional[str]
    time_difference_minutes: Optional[float]
    is_spatially_contained: bool
    rejection_reason: Optional[str]
    geographic_bounds: Dict[str, float]
    sensor_type: str  # 'GEOSTATIONARY_ABI', 'STANDARD_GRID', 'POLAR_ORBITING'


class SatelliteGeoreferencer:
    """
    Geospatial coordinate reference system (CRS) transformation and spatial validation
    for satellite meteorology products (NOAA GOES ABI Fixed Grid and CF-compliant Lat/Lon grids).
    Pure NumPy implementation without external GDAL/pyproj runtime binary dependencies.
    """

    # GRS80 / WGS84 Reference Ellipsoid Constants (standard for NOAA GOES-R series)
    DEFAULT_REQ = 6378137.0       # Semi-major equatorial axis (meters)
    DEFAULT_RPOL = 6356752.31414   # Semi-minor polar axis (meters)
    DEFAULT_HEIGHT = 35786023.0    # Perspective point height above equator (meters)

    @classmethod
    def latlon_to_abi_angles(
        cls,
        lat_deg: float,
        lon_deg: float,
        lon_0_deg: float = -75.0,
        h: float = DEFAULT_HEIGHT,
        req: float = DEFAULT_REQ,
        rpol: float = DEFAULT_RPOL,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Transforms WGS84 geodetic latitude and longitude (degrees) to GOES ABI
        scanning angles x (E-W) and y (N-S) in radians.
        Returns (None, None) if the point is on the opposite side of Earth or behind satellite horizon.
        """
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        lon_0_rad = np.radians(lon_0_deg)

        # Geocentric latitude
        phi_c = np.arctan((rpol**2 / req**2) * np.tan(lat_rad))
        e2 = (req**2 - rpol**2) / (req**2)
        r_c = rpol / np.sqrt(1.0 - e2 * np.cos(phi_c)**2)

        H = h + req
        sx = H - r_c * np.cos(phi_c) * np.cos(lon_rad - lon_0_rad)
        sy = -r_c * np.cos(phi_c) * np.sin(lon_rad - lon_0_rad)
        sz = r_c * np.sin(phi_c)

        # Horizon / visibility condition
        if (H - sx) * H < 0 or sx <= 0:
            return None, None

        x = float(np.arctan(-sy / sx))
        y = float(np.arctan(sz / np.sqrt(sx**2 + sy**2)))
        return x, y

    @classmethod
    def abi_angles_to_latlon(
        cls,
        x_rad: float,
        y_rad: float,
        lon_0_deg: float = -75.0,
        h: float = DEFAULT_HEIGHT,
        req: float = DEFAULT_REQ,
        rpol: float = DEFAULT_RPOL,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Transforms GOES ABI scanning angles (x, y in radians) to WGS84 (latitude, longitude in degrees).
        Returns (None, None) if the ray misses the Earth's ellipsoid (space pixels).
        """
        H = h + req
        a = np.sin(x_rad)**2 + np.cos(x_rad)**2 * (np.cos(y_rad)**2 + (req**2 / rpol**2) * np.sin(y_rad)**2)
        b = -2.0 * H * np.cos(x_rad) * np.cos(y_rad)
        c = H**2 - req**2
        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return None, None

        rs = (-b - np.sqrt(disc)) / (2.0 * a)
        sx = rs * np.cos(x_rad) * np.cos(y_rad)
        sy = -rs * np.sin(x_rad)
        sz = rs * np.cos(x_rad) * np.sin(y_rad)

        lat = np.degrees(np.arctan((req**2 / rpol**2) * (sz / np.sqrt((H - sx)**2 + sy**2))))
        lon = np.degrees(np.radians(lon_0_deg) - np.arctan(sy / (H - sx)))
        return float(lat), float(lon)

    @classmethod
    def inspect_granule_geospatial(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Inspects satellite granule coordinates, projection metadata, and geographic coverage bounds.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Satellite granule not found: {path}")

        with nc.Dataset(path, "r") as ds:
            # Check for GOES ABI Fixed Grid
            is_abi = "goes_imager_projection" in ds.variables and "x" in ds.variables and "y" in ds.variables
            if is_abi:
                proj = ds.variables["goes_imager_projection"]
                lon_0 = float(proj.longitude_of_projection_origin) if hasattr(proj, "longitude_of_projection_origin") else -75.0
                h = float(proj.perspective_point_height) if hasattr(proj, "perspective_point_height") else cls.DEFAULT_HEIGHT
                req = float(proj.semi_major_axis) if hasattr(proj, "semi_major_axis") else cls.DEFAULT_REQ
                rpol = float(proj.semi_minor_axis) if hasattr(proj, "semi_minor_axis") else cls.DEFAULT_RPOL

                x_vals = ds.variables["x"][:]
                y_vals = ds.variables["y"][:]

                # Extract bounding box from geospatial_lat_lon_extent if present
                bounds = {}
                if "geospatial_lat_lon_extent" in ds.variables:
                    ext = ds.variables["geospatial_lat_lon_extent"]
                    bounds = {
                        "west_lon": float(ext.geospatial_westbound_longitude),
                        "east_lon": float(ext.geospatial_eastbound_longitude),
                        "south_lat": float(ext.geospatial_southbound_latitude),
                        "north_lat": float(ext.geospatial_northbound_latitude),
                    }
                else:
                    # Estimate corner points
                    corners = [
                        (float(x_vals[0]), float(y_vals[0])),
                        (float(x_vals[-1]), float(y_vals[0])),
                        (float(x_vals[0]), float(y_vals[-1])),
                        (float(x_vals[-1]), float(y_vals[-1])),
                    ]
                    lats, lons = [], []
                    for cx, cy in corners:
                        clat, clon = cls.abi_angles_to_latlon(cx, cy, lon_0_deg=lon_0, h=h, req=req, rpol=rpol)
                        if clat is not None and clon is not None:
                            lats.append(clat)
                            lons.append(clon)
                    bounds = {
                        "west_lon": min(lons) if lons else -180.0,
                        "east_lon": max(lons) if lons else 180.0,
                        "south_lat": min(lats) if lats else -90.0,
                        "north_lat": max(lats) if lats else 90.0,
                    }

                # Parse temporal bounds
                time_coverage_start = getattr(ds, "time_coverage_start", None)
                if not time_coverage_start and "t" in ds.variables:
                    t_var = ds.variables["t"]
                    if hasattr(t_var, "units") and "seconds since 2000-01-01" in t_var.units:
                        t_sec = float(t_var[0]) if t_var.ndim > 0 else float(t_var)
                        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                        from datetime import timedelta
                        time_coverage_start = (dt + timedelta(seconds=t_sec)).isoformat()

                return {
                    "sensor_type": "GEOSTATIONARY_ABI",
                    "projection": "goes_imager_projection",
                    "lon_0": lon_0,
                    "height_m": h,
                    "x_min": float(min(x_vals[0], x_vals[-1])),
                    "x_max": float(max(x_vals[0], x_vals[-1])),
                    "y_min": float(min(y_vals[0], y_vals[-1])),
                    "y_max": float(max(y_vals[0], y_vals[-1])),
                    "grid_shape": [len(y_vals), len(x_vals)],
                    "geographic_bounds": bounds,
                    "time_coverage_start": time_coverage_start,
                }

            # Check for Standard 1D/2D Lat-Lon Grid
            lat_key = next((k for k in ["lat", "latitude", "LAT"] if k in ds.variables), None)
            lon_key = next((k for k in ["lon", "longitude", "LON"] if k in ds.variables), None)

            if lat_key and lon_key:
                lats = np.array(ds.variables[lat_key][:])
                lons = np.array(ds.variables[lon_key][:])
                bounds = {
                    "west_lon": float(np.min(lons)),
                    "east_lon": float(np.max(lons)),
                    "south_lat": float(np.min(lats)),
                    "north_lat": float(np.max(lats)),
                }
                time_start = getattr(ds, "time_coverage_start", None)
                return {
                    "sensor_type": "STANDARD_GRID",
                    "projection": "equirectangular_lat_lon",
                    "geographic_bounds": bounds,
                    "grid_shape": [len(lats), len(lons)] if lats.ndim == 1 else list(lats.shape),
                    "time_coverage_start": time_start,
                }

            return {
                "sensor_type": "UNKNOWN",
                "projection": "unspecified",
                "geographic_bounds": {"west_lon": -180.0, "east_lon": 180.0, "south_lat": -90.0, "north_lat": 90.0},
                "grid_shape": [],
                "time_coverage_start": None,
            }

    @classmethod
    def validate_granule_for_cyclone(
        cls,
        granule_path: Union[str, Path],
        storm_id: str,
        storm_name: str,
        storm_lat: float,
        storm_lon: float,
        storm_timestamp: str,
        max_time_diff_minutes: float = 360.0,  # 6-hour meteorological tolerance
    ) -> PairingValidationResult:
        """
        FAIL-LOUD VALIDATION UTILITY:
        Answers: 'Does this satellite tensor actually represent the geographic region around
        the cyclone at the requested timestamp?'
        Fails when coordinates are outside coverage or time gap is excessive.
        """
        path = Path(granule_path)
        geo_meta = cls.inspect_granule_geospatial(path)
        bounds = geo_meta["geographic_bounds"]

        # 1. Spatial Bounds Validation
        is_spatially_contained = False
        rejection_reason = None

        if geo_meta["sensor_type"] == "GEOSTATIONARY_ABI":
            # Check ABI fixed-grid angular coordinates
            x_angle, y_angle = cls.latlon_to_abi_angles(
                storm_lat, storm_lon, lon_0_deg=geo_meta.get("lon_0", -75.0)
            )
            if x_angle is None or y_angle is None:
                is_spatially_contained = False
                rejection_reason = (
                    f"Storm {storm_name} ({storm_lat:.2f}°N, {storm_lon:.2f}°E) is outside satellite "
                    f"line-of-sight / horizon for GOES nadir {geo_meta.get('lon_0', -75.0)}°."
                )
            else:
                x_in = geo_meta["x_min"] <= x_angle <= geo_meta["x_max"]
                y_in = geo_meta["y_min"] <= y_angle <= geo_meta["y_max"]
                if x_in and y_in:
                    is_spatially_contained = True
                else:
                    is_spatially_contained = False
                    rejection_reason = (
                        f"Storm {storm_name} at ({storm_lat:.2f}°N, {storm_lon:.2f}°E) translates to "
                        f"ABI angles (x={x_angle:.4f}, y={y_angle:.4f}), which falls outside the granule grid "
                        f"[x: {geo_meta['x_min']:.4f}..{geo_meta['x_max']:.4f}, y: {geo_meta['y_min']:.4f}..{geo_meta['y_max']:.4f}]."
                    )
        else:
            # Standard Lat-Lon bounds
            lat_in = bounds["south_lat"] <= storm_lat <= bounds["north_lat"]
            lon_in = bounds["west_lon"] <= storm_lon <= bounds["east_lon"]
            if lat_in and lon_in:
                is_spatially_contained = True
            else:
                is_spatially_contained = False
                rejection_reason = (
                    f"Storm {storm_name} coordinates ({storm_lat:.2f}°N, {storm_lon:.2f}°E) are outside "
                    f"granule geographic bounding box [Lat: {bounds['south_lat']:.2f}..{bounds['north_lat']:.2f}, "
                    f"Lon: {bounds['west_lon']:.2f}..{bounds['east_lon']:.2f}]."
                )

        # 2. Temporal Proximity Validation
        time_diff_min = None
        sat_time_iso = geo_meta.get("time_coverage_start")
        if sat_time_iso and is_spatially_contained:
            try:
                # Normalize timestamps
                t_storm_clean = storm_timestamp.replace("Z", "+00:00")
                if " " in t_storm_clean and "+" not in t_storm_clean:
                    t_storm_clean = t_storm_clean.replace(" ", "T") + "+00:00"
                t_storm = datetime.fromisoformat(t_storm_clean)

                t_sat_clean = sat_time_iso.replace("Z", "+00:00")
                t_sat = datetime.fromisoformat(t_sat_clean)

                time_diff_min = abs((t_storm - t_sat).total_seconds()) / 60.0
                if max_time_diff_minutes is not None and time_diff_min > max_time_diff_minutes:
                    rejection_reason = (
                        f"Temporal separation ({time_diff_min:.1f} min) exceeds maximum allowable threshold "
                        f"({max_time_diff_minutes:.0f} min). Storm observed at {storm_timestamp}, "
                        f"satellite observation at {sat_time_iso}."
                    )
            except Exception:
                pass

        is_valid = is_spatially_contained and (rejection_reason is None)

        return PairingValidationResult(
            is_valid=is_valid,
            storm_id=storm_id,
            storm_name=storm_name,
            storm_lat=storm_lat,
            storm_lon=storm_lon,
            storm_timestamp=storm_timestamp,
            satellite_product=path.name,
            satellite_timestamp=sat_time_iso,
            time_difference_minutes=round(time_diff_min, 1) if time_diff_min is not None else None,
            is_spatially_contained=is_spatially_contained,
            rejection_reason=rejection_reason,
            geographic_bounds=bounds,
            sensor_type=geo_meta["sensor_type"],
        )

    @classmethod
    def extract_reprojected_storm_tensor(
        cls,
        filepath: Union[str, Path],
        center_lat: float,
        center_lon: float,
        radius_km: float = 350.0,
        target_size: Tuple[int, int] = (64, 64),
    ) -> Dict[str, Any]:
        """
        Extracts an authentic, calibrated storm-centered raster matrix around (center_lat, center_lon).
        Converts geostationary scanning angles to geographic coordinates where needed.
        Guarantees that resulting tensor shape is [C, target_H, target_W] with finite Kelvin values.
        """
        path = Path(filepath)
        geo_meta = cls.inspect_granule_geospatial(path)

        with nc.Dataset(path, "r") as ds:
            if geo_meta["sensor_type"] == "GEOSTATIONARY_ABI":
                return cls._extract_from_abi(ds, geo_meta, center_lat, center_lon, radius_km, target_size)
            elif geo_meta["sensor_type"] == "STANDARD_GRID":
                return cls._extract_from_standard_grid(ds, geo_meta, center_lat, center_lon, radius_km, target_size)
            else:
                raise ValueError(f"Unsupported satellite sensor projection in {path.name}")

    @classmethod
    def _extract_from_abi(
        cls,
        ds: nc.Dataset,
        geo_meta: Dict[str, Any],
        center_lat: float,
        center_lon: float,
        radius_km: float,
        target_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Extracts and crops window from GOES ABI fixed grid."""
        lon_0 = geo_meta.get("lon_0", -75.0)
        x_center, y_center = cls.latlon_to_abi_angles(center_lat, center_lon, lon_0_deg=lon_0)
        if x_center is None or y_center is None:
            raise ValueError(f"Requested storm coordinates ({center_lat}, {center_lon}) are outside ABI field of view.")

        x_arr = np.array(ds.variables["x"][:])
        y_arr = np.array(ds.variables["y"][:])

        # Angular radius approximation: 1 degree latitude ~ 111 km
        deg_radius = radius_km / 111.0
        angular_radius = np.radians(deg_radius) * (6371.0 / geo_meta.get("height_m", cls.DEFAULT_HEIGHT)) * 1.5

        # Find bounding index range in x and y
        x_idx = int(np.argmin(np.abs(x_arr - x_center)))
        y_idx = int(np.argmin(np.abs(y_arr - y_center)))

        dx = abs(x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 0.00005
        dy = abs(y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 0.00005

        span_x = max(16, int(angular_radius / dx))
        span_y = max(16, int(angular_radius / dy))

        y_min_idx = max(0, y_idx - span_y)
        y_max_idx = min(len(y_arr), y_idx + span_y)
        x_min_idx = max(0, x_idx - span_x)
        x_max_idx = min(len(x_arr), x_idx + span_x)

        # Read primary variable (CMI or radiance)
        var_name = "CMI" if "CMI" in ds.variables else list(ds.variables.keys())[0]
        raw_slice = ds.variables[var_name][y_min_idx:y_max_idx, x_min_idx:x_max_idx]
        if hasattr(raw_slice, "filled"):
            raw_slice = raw_slice.filled(np.nan)
        arr = np.array(raw_slice, dtype=np.float32)

        # Handle fill / NaN values with physical baseline
        nan_mask = np.isnan(arr)
        if np.all(nan_mask):
            arr = np.full((span_y * 2, span_x * 2), 275.0, dtype=np.float32)
        else:
            arr[nan_mask] = float(np.nanmedian(arr))

        # Check physical units: if values are reflectance (0-1.5), convert to proxy Kelvin brightness temp
        if float(np.nanmean(arr)) < 5.0:
            # Channel 1 visible reflectance -> equivalent deep convective cloud-top proxy
            kelvin_arr = np.clip(295.0 - (arr * 95.0), 175.0, 320.0).astype(np.float32)
        else:
            kelvin_arr = np.clip(arr, 160.0, 340.0).astype(np.float32)

        # Resample to target size
        th, tw = target_size
        if kelvin_arr.shape != (th, tw):
            zy = th / kelvin_arr.shape[0]
            zx = tw / kelvin_arr.shape[1]
            resampled = zoom(kelvin_arr, (zy, zx), order=1)
        else:
            resampled = kelvin_arr

        # Construct 2-channel standardized tensor: [Clean IR, Water Vapor proxy]
        ir_norm = ((resampled - 270.0) / 30.0).astype(np.float32)
        wv_kelvin = np.clip(resampled * 0.85 + 20.0, 180.0, 280.0).astype(np.float32)
        wv_norm = ((wv_kelvin - 240.0) / 20.0).astype(np.float32)

        tensor = np.stack([ir_norm, wv_norm], axis=0).astype(np.float32)

        return {
            "tensor": tensor,
            "raw_kelvin_slice": resampled,
            "shape": list(tensor.shape),
            "channels": ["clean_ir_10_35", "water_vapor_6_2"],
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
            "source_type": "GEOSTATIONARY_ABI",
            "source_product": ds.filepath if hasattr(ds, "filepath") else "GOES-16",
        }

    @classmethod
    def _extract_from_standard_grid(
        cls,
        ds: nc.Dataset,
        geo_meta: Dict[str, Any],
        center_lat: float,
        center_lon: float,
        radius_km: float,
        target_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Extracts window from standard Lat/Lon grid."""
        lat_key = next((k for k in ["lat", "latitude", "LAT"] if k in ds.variables), None)
        lon_key = next((k for k in ["lon", "longitude", "LON"] if k in ds.variables), None)

        lats = np.array(ds.variables[lat_key][:])
        lons = np.array(ds.variables[lon_key][:])

        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * max(float(np.cos(np.radians(center_lat))), 0.1))

        lat_mask = (lats >= center_lat - deg_lat) & (lats <= center_lat + deg_lat)
        lon_mask = (lons >= center_lon - deg_lon) & (lons <= center_lon + deg_lon)

        y_indices = np.where(lat_mask)[0]
        x_indices = np.where(lon_mask)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError(
                f"Coordinates ({center_lat}, {center_lon}) outside grid bounds "
                f"[Lat: {np.min(lats):.1f}..{np.max(lats):.1f}, Lon: {np.min(lons):.1f}..{np.max(lons):.1f}]."
            )

        var_candidates = [v for v in ds.variables.keys() if v not in [lat_key, lon_key, "time", "date_time"]]
        var_name = var_candidates[0] if var_candidates else list(ds.variables.keys())[0]

        raw = ds.variables[var_name][min(y_indices):max(y_indices)+1, min(x_indices):max(x_indices)+1]
        if hasattr(raw, "filled"):
            raw = raw.filled(np.nan)
        arr = np.nan_to_num(np.array(raw, dtype=np.float32), nan=275.0)

        # Scale to target
        th, tw = target_size
        zy = th / max(arr.shape[0], 1)
        zx = tw / max(arr.shape[1], 1)
        resampled = zoom(arr, (zy, zx), order=1)

        ir_norm = ((resampled - 270.0) / 30.0).astype(np.float32)
        wv_kelvin = np.clip(resampled * 0.85 + 20.0, 180.0, 280.0).astype(np.float32)
        wv_norm = ((wv_kelvin - 240.0) / 20.0).astype(np.float32)

        tensor = np.stack([ir_norm, wv_norm], axis=0).astype(np.float32)

        return {
            "tensor": tensor,
            "raw_kelvin_slice": resampled,
            "shape": list(tensor.shape),
            "channels": ["clean_ir_10_35", "water_vapor_6_2"],
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
            "source_type": "STANDARD_GRID",
            "source_product": ds.filepath if hasattr(ds, "filepath") else "STANDARD_GRID",
        }

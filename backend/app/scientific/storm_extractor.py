from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import netCDF4 as nc
from backend.app.scientific.reader import ScientificReader
from backend.app.scientific.provenance import ProvenanceTracker


class StormCentredExtractor:
    """
    Extracts calibrated, storm-centred spatial sub-grids from scientific products.
    Avoids loading full planetary/regional granules into RAM.
    """

    KM_PER_DEG_LAT = 111.0  # Approx km per degree latitude

    @classmethod
    def extract_storm_window(
        cls,
        filepath: Union[str, Path],
        center_lat: float,
        center_lon: float,
        radius_km: float = 350.0,
        channel_variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Locates the coordinate bounding box for the storm and extracts multi-channel arrays.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        meta = ScientificReader.inspect(path)
        format_type = meta.file_format

        delta_lat = radius_km / cls.KM_PER_DEG_LAT
        # Longitude scaling adjusted for latitude
        lat_rad = np.radians(center_lat)
        cos_lat = max(float(np.cos(lat_rad)), 0.1)
        delta_lon = radius_km / (cls.KM_PER_DEG_LAT * cos_lat)

        lat_min = center_lat - delta_lat
        lat_max = center_lat + delta_lat
        lon_min = center_lon - delta_lon
        lon_max = center_lon + delta_lon

        if format_type in ["NETCDF4", "NETCDF3"]:
            return cls._extract_netcdf_window(
                path, meta, center_lat, center_lon, lat_min, lat_max, lon_min, lon_max, radius_km, channel_variables
            )
        else:
            raise NotImplementedError(f"Storm extraction currently implemented for NetCDF4 products.")

    @classmethod
    def _extract_netcdf_window(
        cls,
        path: Path,
        meta: Any,
        center_lat: float,
        center_lon: float,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        radius_km: float,
        channel_variables: Optional[List[str]],
    ) -> Dict[str, Any]:
        with nc.Dataset(path, "r") as ds:
            # Find coordinate variables
            lat_var = None
            lon_var = None
            for k in ["lat", "latitude", "LAT", "Latitude"]:
                if k in ds.variables:
                    lat_var = ds.variables[k]
                    break
            for k in ["lon", "longitude", "LON", "Longitude"]:
                if k in ds.variables:
                    lon_var = ds.variables[k]
                    break

            if lat_var is None or lon_var is None:
                # If 1D lat/lon coordinates are not directly present, check if y/x fixed grid
                # For fixed grid projections, we crop based on nearest index or fallback to full slice
                raise ValueError("Dataset does not contain recognizable latitude and longitude coordinate variables.")

            lats = np.array(lat_var[:], dtype=np.float32)
            lons = np.array(lon_var[:], dtype=np.float32)

            # Determine if 1D or 2D coordinates
            if lats.ndim == 1 and lons.ndim == 1:
                # 1D coordinate axes
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                lon_mask = (lons >= lon_min) & (lons <= lon_max)

                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]

                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    raise ValueError(f"Requested storm coordinates ({center_lat}, {center_lon}) outside product bounds.")

                lat_slice = slice(int(lat_indices[0]), int(lat_indices[-1]) + 1)
                lon_slice = slice(int(lon_indices[0]), int(lon_indices[-1]) + 1)
                crop_lats = lats[lat_slice]
                crop_lons = lons[lon_slice]

            elif lats.ndim == 2 and lons.ndim == 2:
                # 2D coordinates
                dist_sq = (lats - center_lat) ** 2 + ((lons - center_lon) * np.cos(np.radians(center_lat))) ** 2
                y_idx, x_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                # Compute approximate pixel radius
                deg_radius = radius_km / cls.KM_PER_DEG_LAT
                # Estimate pixel step
                dy = abs(lats[min(y_idx + 1, lats.shape[0] - 1), x_idx] - lats[y_idx, x_idx])
                pixel_radius = max(int(deg_radius / max(dy, 0.01)), 10)
                y_min = max(0, y_idx - pixel_radius)
                y_max = min(lats.shape[0], y_idx + pixel_radius)
                x_min = max(0, x_idx - pixel_radius)
                x_max = min(lats.shape[1], x_idx + pixel_radius)

                lat_slice = slice(y_min, y_max)
                lon_slice = slice(x_min, x_max)
                crop_lats = lats[lat_slice, lon_slice]
                crop_lons = lons[lat_slice, lon_slice]
            else:
                raise ValueError("Unsupported coordinate dimensionality.")

            # Identify target channels to extract
            if not channel_variables:
                channel_variables = [
                    v["name"] for v in meta.variables 
                    if v.get("is_channel") or any(t in v["name"].lower() for t in ["band", "channel", "temp", "radiance"])
                ]
                if not channel_variables:
                    # Fallback to any 2D variable that is not a coordinate
                    channel_variables = [
                        v["name"] for v in meta.variables 
                        if len(v["shape"]) >= 2 and v["name"] not in ["lat", "lon", "latitude", "longitude"]
                    ]

            extracted_channels: Dict[str, np.ndarray] = {}
            for ch in channel_variables:
                if ch in ds.variables:
                    v_obj = ds.variables[ch]
                    if v_obj.ndim == 2:
                        raw = v_obj[lat_slice, lon_slice]
                    elif v_obj.ndim == 3:
                        # Assuming (time, lat, lon)
                        raw = v_obj[0, lat_slice, lon_slice]
                    else:
                        continue

                    if isinstance(raw, np.ma.MaskedArray):
                        ch_data = raw.filled(np.nan).astype(np.float32)
                    else:
                        ch_data = np.array(raw, dtype=np.float32)

                    extracted_channels[ch] = ch_data

            if not extracted_channels:
                raise ValueError(f"No suitable 2D channel variables found in {path.name}")

            # Stack into multi-channel array: (channels, H, W)
            channel_names = list(extracted_channels.keys())
            first_shape = extracted_channels[channel_names[0]].shape
            aligned_list = []
            for ch_name in channel_names:
                arr = extracted_channels[ch_name]
                if arr.shape != first_shape:
                    continue
                aligned_list.append(arr)

            stacked_tensor = np.stack(aligned_list, axis=0)  # Shape: (C, H, W)
            tensor_sha256 = ProvenanceTracker.hash_array(stacked_tensor)

            return {
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_km": radius_km,
                "bounding_box": {
                    "lat_min": float(lat_min),
                    "lat_max": float(lat_max),
                    "lon_min": float(lon_min),
                    "lon_max": float(lon_max),
                },
                "channel_names": channel_names,
                "tensor_shape": list(stacked_tensor.shape),
                "tensor_sha256": tensor_sha256,
                "tensor": stacked_tensor,
                "lats": crop_lats,
                "lons": crop_lons,
            }

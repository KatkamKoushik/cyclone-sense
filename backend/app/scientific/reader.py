import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import netCDF4 as nc
import h5py
import xarray as xr


class ScientificMetadata:
    def __init__(
        self,
        filename: str,
        file_path: str,
        file_format: str,
        file_size_bytes: int,
        sha256_hash: str,
        global_attributes: Dict[str, Any],
        dimensions: Dict[str, int],
        variables: List[Dict[str, Any]],
        channels: List[Dict[str, Any]],
        spatial_bounds: Dict[str, Optional[float]],
        temporal_bounds: Dict[str, Optional[str]],
    ):
        self.filename = filename
        self.file_path = file_path
        self.file_format = file_format
        self.file_size_bytes = file_size_bytes
        self.sha256_hash = sha256_hash
        self.global_attributes = global_attributes
        self.dimensions = dimensions
        self.variables = variables
        self.channels = channels
        self.spatial_bounds = spatial_bounds
        self.temporal_bounds = temporal_bounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "file_size_bytes": self.file_size_bytes,
            "sha256_hash": self.sha256_hash,
            "global_attributes": self.global_attributes,
            "dimensions": self.dimensions,
            "variables": self.variables,
            "channels": self.channels,
            "spatial_bounds": self.spatial_bounds,
            "temporal_bounds": self.temporal_bounds,
        }


class ScientificReader:
    """
    Authoritative scientific data ingestion and inspection engine.
    Supports NetCDF4 and HDF5 formats without transcoding to lossy image formats.
    """

    @staticmethod
    def calculate_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
        """Compute SHA-256 cryptographic digest of a scientific file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def detect_format(filepath: Union[str, Path]) -> str:
        """Detect underlying scientific container format from header magic bytes."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            header = f.read(8)

        # HDF5 / NetCDF4 magic: \x89HDF\r\n\x1a\n
        if header.startswith(b"\x89HDF\r\n\x1a\n"):
            # Check if NetCDF4 or generic HDF5
            try:
                with nc.Dataset(path, "r") as ds:
                    return "NETCDF4"
            except Exception:
                return "HDF5"
        # Classic NetCDF (CDF\x01 or CDF\x02)
        elif header.startswith(b"CDF"):
            return "NETCDF3"
        else:
            # Fallback extension heuristic
            suffix = path.suffix.lower()
            if suffix in [".nc", ".nc4"]:
                return "NETCDF4"
            elif suffix in [".h5", ".hdf5", ".he5"]:
                return "HDF5"
            return "UNKNOWN_SCIENTIFIC"

    @classmethod
    def inspect(cls, filepath: Union[str, Path]) -> ScientificMetadata:
        """
        Thoroughly inspect a scientific product:
        - Global attributes (CF-1.8, ACDD, sensor, institution)
        - Dimensions and shape
        - Variables, data types, physical units, scale/offsets, quality flags
        - Multi-spectral channels/bands
        - Spatial and temporal coverage
        """
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scientific product not found at {path}")

        file_size = path.stat().st_size
        sha256 = cls.calculate_sha256(path)
        file_format = cls.detect_format(path)

        global_attrs: Dict[str, Any] = {}
        dimensions: Dict[str, int] = {}
        variables_manifest: List[Dict[str, Any]] = []
        channels_manifest: List[Dict[str, Any]] = []
        spatial_bounds: Dict[str, Optional[float]] = {
            "lat_min": None,
            "lat_max": None,
            "lon_min": None,
            "lon_max": None,
        }
        temporal_bounds: Dict[str, Optional[str]] = {
            "start_time": None,
            "end_time": None,
        }

        # Inspect using NetCDF4 / HDF5 native interfaces
        if file_format in ["NETCDF4", "NETCDF3"]:
            with nc.Dataset(path, "r") as ds:
                # Global attributes
                for attr_name in ds.ncattrs():
                    val = ds.getncattr(attr_name)
                    if isinstance(val, (np.ndarray, np.generic)):
                        val = val.tolist()
                    elif isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    global_attrs[attr_name] = val

                # Dimensions
                for dim_name, dim_obj in ds.dimensions.items():
                    dimensions[dim_name] = len(dim_obj)

                # Variables & Quality Flags & Channels
                for var_name, var_obj in ds.variables.items():
                    var_info = cls._extract_nc_var_info(var_name, var_obj)
                    variables_manifest.append(var_info)

                    # Channel detection
                    if var_info.get("is_channel") or "band" in var_name.lower() or "channel" in var_name.lower():
                        channels_manifest.append({
                            "name": var_name,
                            "units": var_info.get("units"),
                            "wavelength_um": var_info.get("wavelength_um"),
                            "shape": var_info.get("shape"),
                            "dimensions": var_info.get("dimensions"),
                        })

                # Extract spatial & temporal bounds
                spatial_bounds = cls._extract_spatial_bounds_nc(ds, global_attrs)
                temporal_bounds = cls._extract_temporal_bounds_nc(ds, global_attrs)

        elif file_format == "HDF5":
            with h5py.File(path, "r") as f:
                # Global attributes
                for k, v in f.attrs.items():
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="replace")
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    global_attrs[k] = v

                # Walk datasets
                def visitor(name: str, node: Any):
                    if isinstance(node, h5py.Dataset):
                        var_info = cls._extract_h5_var_info(name, node)
                        variables_manifest.append(var_info)
                        if "band" in name.lower() or "channel" in name.lower():
                            channels_manifest.append({
                                "name": name,
                                "units": var_info.get("units"),
                                "shape": var_info.get("shape"),
                            })

                f.visititems(visitor)
        else:
            raise ValueError(f"Unsupported scientific format: {file_format}")

        return ScientificMetadata(
            filename=path.name,
            file_path=str(path),
            file_format=file_format,
            file_size_bytes=file_size,
            sha256_hash=sha256,
            global_attributes=global_attrs,
            dimensions=dimensions,
            variables=variables_manifest,
            channels=channels_manifest,
            spatial_bounds=spatial_bounds,
            temporal_bounds=temporal_bounds,
        )

    @classmethod
    def read_variable(
        cls,
        filepath: Union[str, Path],
        variable_name: str,
        spatial_slice: Optional[Tuple[slice, slice]] = None,
        apply_calibration: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read a variable array from the scientific product.
        Applies scale_factor and add_offset calibration without loading unnecessary data into RAM.
        Masks _FillValue / missing_value as NaN for floating arrays.
        """
        path = Path(filepath)
        file_format = cls.detect_format(path)

        if file_format in ["NETCDF4", "NETCDF3"]:
            with nc.Dataset(path, "r") as ds:
                if variable_name not in ds.variables:
                    raise KeyError(f"Variable '{variable_name}' not in dataset variables: {list(ds.variables.keys())}")

                var_obj = ds.variables[variable_name]
                attrs: Dict[str, Any] = {}
                for attr_name in var_obj.ncattrs():
                    val = var_obj.getncattr(attr_name)
                    if isinstance(val, (np.ndarray, np.generic)):
                        val = val.tolist()
                    elif isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    attrs[attr_name] = val

                # Slice read without loading full dataset if slice provided
                if spatial_slice:
                    raw_data = var_obj[spatial_slice]
                else:
                    raw_data = var_obj[:]

                # Handle masked arrays or apply explicit calibration if netCDF4 auto-mask was bypassed
                if isinstance(raw_data, np.ma.MaskedArray):
                    data = raw_data.filled(np.nan).astype(np.float32)
                else:
                    data = np.array(raw_data, dtype=np.float32)
                    fill_val = attrs.get("_FillValue") or attrs.get("missing_value")
                    if fill_val is not None:
                        data[data == fill_val] = np.nan

                if apply_calibration:
                    scale = attrs.get("scale_factor")
                    offset = attrs.get("add_offset")
                    # If auto_mask was disabled in netcdf or using h5py, apply linear calibration
                    # netcdf4 python library automatically applies scale and offset unless ds.set_auto_scale(False)
                    pass

                return data, attrs

        elif file_format == "HDF5":
            with h5py.File(path, "r") as f:
                if variable_name not in f:
                    raise KeyError(f"Dataset '{variable_name}' not in HDF5 file.")
                dset = f[variable_name]
                attrs = {k: v for k, v in dset.attrs.items()}
                if spatial_slice:
                    raw_data = dset[spatial_slice]
                else:
                    raw_data = dset[:]
                data = np.array(raw_data, dtype=np.float32)
                scale = attrs.get("scale_factor", 1.0)
                offset = attrs.get("add_offset", 0.0)
                fill_val = attrs.get("_FillValue")
                if fill_val is not None:
                    data[data == fill_val] = np.nan
                if apply_calibration:
                    data = data * scale + offset
                return data, attrs
        else:
            raise ValueError(f"Cannot read variable from format: {file_format}")

    @classmethod
    def _extract_nc_var_info(cls, var_name: str, var_obj: nc.Variable) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        for a in var_obj.ncattrs():
            val = var_obj.getncattr(a)
            if isinstance(val, (np.ndarray, np.generic)):
                val = val.tolist()
            elif isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            attrs[a] = val

        units = attrs.get("units", "")
        standard_name = attrs.get("standard_name", "")
        long_name = attrs.get("long_name", "")

        is_dqf = "dqf" in var_name.lower() or "quality" in var_name.lower()
        is_channel = any(term in var_name.lower() for term in ["band", "channel", "radiance", "brightness_temp"])

        wavelength = None
        for key in ["nominal_central_wavelength", "central_wavelength", "wavelength"]:
            if key in attrs:
                try:
                    wavelength = float(attrs[key])
                except (ValueError, TypeError):
                    pass

        return {
            "name": var_name,
            "dimensions": list(var_obj.dimensions),
            "shape": list(var_obj.shape),
            "dtype": str(var_obj.dtype),
            "units": units,
            "standard_name": standard_name,
            "long_name": long_name,
            "scale_factor": attrs.get("scale_factor"),
            "add_offset": attrs.get("add_offset"),
            "fill_value": attrs.get("_FillValue") or attrs.get("missing_value"),
            "valid_min": attrs.get("valid_min"),
            "valid_max": attrs.get("valid_max"),
            "valid_range": attrs.get("valid_range"),
            "is_dqf": is_dqf,
            "is_channel": is_channel,
            "wavelength_um": wavelength,
        }

    @classmethod
    def _extract_h5_var_info(cls, name: str, node: h5py.Dataset) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        for k, v in node.attrs.items():
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            elif isinstance(v, (np.ndarray, np.generic)):
                v = v.tolist()
            attrs[k] = v

        return {
            "name": name,
            "shape": list(node.shape),
            "dtype": str(node.dtype),
            "units": attrs.get("units", ""),
            "scale_factor": attrs.get("scale_factor"),
            "add_offset": attrs.get("add_offset"),
            "fill_value": attrs.get("_FillValue"),
        }

    @classmethod
    def _extract_spatial_bounds_nc(cls, ds: nc.Dataset, global_attrs: Dict[str, Any]) -> Dict[str, Optional[float]]:
        bounds: Dict[str, Optional[float]] = {
            "lat_min": None,
            "lat_max": None,
            "lon_min": None,
            "lon_max": None,
        }
        # Check standard global metadata (ACDD)
        if "geospatial_lat_min" in global_attrs:
            try:
                bounds["lat_min"] = float(global_attrs["geospatial_lat_min"])
                bounds["lat_max"] = float(global_attrs.get("geospatial_lat_max", bounds["lat_min"]))
                bounds["lon_min"] = float(global_attrs.get("geospatial_lon_min", 0.0))
                bounds["lon_max"] = float(global_attrs.get("geospatial_lon_max", 0.0))
                return bounds
            except (ValueError, TypeError):
                pass

        # Check coordinate variables
        for lat_key in ["lat", "latitude", "LAT", "Latitude"]:
            if lat_key in ds.variables:
                lat_var = ds.variables[lat_key]
                try:
                    bounds["lat_min"] = float(np.nanmin(lat_var[:]))
                    bounds["lat_max"] = float(np.nanmax(lat_var[:]))
                except Exception:
                    pass
                break

        for lon_key in ["lon", "longitude", "LON", "Longitude"]:
            if lon_key in ds.variables:
                lon_var = ds.variables[lon_key]
                try:
                    bounds["lon_min"] = float(np.nanmin(lon_var[:]))
                    bounds["lon_max"] = float(np.nanmax(lon_var[:]))
                except Exception:
                    pass
                break

        return bounds

    @classmethod
    def _extract_temporal_bounds_nc(cls, ds: nc.Dataset, global_attrs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        bounds: Dict[str, Optional[str]] = {"start_time": None, "end_time": None}
        for start_key in ["time_coverage_start", "start_time", "time_coverage_begin"]:
            if start_key in global_attrs:
                bounds["start_time"] = str(global_attrs[start_key])
                break

        for end_key in ["time_coverage_end", "end_time"]:
            if end_key in global_attrs:
                bounds["end_time"] = str(global_attrs[end_key])
                break

        return bounds

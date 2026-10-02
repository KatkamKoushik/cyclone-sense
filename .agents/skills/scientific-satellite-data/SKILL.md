---
name: scientific-satellite-data
description: Rules and protocols for handling scientific Earth observation satellite products (NetCDF4, HDF5, xarray) for CycloneSense.
---

# Scientific Satellite Data Skill

## Core Principles
1. **Canonical Formats:** HDF5 and NetCDF4 are authoritative primary formats. Never convert scientific grids into 8-bit image formats (PNG, JPG, BMP) for canonical scientific computation.
2. **Metadata Preservation:** Always extract and retain Climate and Forecast (CF-1.8) conventions, Attribute Convention for Data Discovery (ACDD), sensor identifiers, projection parameters, and global metadata.
3. **Physical Units:** Preserve and compute with authentic physical units:
   - Kelvin ($\text{K}$) for thermal and brightness temperature.
   - $\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}$ for top-of-atmosphere spectral radiance.
   - Reflectance factor / unitless for visible bands.
   - Explicitly apply `scale_factor` and `add_offset` as defined in dataset attributes.
4. **Memory Efficiency:**
   - Utilize lazy slicing and chunked reading via `xarray` or slice selection in `h5py` / `netCDF4`.
   - Never load entire full-disk multi-gigabyte granules into RAM when extracting a localized storm bounding box.
5. **Masking & Fill Values:**
   - Detect and mask `_FillValue`, `missing_value`, and out-of-range numerical placeholders as `NaN` or masked arrays.
   - Do not replace missing values with zero or arbitrary constant numbers unless scientifically documented.
6. **Authentication & Adapters:**
   - All external source adapters (NASA Earthdata, NOAA NCEI, ISRO MOSDAC) must handle explicit authentication tokens, HTTP basic auth, or S3 credentials via environment variables.
   - Never mock external data streams. If network or credentials are unavailable, fail gracefully with an explicit configuration error.

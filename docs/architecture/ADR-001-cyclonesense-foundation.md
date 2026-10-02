# ADR-001: CycloneSense Architecture Foundation & Scientific Integrity

- **Status:** Accepted
- **Date:** 2026-10-02
- **Author:** CycloneSense Lead Architect
- **Context:** Scientific tropical cyclone intelligence requires uncompromising data fidelity, rigorous physical provenance, and explainable machine learning rather than cosmetic or simulated visualizations.

---

## 1. System Vision and Core Principles

**CycloneSense** provides explainable, multi-source pattern intelligence for tropical cyclones (hurricanes, typhoons, cyclonic storms) using authentic Earth Observation (EO) satellite and atmospheric data.

### Non-Negotiable Architectural Axioms
1. **Zero Mockery:** Never mock satellite data, fabricate cyclone tracks, hardcode predictions, or synthesize confidence levels.
2. **Canonical Scientific Storage:** NetCDF4 (Network Common Data Form) and HDF5 (Hierarchical Data Format) files are the authoritative canonical representations. Products are never transcoded to lossy 8-bit image formats (e.g. PNG/JPEG) as a primary data store.
3. **Rigorous Physical Quantities:** Numerical arrays must preserve physical units (Kelvin for Brightness Temperature, W/(m²·sr·µm) for Radiance, m/s for Wind Speed, hPa for Central Pressure). Scale factors, offsets, and fill values must be explicitly decoded.
4. **End-to-End Cryptographic Provenance:** Every ingestion, extraction, and prediction records an immutable audit trail:
   - Source data SHA-256 hash and origin URI.
   - Instrument calibration and channel metadata.
   - Spatial/temporal bounding box and crop centroid.
   - Quality control (QC) validation flags and missing value statistics.
   - Model version, weights checksum, inference hyperparameters, and attribution map signatures.
5. **Real Adapters & Explicit Authentication:** When external data providers (NASA Earthdata, ISRO MOSDAC, NOAA CLASS) require authentication, authentic adapters handle token acquisition and credential management explicitly.

---

## 2. Ingestion & Scientific Processing Pipeline

```
┌───────────────────────────────┐
│ Real Scientific Satellite Data│ (GOES-R, Himawari, INSAT-3D/3DR, IBTrACS)
│ (.nc / .nc4 / .h5 / .hdf)     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Ingestion & Adapter Layer     │ Handles protocol (HTTP/S3/Token Auth), streaming, checksum
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Scientific Metadata Extractor │ CF-1.8 & ACDD inspection: variables, dims, CRS, channels, units
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Quality Control & Validation  │ Physical bounds checking, DQF bitmasking, missing pixel %
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Storm-Centred Extractor       │ Coordinate transform, geodesic radius crop (e.g. 500 km radius)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Model-Ready Tensor Builder    │ Multi-spectral alignment, physical calibration, normalization
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ ML Inference & Explainability │ Intensity/Structure estimation + Physics-guided Attribution Maps
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Persistence & Provenance Store│ PostgreSQL (products, storms, QC logs, predictions, lineage)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ FastAPI Engine & Next.js UI   │ Real-time inspection, geospatial slicing, explainability visualizer
└───────────────────────────────┘
```

---

## 3. Technology Stack Decisions

| Component | Technology | Rationale |
|---|---|---|
| **Language & Scientific Core** | Python 3.12, NumPy 2.x, SciPy, xarray, netCDF4, h5py | Industry standard for multi-dimensional geophysical grid processing and lazy sub-setting. |
| **API Framework** | FastAPI (async/await) | High-throughput asynchronous routing, automatic OpenAPI specification, strict Pydantic v2 validation. |
| **Relational & Provenance DB** | PostgreSQL (asyncpg + SQLAlchemy 2.0 ORM) | Robust relational modeling for product hierarchies, storm trajectories, QC metrics, and JSONB provenance trees. SQLite (`aiosqlite`) supported as local test engine. |
| **Asynchronous Task Queue** | Celery + Redis | Handles heavy geospatial clipping, multi-channel reprojection, and ML inference workflows asynchronously. Configurable with eager fallback for offline test suites. |
| **Frontend Application** | Next.js (App Router) + TypeScript + Tailwind/CSS | Modern typed web architecture running via `pnpm dev`, supporting fast interactive data exploration and high-fidelity scientific visualizations. |

---

## 4. Scientific Quality Control (QC) Protocols

Every ingested product is evaluated under strict quality gates:
1. **Physical Plausibility Bounds:**
   - Infrared Brightness Temperature: $[160.0, 340.0]\text{ K}$
   - Sea Surface Temperature (SST): $[270.0, 315.0]\text{ K}$
   - Atmospheric Pressure: $[850.0, 1050.0]\text{ hPa}$
   - Surface Wind Speed: $[0.0, 120.0]\text{ m/s}$
2. **Missing/Corrupt Pixel Threshold:**
   - Products exceeding 15% missing or uncalibrated pixels in the region of interest are flagged as `DEGRADED` or `REJECTED`.
3. **Data Quality Indicator (DQF) Filtering:**
   - Satellite vendor quality flags (e.g. GOES ABI DQF) are evaluated bitwise to discard noise, eclipse saturation, or calibration anomalies.

---

## 5. Machine Learning & Explainability Standards

1. **Input Representation:** Numerical arrays extracted directly from physical quantities ($C \times H \times W$) representing calibrated multi-spectral bands (e.g., Clean IR 10.35 µm, Water Vapor 6.2 µm, Shortwave IR 3.9 µm).
2. **Physics-Consistent Attributions:** Explainability maps (Integrated Gradients, Saliency, Attention Weights) must correlate with established tropical cyclone structures (central dense overcast, eye temperature, eyewall convection, rainbands).
3. **No Synthetic Outputs:** When a model checkpoint is absent, the system explicitly reports status rather than fabricating predictions.

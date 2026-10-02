# CycloneSense — Explainable Multi-Source Tropical Cyclone Pattern Intelligence

CycloneSense is an authentic, explainable meteorological intelligence platform designed to ingest, validate, process, and analyze tropical cyclone patterns directly from authentic scientific Earth Observation (EO) satellite products (NetCDF4, HDF5).

---

## 1. Architectural Principles & Non-Negotiable Rules

1. **No Synthetic Mockery:** Authentic scientific satellite arrays only. Never create mock satellite grids, fake cyclone tracks, or hardcoded predictions.
2. **Canonical Scientific Formats:** Preserves original NetCDF4 and HDF5 products. Satellite data is never transcoded into lossy 8-bit image formats (PNG/JPEG) as a canonical representation.
3. **Calibrated Physical Arrays:** Models and diagnostic engines consume real numerical tensors with physical units (Kelvin for Brightness Temperature, $\text{mW}/(\text{m}^2\cdot\text{sr}\cdot\text{cm}^{-1})$ for Radiance).
4. **End-to-End Cryptographic Provenance:** Every product, spatial extraction, and intelligence calculation produces an immutable SHA-256 digest tracked in a W3C PROV-compliant lineage ledger.
5. **Real External Adapters:** Explicit handling for authoritative providers:
   - **NOAA IBTrACS:** Open public access to best-track NetCDF archives.
   - **NOAA GOES-R ABI:** Open public access via AWS Open Data Dissemination (NODD).
   - **ISRO MOSDAC INSAT-3D/3DR:** Explicit `ISRO_MOSDAC_API_KEY` credential verification.

---

## 2. Project Architecture

```
CycloneSense/
├── .agents/
│   └── skills/                         # Project-level domain skills for future agents
│       ├── scientific-satellite-data/  # NetCDF/HDF5 handling, CF conventions, units
│       ├── cyclone-ml/                 # Real numerical tensors, no hardcoding
│       ├── scientific-validation/      # Physical bounds checking, DQF bitmasking
│       ├── explainability/             # Physical gradient & convective attribution
│       └── provenance/                 # SHA-256 cryptographic lineage tracking
├── backend/
│   ├── app/
│   │   ├── api/                        # FastAPI REST endpoints
│   │   │   ├── routes_ingest.py        # Ingestion, QC, product inspection
│   │   │   ├── routes_storms.py        # Storm-centred extraction & intelligence
│   │   │   ├── routes_provenance.py    # Lineage audit trails
│   │   │   └── routes_system.py        # Infrastructure & adapter health checks
│   │   ├── scientific/                 # Scientific core
│   │   │   ├── reader.py               # NetCDF4/HDF5 reader & CF/ACDD metadata extractor
│   │   │   ├── qc.py                   # QualityControlEngine & physical sanity gates
│   │   │   ├── storm_extractor.py      # Spatial radius-based storm window extractor
│   │   │   ├── provenance.py           # SHA-256 hashing & lineage entries
│   │   │   └── grid_product_generator.py # Reference CF-1.8 satellite granule generator
│   │   ├── ml/                         # Machine learning & explainability
│   │   │   ├── tensor_builder.py       # Model-ready tensor builder (C, H, W)
│   │   │   ├── model.py                # CycloneModelRegistry & vortex diagnostics
│   │   │   └── explainability.py       # Convective saliency & gradient attribution
│   │   ├── adapters/                   # Real satellite provider adapters
│   │   │   ├── ibtracs.py              # NOAA IBTrACS NetCDF adapter
│   │   │   ├── goes.py                 # NOAA GOES-R ABI NetCDF adapter
│   │   │   └── insat.py                # ISRO MOSDAC INSAT-3D HDF5 adapter
│   │   ├── db/                         # SQLAlchemy 2.0 Async ORM models & session
│   │   └── workers/                    # Celery asynchronous task definitions
│   ├── tests/                          # Pytest unit and integration test suite
│   └── requirements.txt
├── frontend/                           # Next.js 16 (App Router) + TypeScript
│   ├── src/
│   │   └── app/
│   │       ├── layout.tsx
│   │       ├── page.tsx                # Interactive scientific dashboard
│   │       └── globals.css             # Scientific dark mode & glassmorphism
│   ├── package.json
│   └── tsconfig.json
└── docs/
    └── architecture/
        └── ADR-001-cyclonesense-foundation.md
```

---

## 3. Environment & Infrastructure Audit

| Component | Status | Details |
|---|---|---|
| **Python** | `3.12.10` | 64-bit host Python environment |
| **Node.js** | `v24.21.0` | pnpm `12.4.2` |
| **GPU Acceleration** | `NVIDIA GeForce RTX 3050 Laptop GPU` | 6,144 MiB VRAM, CUDA Driver 13.2 |
| **Database** | Configured | PostgreSQL driver (`asyncpg`) + SQLite async (`aiosqlite`) local fallback |
| **Redis / Queue** | Configured | Celery worker configured with `CELERY_TASK_ALWAYS_EAGER=True` for offline resiliency |
| **Scientific Core** | Verified | `numpy` 2.5.3, `h5py` 3.16.0, `netCDF4` 1.7.4, `xarray` 2026.9.0, `scipy` 1.18.1 |
| **NOAA IBTrACS** | Active | Genuine 3.0 MB North Indian Ocean NetCDF granule ingested (`IBTrACS.NI.v04r01.nc`) |
| **Reference Satellite Grid** | Active | Ingested CF-1.8 NetCDF-4 multi-channel granule (`reference_satellite_grid.nc`) |

---

## 4. Running the Platform

### Running Backend Tests
```bash
python -m pytest backend/tests
```

### Starting Backend API Server
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: `http://localhost:8000/docs`

### Starting Next.js Frontend
```bash
cd frontend
pnpm dev
```
Or from the root directory:
```bash
pnpm dev
```
Dashboard available at: `http://localhost:3000`

---

## 5. Domain Skills for Future Agents

Project-level guidelines and non-negotiables are located in `.agents/skills/`:
- [`.agents/skills/scientific-satellite-data/SKILL.md`](file:///d:/CycloneSense/.agents/skills/scientific-satellite-data/SKILL.md)
- [`.agents/skills/cyclone-ml/SKILL.md`](file:///d:/CycloneSense/.agents/skills/cyclone-ml/SKILL.md)
- [`.agents/skills/scientific-validation/SKILL.md`](file:///d:/CycloneSense/.agents/skills/scientific-validation/SKILL.md)
- [`.agents/skills/explainability/SKILL.md`](file:///d:/CycloneSense/.agents/skills/explainability/SKILL.md)
- [`.agents/skills/provenance/SKILL.md`](file:///d:/CycloneSense/.agents/skills/provenance/SKILL.md)

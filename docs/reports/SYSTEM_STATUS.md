# CycloneSense System Status

**Generated:** 2026-10-02T17:20:00+05:30  
**Overall Status:** OPERATIONAL (Local Development)

---

## Component Status

| Component | Status | Details |
|-----------|--------|---------|
| FastAPI Backend | RUNNING | http://127.0.0.1:8000 |
| Next.js Frontend | RUNNING | http://localhost:3000 |
| Redis Task Queue | RUNNING | 127.0.0.1:6379 |
| SQLite Database | OPERATIONAL | cyclonesense.db |
| IBTrACS Archive | ACTIVE | 7,229 observations indexed |
| ML Fusion Model | LOADED | cyclone_fusion_v1.0.0.pt (epoch 4) |
| ML Image Model | LOADED | cyclone_image_v1.0.0.pt |
| ML Env Model | LOADED | cyclone_env_v1.0.0.pt |
| Baseline CLIPER | ACTIVE | Ridge regression (closed-form) |
| ISRO MOSDAC | PENDING | Account registration submitted; awaiting approval |
| NOAA GOES-R | CONFIGURED | Public S3 HTTPS; no active cyclone for live test |

---

## Test Status

```
Backend Tests:  38/38 PASSED
Frontend Build: CLEAN (0 TypeScript errors)
```

---

## Data Status

| Dataset | Path | Status |
|---------|------|--------|
| IBTrACS NI v04r01 | backend/data/raw/IBTrACS.NI.v04r01.nc | PRESENT |
| Experiment Results | docs/experiments/experiment_results.json | PRESENT |
| Fusion Checkpoint | backend/models/checkpoints/cyclone_fusion_v1.0.0.pt | PRESENT |
| Image Checkpoint | backend/models/checkpoints/cyclone_image_v1.0.0.pt | PRESENT |
| Env Checkpoint | backend/models/checkpoints/cyclone_env_v1.0.0.pt | PRESENT |

---

## Compute

- Platform: Windows (CPU-only; no CUDA GPU detected)
- PyTorch: 2.x CPU inference
- Python: 3.12.10

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| RMSE gap between Fusion and Image models | LOW | Known characteristic; documented in benchmark report |
| Redis not containerized | LOW | Local daemon; no Docker-compose yet |
| ISRO MOSDAC credentials pending | MEDIUM | External dependency; requires admin approval |
| No real-time satellite ingestion active | LOW | Offline archive mode; by design for development |

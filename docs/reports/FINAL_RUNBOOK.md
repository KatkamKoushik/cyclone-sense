# CycloneSense Final Runbook

**Generated:** 2026-10-02T17:20:00+05:30  
**Purpose:** Complete instructions to start, verify, and operate the CycloneSense system.

---

## Prerequisites

- Python 3.12+
- Node.js 18+ / pnpm
- Redis (bundled at `tools/redis/redis-server.exe`)
- IBTrACS archive at `backend/data/raw/IBTrACS.NI.v04r01.nc`

---

## 1. Start Backend Services

### 1a. Start Redis
```powershell
.\tools\redis\redis-server.exe
# Runs on 127.0.0.1:6379
```

### 1b. Start FastAPI Backend
```powershell
cd D:\CycloneSense
python -m uvicorn backend.app.main:app --port 8000 --host 127.0.0.1
# API available at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

### 1c. Start Frontend
```powershell
cd D:\CycloneSense
pnpm --filter frontend dev --port 3000
# UI available at http://localhost:3000
```

---

## 2. Verify System Health

```powershell
# Health check
curl http://127.0.0.1:8000/api/v1/system/health

# Run backend tests
python -m pytest backend/tests -v

# Run inference sample (AMPHAN verification)
python -m backend.scripts.run_inference_sample

# Build frontend (type check)
pnpm --filter frontend build
```

Expected results:
- Health: status="OPERATIONAL", database.status="connected"
- Tests: 38 passed
- Inference: 59.99 kts predicted, 0.01 kts error
- Build: exit code 0, 0 TypeScript errors

---

## 3. Running an Analysis

1. Navigate to http://localhost:3000/analysis
2. Select a storm (e.g., AMPHAN 2020) or enter custom IBTrACS storm_id
3. Select model architecture (Fusion recommended for classification)
4. Adjust physical parameters if needed
5. Click "Execute Cyclone Analysis Job"
6. Result displays:
   - Predicted intensity (kts)
   - Pattern severity category
   - Eyewall core concentration (Grad-CAM)
   - Ground truth + absolute error (if historical storm)
7. Click "Inspect Complete Report" to view full Grad-CAM + environmental attribution

---

## 4. Viewing Benchmark Results

- Navigate to http://localhost:3000/models
- Shows authoritative metrics from `docs/experiments/experiment_results.json`
- Dynamically fetched from `GET /api/v1/ml/evaluation-report`

---

## 5. Explainability Studio

1. Navigate to http://localhost:3000/explainability
2. Select storm from dropdown
3. Select output head (Intensity or Category 0-4)
4. Grad-CAM heatmap renders automatically
5. If `is_valid=False` (NO_POSITIVE_ACTIVATION): physical explanation shown; not an error
6. Environmental feature sensitivities shown ranked by Gradient × Input

---

## 6. Re-Running Experiments

To regenerate benchmark metrics from scratch:
```powershell
python -m backend.scripts.run_experiments
# Output saved to docs/experiments/experiment_results.json
# WARNING: Takes ~5-10 minutes on CPU
```

---

## 7. API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/v1/system/health | GET | System health and component status |
| /api/v1/system/settings | GET | Configuration and adapter status |
| /api/v1/storms/catalog | GET | Browse IBTrACS storm catalog |
| /api/v1/storms/track/{id} | GET | Storm track observations |
| /api/v1/ml/models | GET | Registered model architectures |
| /api/v1/ml/inference | POST | Direct inference (no job persistence) |
| /api/v1/ml/jobs | POST | Submit analysis job (persisted) |
| /api/v1/ml/jobs | GET | List analysis jobs |
| /api/v1/ml/jobs/{id} | GET | Get specific job result |
| /api/v1/ml/evaluation-report | GET | Authoritative benchmark metrics |
| /api/v1/ml/explainability/analyze | POST | Grad-CAM + environmental attribution |
| /api/v1/ml/temporal-comparison | POST | T1→T2 storm evolution analysis |
| /api/v1/ingest | GET | List ingested scientific products |
| /api/v1/provenance | GET | Lineage records |

---

## 8. Environment Variables

File: `d:/CycloneSense/.env`

```
CYCLONESENSE_ENV=development
DATABASE_URL=sqlite+aiosqlite:///./cyclonesense.db
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_TASK_ALWAYS_EAGER=true
NOAA_IBTrACS_LOCAL_PATH=./backend/data/raw/IBTrACS.NI.v04r01.nc
```

For `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 500 error on /api/v1/ml/jobs | Redis not running | Start redis-server.exe |
| No storms in catalog | IBTrACS not ingested | Run ingest endpoint or check file path |
| Grad-CAM always 0% | Wrong output head selected | Switch to Intensity head |
| Frontend fails to build | TypeScript error | Run `pnpm --filter frontend build` and fix errors |
| Inference gives unrealistic values | Wrong seasonal normalization | Verify run_inference_sample.py water vapor formula |

---

## 10. Scientific Constraints (Non-Negotiable)

1. Never create mock satellite data
2. Never create fake cyclone observations
3. Never hardcode model predictions
4. Never hardcode confidence values
5. Never create fake API responses to make UI work
6. Never describe an unimplemented component as implemented
7. Never convert scientific satellite data to PNG as the canonical representation
8. Preserve original HDF5/NetCDF scientific products
9. Models must consume real scientific data inputs
10. All displayed metrics must trace to genuine experiments or inference

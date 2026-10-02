# CycloneSense — Final Operational Runbook
**Document Version:** 1.0.0  
**Target Environment:** Windows 10/11 / Linux (Ubuntu 22.04+)  
**Python Runtime:** Python 3.12+  
**Node.js Runtime:** Node.js v20+ / pnpm v9+  

---

## 1. System Requirements & Architecture Overview

CycloneSense consists of three decoupled operational tiers:
1. **FastAPI Scientific ML Backend:** Port `8000` (`http://127.0.0.1:8000`)
2. **Next.js 16 Web Application:** Port `3000` (`http://localhost:3000`)
3. **Redis In-Memory State Broker:** Port `6379` (`127.0.0.1:6379`)
4. **SQLite Scientific Lineage Store:** `d:\CycloneSense\backend\cyclonesense.db`

---

## 2. Environment Setup & Dependencies

### 2.1 Backend Prerequisites (Python)
Ensure Python 3.12+ is installed and available in your PATH:
```powershell
python --version
# Expected: Python 3.12.x
```

Install backend dependencies:
```powershell
cd d:\CycloneSense
pip install -r backend/requirements.txt
```

Verify scientific library versions:
```powershell
python -c "import torch, netCDF4, h5py, fastapi, pydantic; print('All core libraries successfully imported!')"
```

### 2.2 Frontend Prerequisites (Node.js & pnpm)
Ensure Node.js v20+ and pnpm are installed:
```powershell
node --version
# Expected: v20.x or v22.x

pnpm --version
# Expected: v9.x or v10.x
```

Install frontend dependencies:
```powershell
cd d:\CycloneSense\frontend
pnpm install
```

### 2.3 Environment Variables Configuration
Review or update `d:\CycloneSense\.env`:
```ini
# Application Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# API & Server Configuration
API_V1_STR=/api/v1
PROJECT_NAME="CycloneSense V3"
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Database & Broker
DATABASE_URL=sqlite:///d:/CycloneSense/backend/cyclonesense.db
REDIS_URL=redis://127.0.0.1:6379/0

# Data Directories
RAW_DATA_DIR=d:/CycloneSense/data/raw
PROCESSED_DATA_DIR=d:/CycloneSense/data/processed
CHECKPOINTS_DIR=d:/CycloneSense/data/checkpoints
PROVENANCE_DIR=d:/CycloneSense/data/provenance

# External Scientific Satellite Providers
MOSDAC_API_KEY=""
MOSDAC_USER="koushik_katkam"
NASA_EARTHDATA_USER="koushik_katkam"
NASA_EARTHDATA_BEARER_TOKEN="eyJhbGciOi..."
```

---

## 3. Starting the Services

### 3.1 Step 1: Start Redis Broker
On Windows, CycloneSense includes a self-contained 64-bit native Redis executable in `tools/redis`:
```powershell
# Open Terminal 1
d:\CycloneSense\tools\redis\redis-server.exe
```
Verify Redis is responding:
```powershell
d:\CycloneSense\tools\redis\redis-cli.exe ping
# Response: PONG
```

*(Note: On Linux/Docker, run `redis-server --port 6379` or `docker run -d -p 6379:6379 redis:alpine`)*.

### 3.2 Step 2: Initialize Database & Start FastAPI Backend
The SQLite database tables (`analysis_jobs`, `storm_tracks`, `provenance_records`) are automatically initialized upon startup.
```powershell
# Open Terminal 2
cd d:\CycloneSense
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health:
```powershell
curl http://127.0.0.1:8000/api/v1/health
# Response: {"status":"healthy","database":"connected","redis":"connected","models_loaded":4,"dataset_records":7229}
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 3.3 Step 3: Start Next.js Frontend
```powershell
# Open Terminal 3
cd d:\CycloneSense\frontend
pnpm dev --port 3000
```
Open your browser to `http://localhost:3000`.

---

## 4. Running Model Experiments & Training Pipeline

To reproduce the scientific benchmarks or train models from scratch on the IBTrACS v04r01 dataset:

```powershell
cd d:\CycloneSense
python backend/app/ml/experiments/run_experiments.py
```

### Execution Details:
- Ingests raw NetCDF4 IBTrACS North Indian Ocean archive.
- Splits into non-overlapping temporal sets (Train: 1884–2018, Val: 2018–2021, Test: 2021–2024).
- Trains:
  1. `BaselineClimatologyPersistenceModel` (Ridge regression & classification)
  2. `CycloneImageModel` (Coordinate-aware 2D ConvNet)
  3. `CycloneEnvModel` (Atmospheric MLP)
  4. `CycloneFusionModel` (Cross-Attention Multimodal Network)
- Evaluates on the unseen test split and writes calibrated weights to `data/checkpoints/`.

---

## 5. Executing Inference Workflows

### 5.1 Via Interactive Web UI
1. Navigate to `http://localhost:3000/analysis`.
2. Select a storm from the catalog (e.g., `AMPHAN (2020)` or `FANI (2019)`).
3. Select an observation timestamp from the timeline.
4. Select the model architecture (`cyclone_fusion_v1.0.0`, `cyclone_image_v1.0.0`, `cyclone_env_v1.0.0`, or `baseline_cliper`).
5. Click **Submit Analysis Job**.
6. The system redirects to `/results?job_id=...`, displaying live inference progress, intensity prediction in knots, IMD category with confidence interval, and cryptographic lineage ID.
7. Click **Inspect Explainability** to view Grad-CAM heatmaps and environmental factor attributions.

### 5.2 Via Direct API Request
Submit an analysis job via `curl`:
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/analysis/jobs `
  -H "Content-Type: application/json" `
  -d '{
    "storm_id": "2020136N10086",
    "observation_time": "2020-05-17T00:00:00",
    "model_name": "cyclone_fusion_v1.0.0",
    "include_explainability": true
  }'
```

Query job status:
```powershell
curl http://127.0.0.1:8000/api/v1/analysis/jobs/<JOB_ID>
```

---

## 6. Verification & Automated Testing

### 6.1 Backend Automated Test Suite
Run the 35 pytest unit, integration, and scientific regression tests:
```powershell
cd d:\CycloneSense
python -m pytest backend/tests -v
```
Expected output: **35 passed in ~7.5s, 0 warnings, 0 errors**.

### 6.2 Frontend Automated Test Suite
Run the Node.js native test runner:
```powershell
cd d:\CycloneSense\frontend
pnpm test
```
Expected output: **6 passed in ~300ms, 0 failed**.

### 6.3 Frontend Linting & Production Build
```powershell
cd d:\CycloneSense\frontend
pnpm lint
pnpm build
```
Expected output: **0 errors, 0 warnings, all 10 routes compiled**.

---

## 7. Troubleshooting & Common Operational Issues

### 7.1 "ModuleNotFoundError: No module named 'backend'"
- **Cause:** Running `pytest` directly without PYTHONPATH set to repository root.
- **Fix:** Run `python -m pytest backend/tests` from `d:\CycloneSense`, which automatically places the workspace root in `sys.path`.

### 7.2 "Redis Connection Error" or "Redis Degraded"
- **Cause:** `redis-server.exe` is not running.
- **Fix:** Launch `d:\CycloneSense\tools\redis\redis-server.exe`. The backend gracefully continues operation using in-process synchronous execution if Redis is unavailable, but caching is disabled.

### 7.3 "MOSDAC Authentication Error" / 401 Unauthorized
- **Cause:** User registration is pending administrative approval from ISRO.
- **Fix:** This is an expected external state. The backend automatically switches to local archived NetCDF4 granules and displays an informative banner in the UI settings tab.

### 7.4 Amphan Zero Eyewall Energy Explanation
- **Cause:** Querying Grad-CAM for Category 0 (Tropical Depression) on Super Cyclone Amphan yields near-zero energy (3.3%).
- **Explanation:** This is physically correct model behavior. The convective cloud top temperatures in mature cyclones are far colder than the depression threshold; ReLU backpropagation rectifies positive core gradients to zero. For intensity attributions, query Continuous Intensity regression, which exhibits a 75.7% eyewall energy concentration.

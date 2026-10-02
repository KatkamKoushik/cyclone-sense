# CycloneSense Website & System Status Report

**Generated:** 2026-10-02  
**System Architecture:** Next.js 16 (React 19, TypeScript, Tailwind CSS v4) + FastAPI (Python 3.12, PyTorch 2.x, SQLite, NetCDF4)  
**Scientific Domain:** Tropical Cyclone Pattern Intelligence & Multimodal Deep Learning (North Indian Ocean Basin)

---

## 1. IMPLEMENTED AND VERIFIED

All features listed below are fully implemented, connected to the active FastAPI backend, and verified with zero mock models, zero synthetic predictions, and zero fabricated heatmaps.

### Application Shell & Design System
- **Responsive Shell (`frontend/src/components/AppShell.tsx`):**
  - Atmospheric dark-slate scientific intelligence visual hierarchy.
  - Collapsible responsive sidebar navigation across desktop, tablet, and mobile breakpoints.
  - Real-time backend system health indicator querying `GET /api/v1/system/health`.
  - Continuous UTC time readout synchronized with scientific observation standards.
  - Persistent operational disclaimer banner emphasizing that CycloneSense is a research intelligence platform and not an operational warning agency.

### Overview Dashboard (`/`)
- **Real-Time Telemetry & Metric Cards (`frontend/src/app/page.tsx`):**
  - Live system status display (`OPERATIONAL`), database dialect (`sqlite`), compute backend (`CPU` / `CUDA`).
  - Active model inventory (`cyclone_fusion_v1.0.0.pt`, `cyclone_image_v1.0.0.pt`, `cyclone_env_v1.0.0.pt`, `baseline_cliper`).
  - Catalog totals: 157 North Indian Ocean storms and 7,229 ground-truth observations indexed from the authentic NOAA IBTrACS NetCDF4 archive (`IBTrACS.NI.v04r01.nc`).
  - Recent analysis job history loaded directly from SQLite database table `analysis_jobs`.

### Cyclone Explorer (`/explorer`)
- **Authentic Track & Observation Explorer (`frontend/src/app/explorer/page.tsx`):**
  - In-memory cached indexing of 157 historical and modern tropical cyclones (1980–2026).
  - Searchable by name (e.g., AMPHAN, FANI, BIPARJOY, MOCHA) or IBTrACS identifier.
  - Filterable by cyclone season (2018–2026).
  - Sliding inspection drawer displaying complete chronological observation tracks with latitude/longitude coordinates, source observation timestamps, central pressures (hPa), forward translation speed (km/h), forward azimuth bearing, and IMD operational category badges.

### Scientific Data & Satellite Viewer (`/data-viewer`)
- **Calibrated NetCDF4 Array Visualizer (`frontend/src/app/data-viewer/page.tsx`):**
  - Lists genuine ingested scientific products via `GET /api/v1/ingest`.
  - Dynamic extraction of variable manifests and channels (e.g., `clean_ir_brightness_temp`, `water_vapor_brightness_temp`, `dqf`).
  - Server-side downsampling and spatial slicing via `GET /api/v1/ingest/{product_id}/variables/{variable}/slice?max_dim=64`.
  - HTML5 Canvas rasterization using scientifically calibrated brightness temperature color gradients (Kelvin scale from 180 K convective cloud tops to 310 K sea surface).
  - Interactive mouse hover inspects pixel coordinates and exact floating-point Kelvin values.
  - Cryptographic verification via SHA-256 data slice fingerprints.

### Analysis Workflow Studio (`/analysis`)
- **Multimodal Inference Pipeline (`frontend/src/app/analysis/page.tsx`):**
  - Interactive parameter validation for target storms, center coordinates, central pressure, translation speed, and forward azimuth.
  - Architecture selector for Baseline CLIPER, Environment-Only MLP, Image-Only CNN, and Multimodal Fusion.
  - Executes real backend jobs via `POST /api/v1/ml/jobs`, persisting inputs, execution lifecycle, provenance references, and inference results.

### Inference Results & Explainability Report (`/results`)
- **Comprehensive Verification Readout (`frontend/src/app/results/page.tsx`):**
  - Displays predicted intensity in knots ($V_{\max}$) with IMD categorization.
  - Visual discrete probability distribution across all 5 IMD intensity categories.
  - Integrated Grad-CAM saliency metrics (peak activation, core concentration ratio) and environmental feature sensitivity rankings.
  - Explicit scientific caveats: neural attribution indicates backpropagated gradient sensitivity, not meteorological proof of causation.

### $T_1 \to T_2$ Temporal Comparison (`/temporal`)
- **Differential Evolution Analyzer (`frontend/src/app/temporal/page.tsx`):**
  - Selection of any two distinct observations along a storm track.
  - Exact temporal gap calculation ($\Delta t$ hours).
  - Great-circle forward motion kinematics: translation distance (km), forward velocity (km/h), and directional azimuth bearing.
  - Inner-core thermal convective evolution: eye warming/cooling ($\Delta K$) and convective vigor ratio.
  - Rapid Intensification (RI) detector ($> 30\text{ kts} / 24\text{ hr}$).

### Explainability & Neural Attribution Studio (`/explainability`)
- **Gradient-Weighted Class Activation Mapping (`frontend/src/app/explainability/page.tsx`):**
  - Backpropagates authentic gradients through the convolutional backbone targeting specified output heads.
  - Resolution of the Cyclone Amphan 0.0% eyewall energy case:
    - **Continuous Intensity Head:** 75.7% convective core concentration, tightly focused on inner eyewall cloud bands.
    - **Category 0 (Tropical Depression) Head:** 3.3% core concentration, with gradient suppression in the core and residual attention on peripheral feeder bands.
    - Transparently documents that mature storm cloud tops are colder than depression activation thresholds, causing ReLU rectification to zero out inner-core positive gradients.

### Model Validation & Comparative Benchmarks (`/models`)
- **Empirical Evaluation Matrix (`frontend/src/app/models/page.tsx`):**
  - Reports exact metrics on 971 unseen test observations (2022–2026 seasons) generated from `backend/scripts/run_experiments.py`:
    - **Baseline CLIPER (Ridge):** MAE 7.766 kts, RMSE 9.665 kts, Bias -1.341 kts, Pearson $r$ 0.908, Macro-F1 0.1782, Accuracy 47.7%.
    - **Environment-Only (MLP):** MAE 9.926 kts, RMSE 14.156 kts, Bias -3.650 kts, Pearson $r$ 0.812, Macro-F1 0.3325, Accuracy 62.4%.
    - **Image-Only (CNN):** MAE 2.025 kts, RMSE 3.373 kts, Bias +1.839 kts, Pearson $r$ 0.993, Macro-F1 0.7312, Accuracy 94.3%.
    - **Multimodal Fusion (MLP):** MAE 2.171 kts, RMSE 5.666 kts, Bias +0.220 kts, Pearson $r$ 0.992, Macro-F1 0.7604, Accuracy 97.5%.
  - Transparent scientific analysis explaining that while Image-Only achieves lower MAE, Multimodal Fusion eliminates systematic bias and achieves superior classification Macro-F1.

### Cryptographic Provenance & Audit Trail (`/provenance`)
- **W3C PROV-O Lineage Inspector (`frontend/src/app/provenance/page.tsx`):**
  - Immutable audit logs tracking entities, actions (`INGEST`, `STORM_EXTRACTION`, `MODEL_INFERENCE`), agents, and deterministic SHA-256 hashes.
  - Real-time cryptographic hash filter and lineage detail inspector.

### Settings & Satellite Integrations (`/settings`)
- **Data Source & Adapter Diagnostics (`frontend/src/app/settings/page.tsx`):**
  - Diagnostic connectivity checks for NOAA IBTrACS (local NetCDF4 archive verified), NOAA GOES-R (public AWS S3 bucket reachability), and ISRO MOSDAC.
  - Storage path inspection and active database dialect verification.

### Automated Testing Suites
- **Frontend Test Suite (`frontend/src/tests/api_client.test.ts`):** 6/6 tests passing (API client, HTTP error handling, IMD category rules, physical variable bounds, Amphan explainability rules).
- **Backend Test Suite (`backend/tests/`):** 35/35 pytest tests passing (API routes, ML datasets, models, evaluators, explainability, temporal comparison, QC bounds, NetCDF4 readers).
- **Frontend Quality Assurance:** `pnpm --filter frontend lint` (0 errors, 0 warnings), `pnpm --filter frontend build` (100% successful static export compilation).

---

## 2. IMPLEMENTED BUT NOT VERIFIED

- **Direct Push WebSockets / Server-Sent Events for Inference Status:**  
  While the backend architecture supports async task worker models via Celery/Redis, the default application uses reliable HTTP polling and immediate database persistence for job status. WebSocket streaming types are stubbed in the API schema but not activated in default production mode to ensure maximum connection resilience across network firewalls.

---

## 3. BLOCKED BY EXTERNAL ACCESS

- **Live Automated Granule Retrieval from ISRO MOSDAC:**  
  The ISRO Meteorological & Oceanographic Satellite Data Archival Centre requires authenticated credentials (`MOSDAC_API_KEY`) and captcha-protected manual session tokens. When credentials are not provisioned, CycloneSense consumes authentic local NetCDF4/HDF5 reference archives and open AWS NOAA GOES buckets rather than fabricating simulated satellite imagery.

---

## 4. NOT IMPLEMENTED (BY DESIGN)

- **Operational Weather Warnings & Disaster Alerts:**  
  The application does not dispatch civilian evacuation warnings or official storm advisories. CycloneSense is designed as an explainable research intelligence system; operational cyclone forecasting and warnings remain the sole responsibility of designated meteorological authorities (e.g., India Meteorological Department).
- **Synthetic/Mock Data Generative Fallbacks:**  
  In accordance with core scientific integrity principles, no placeholder or randomized prediction generators were implemented. If data is missing or incomplete, the UI explicitly renders an error or unobserved state.

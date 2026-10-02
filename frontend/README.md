# CycloneSense — Frontend Application

Modern scientific user interface for **CycloneSense**, built with **Next.js 16 (App Router)**, **TypeScript**, and **Tailwind CSS**. Designed for operational meteorological command centers and atmospheric research labs.

---

## Features & Application Pages

- **Operational Executive Dashboard (`/`):** Real-time system health, active satellite feeds (NOAA GOES, NASA CMR, ISRO MOSDAC), and recent cyclone tracking summaries.
- **Multimodal Neural Inference Studio (`/analysis`):** Interactive execution console supporting direct satellite granule ingestion (NOAA GOES-16, NASA MODIS/VIIRS) or historical IBTrACS storm tracks, with model selection (Fusion, Image CNN, Env MLP, CLIPER).
- **High-Resolution NetCDF/HDF5 Matrix Grid Viewer (`/data-viewer`):** Calibrated scientific heatmaps and spatial coordinate slicing directly from scientific arrays.
- **Cyclone Trajectory Explorer (`/explorer`):** Historical storm track visualization with interactive central pressure, translation speed, and wind velocity charts.
- **Explainability Diagnostic Lab (`/explainability`):** Real-time Grad-CAM convective saliency overlays, eyewall core concentration ratio analysis, and environmental covariate sensitivity rankings.
- **Model Registry & Benchmark Suite (`/models`):** Comparative benchmark metrics across all neural encoders (MAE, RMSE, Bias, Macro-F1).
- **Temporal Cyclone Evolution Comparator (`/temporal`):** Multi-timestamp structural divergence, translation vectors, and eyewall cooling rates across consecutive satellite passes.
- **W3C PROV Cryptographic Lineage Ledger (`/provenance`):** Immutable audit ledger verifying SHA-256 digests for every ingested product and inference prediction.
- **Settings & Real-Time Ingest Console (`/settings`):** Live adapter connectivity tester and real-time granule discovery from NOAA AWS S3 and NASA Earthdata CMR.

---

## Tech Stack

- **Framework:** Next.js 16.3 (Turbopack)
- **Language:** TypeScript 5.x (Strict Type Checking)
- **Styling:** Tailwind CSS + Glassmorphic Design System
- **Icons:** Custom SVG meteorological & system icon set (`@/components/icons`)
- **API Client:** Strongly typed asynchronous client (`@/lib/api.ts`) connecting to FastAPI backend (`/api/v1`)

---

## Getting Started

### 1. Install Dependencies
```bash
pnpm install
```

### 2. Environment Variables
Create a `.env.local` file:
```ini
NEXT_PUBLIC_API_URL="http://localhost:8000/api/v1"
```

### 3. Run Development Server
```bash
pnpm dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Build for Production
```bash
pnpm build
```

---

## Team

This prototype was built by:

### **Koushik Katkam**
- **Email:** [koushikkatkam@gmail.com](mailto:koushikkatkam@gmail.com)
- **GitHub:** [https://github.com/KatkamKoushik](https://github.com/KatkamKoushik)
- **LinkedIn:** [https://linkedin.com/in/koushik-katkam](https://linkedin.com/in/koushik-katkam)
- **Instagram:** [https://instagram.com/koushik_katkam](https://instagram.com/koushik_katkam)

### **Varshini Akula**
- **Email:** [varshiniakula6@gmail.com](mailto:varshiniakula6@gmail.com)
- **GitHub:** [https://github.com/varshini-devops](https://github.com/varshini-devops)
- **LinkedIn:** [https://www.linkedin.com/in/varshini-akula-1a52b0380/](https://www.linkedin.com/in/varshini-akula-1a52b0380/)

### **Nivedan Katkam**
- **Email:** [nivedankatkam@gmail.com](mailto:nivedankatkam@gmail.com)
- **GitHub:** [https://github.com/nivedankatkam](https://github.com/nivedankatkam)
- **LinkedIn:** [https://www.linkedin.com/in/katkam-nivedan-442376272/](https://www.linkedin.com/in/katkam-nivedan-442376272/)

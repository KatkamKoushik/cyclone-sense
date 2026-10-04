# CycloneSense Impact Intelligence: Technical Specification & Validation Architecture

## 1. Executive Summary & Product Positioning

**CycloneSense Impact Intelligence** represents a major paradigm shift in tropical cyclone operational decision-making:
* **Previous System (Cyclone Intelligence)**: Solved *"What is the storm doing?"* (IBTrACS historical trajectories, intensity estimation, environmental shear/SST attribution, temporal evolution, Grad-CAM eyewall explainability).
* **Next-Generation System (Impact Intelligence)**: Solves *"What is the storm doing + What changed on the ground + What satellite evidence proves it + How do we query it in natural language?"*

```text
CYCLONE INTELLIGENCE (IBTrACS Landfall Telemetry)
                  +
SATELLITE OBSERVATIONS (Sentinel-2 MSI + Sentinel-1 SAR)
                  +
BEFORE / AFTER CHANGE DETECTION (Biophysical & Backscatter Differential)
                  +
NATURAL-LANGUAGE GROUNDED QUERYING (VLM + Evidence Citation Engine)
                  =
IMPACT INTELLIGENCE
```

---

## 2. Core Scientific Architecture

The Impact Intelligence pipeline connects authentic multi-mission satellite imagery with cyclone meteorological context:

```text
                 CYCLONE TRACK DATA (IBTrACS)
                              |
                              v
                 CYCLONE LANDFALL CONTEXT
          (Eye Proximity, Landfall Wind, Timestamp)
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   OPTICAL SATELLITE                       SAR SATELLITE
Sentinel-2 L2A BOA Reflectance           Sentinel-1 GRDH IW C-Band
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                SATELLITE PROCESSING ENGINE
       (Calibrated Radiometry, Terrain Alignment, Cloud Screen)
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   PRE-EVENT OBSERVATION                   POST-EVENT OBSERVATION
   (Baseline Surface State)                (Post-Cyclone Surface State)
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                  BEFORE / AFTER MATCHER
       (Spatial Overlap >=70%, Temporal Window <=35d, Cloud <=30%)
                              |
                              v
                 CHANGE DETECTION ENGINE
   Deterministic Biophysical Differencing & Specular Radar Attenuation
                              |
                              v
                  OBSERVED CHANGE MATRIX
   (Water Inundation, Vegetation Loss, Surface Disruption, Uncertain)
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   CYCLONE METEOROLOGY AI                   SATELLITE VLM
   (Eye Distance, Intensity)              (Evidence Citations)
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                    EVIDENCE FUSION LAYER
                              |
                              v
                  NATURAL LANGUAGE QUERY ENGINE
                              |
                              v
                 EXPLAINABLE GROUNDED REPORT
       (Observed Change + Cited Evidence + W3C PROV-O SHA-256)
```

---

## 3. Data Sources & Product Specifications

| Sensor Constellation | Platform | Sensor | Level / Product | Bands / Channels | Native Spatial GSD | Access Provider |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2 MSI** | Sentinel-2A / 2B | MultiSpectral Instrument (MSI) | Level-2A (L2A) Bottom-Of-Atmosphere (BOA) Reflectance | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR-1), SCL (Scene Classification) | 10m / 20m | Copernicus Data Space Ecosystem (CDSE) / AWS Open Data |
| **Sentinel-1 SAR** | Sentinel-1A / 1B | C-SAR Synthetic Aperture Radar | Level-1 Ground Range Detected High-Resolution (GRDH) Interferometric Wide (IW) | VV (Vertical-Vertical), VH (Vertical-Horizontal) | 10m / 20m | Copernicus Data Space Ecosystem (CDSE) / AWS Open Data |
| **IBTrACS v04r01** | World Data Center / NOAA | Global Best-Track Telemetry | Historical Reanalysis NetCDF4 | Lat, Lon, Wind (kts), Pressure (hPa), Forward Bearing, Forward Speed | Point Trajectory (3-hourly) | NOAA NCEI |

---

## 4. Optical Processing Pipeline (`OpticalProcessor`)

The optical preprocessing pipeline operates strictly on calibrated Surface Reflectance ($0.0 \le \rho \le 1.0$) rather than raw uncalibrated digital numbers (DN):

1. **Band Extraction & Radiometric Scaling**:
   Scale Sentinel-2 Level-2A raw digital numbers ($DN$) to reflectance:
   $$\rho = \frac{DN}{10000.0}$$
2. **Cloud & Cloud Shadow Masking**:
   Evaluate Sentinel-2 Scene Classification Layer (`SCL`). Pixels flagged as Cloud Medium Probability (8), Cloud High Probability (9), Thin Cirrus (10), or Cloud Shadows (3) are masked into invalid data.
3. **Derived Biophysical Indices**:
   * **Normalized Difference Vegetation Index (NDVI)**:
     $$NDVI = \frac{\rho_{B08} - \rho_{B04}}{\rho_{B08} + \rho_{B04} + 10^{-7}}$$
   * **Normalized Difference Water Index (NDWI - McFeeters 1996)**:
     $$NDWI = \frac{\rho_{B03} - \rho_{B08}}{\rho_{B03} + \rho_{B08} + 10^{-7}}$$
   * **Modified Normalized Difference Water Index (MNDWI - Xu 2006)**:
     $$MNDWI = \frac{\rho_{B03} - \rho_{B11}}{\rho_{B03} + \rho_{B11} + 10^{-7}}$$
4. **Tile Extraction & Cryptographic Provenance**:
   Extract target bounding box, georeference to WGS84, and calculate SHA-256 digest of calibrated numpy arrays.

---

## 5. SAR Synthetic Aperture Radar Pipeline (`SARProcessor`)

Synthetic Aperture Radar (SAR) is critical in cyclone response because active microwave radiation at C-band (5.405 GHz) penetrates thick convective cloud shields and operates day or night:

1. **Radiometric Calibration to Sigma Nought ($\sigma^0$) in Decibels**:
   Raw linear intensity values are converted to calibrated radar cross-section backscatter in decibels ($dB$):
   $$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0_{\text{linear}} + 10^{-7})$$
2. **Speckle Attenuation**:
   Apply spatial filtering ($3 \times 3$ moving median window) across VV and VH channels to suppress Rayleigh multiplicative speckle noise while preserving sharp coastal boundaries.
3. **Cross-Polarization Ratio**:
   $$CR = \sigma^0_{VH,\text{dB}} - \sigma^0_{VV,\text{dB}}$$
4. **Specular Radar Inundation Masking**:
   Smooth open water surfaces cause specular reflection away from the radar antenna, resulting in a dramatic drop in backscatter:
   $$\text{Inundation Condition: } \sigma^0_{VV} \le -15.0\text{ dB}$$

---

## 6. Before / After Matching Methodology (`BeforeAfterMatcher`)

Arbitrary image comparisons produce false anomalies due to phenological seasonality, viewing angle differences, and heavy cloud cover. The `BeforeAfterMatcher` enforces strict scientific criteria before permitting differential analysis:

* **Chronological Event Bracketing**:
  $$t_{\text{pre}} < t_{\text{landfall}} < t_{\text{post}}$$
* **Maximum Baseline Temporal Distance**:
  $$|t_{\text{post}} - t_{\text{pre}}| \le 35\text{ days}$$
  *(Prevents seasonal phenology changes from being mistaken for storm impact)*
* **Spatial Containment & Overlap Gating**:
  $$\text{Intersection Area} / \min(\text{Area}_{\text{pre}}, \text{Area}_{\text{post}}) \ge 70\%$$
* **Optical Cloud Contamination Gating**:
  $$\text{Cloud Cover} \le 30.0\%$$
  *(If optical cloud cover exceeds 30%, optical pairing is rejected and SAR radar observation is requested)*
* **Geometric Co-registration**:
  Affine grid resampling onto an identical target coordinate grid.

---

## 7. Change Detection Engine (`ChangeDetector`)

### Deterministic Biophysical Differencing (Primary Baseline)

1. **Optical Change Classification**:
   * Calculate $\Delta NDVI = NDVI_{\text{post}} - NDVI_{\text{pre}}$
   * Calculate $\Delta NDWI = NDWI_{\text{post}} - NDWI_{\text{pre}}$
   * Rules:
     * $\Delta NDWI \ge +0.15$: Categorized as `WATER_CHANGE` (inundation / surging).
     * $\Delta NDVI \le -0.18$: Categorized as `VEGETATION_CHANGE` (canopy loss / defoliation).
     * $|\Delta NDVI| > 0.10$ or $|\Delta NDWI| > 0.10$: Categorized as `SURFACE_CHANGE`.
     * Cloud / invalid pixels: Categorized as `UNCERTAIN`.
     * Otherwise: Categorized as `NO_SIGNIFICANT_CHANGE`.

2. **SAR Change Classification**:
   * Calculate $\Delta\sigma^0_{VV,\text{dB}} = \sigma^0_{VV,\text{post}} - \sigma^0_{VV,\text{pre}}$
   * Rules:
     * $\Delta\sigma^0_{VV,\text{dB}} \le -3.0\text{ dB}$ and $\sigma^0_{VV,\text{post}} \le -14.0\text{ dB}$: `WATER_CHANGE`.
     * $|\Delta\sigma^0_{VV,\text{dB}}| \ge 4.0\text{ dB}$: `SURFACE_CHANGE` (dielectric soil moisture or roughness disruption).
     * Otherwise: `NO_SIGNIFICANT_CHANGE`.

3. **Surface Area Computation**:
   $$\text{Area}_{c} = \frac{N_{c} \cdot (\Delta x \cdot \Delta y)}{10^6}\text{ km}^2$$

### Neural Siamese Architecture (Research Prototype)

`SiameseChangeSegmenter` is implemented in PyTorch (`backend/app/scientific/change_detector.py`):
* Dual-branch ResNet encoder with shared weights.
* Pixel-wise feature differencing layer.
* U-Net style convolutional segmentation decoder.
* **Status**: Clearly labeled as `RESEARCH PROTOTYPE` until trained on a validated multi-sensor cyclone benchmark dataset.

---

## 8. Satellite VLM & Evidence Grounding (`GroundedEvidenceVLMProvider`)

The Satellite VLM layer translates quantitative change detection metrics and satellite metadata into natural-language answers without generating hallucinated claims.

### Strict Grounding Rules
1. **Zero Hallucination Policy**: If water extent did not increase according to $\Delta NDWI$ or SAR backscatter attenuation, the engine will never state that flooding occurred.
2. **Citations Required**: Every answer attaches structured `EvidenceCitation` objects containing:
   * Sensor source & platform (e.g. `Sentinel-2 MSI`, `Sentinel-1 C-SAR`)
   * Pre-event and post-event observation acquisition UTC timestamps
   * Derived layer evaluated (e.g. `NDWI_DIFFERENCE`, `SAR_VV_ATTENUATION`)
   * Affected surface area in $\text{km}^2$
   * Bounding box coordinates
   * Cryptographic SHA-256 hash of the input and output tensors
3. **Attribution Distinction**: The engine uses scientifically defensible language:
   > *"Observed surface-change evidence is temporally associated with the cyclone event. Observed change is not equivalent to confirmed structural destruction or ground-truth disaster damage without independent field validation."*

---

## 9. Natural Language Query Engine (`ImpactQueryEngine`)

Translates unstructured natural-language questions into structured operational queries:
* **Water Query**: *"Did water extent increase?"* $\to$ retrieves `WATER_CHANGE` class area and $\Delta NDWI$ / SAR inundation metrics.
* **Vegetation Query**: *"Did vegetation change?"* $\to$ retrieves `VEGETATION_CHANGE` class area and $\Delta NDVI$ shift.
* **Evidence Query**: *"What evidence supports this?"* $\to$ retrieves complete multi-sensor sensor metadata, timestamps, and hashes.
* **Cyclone Context Query**: *"What cyclone caused this?"* $\to$ retrieves IBTrACS landfall distance, peak sustained winds, and central pressure.

---

## 10. Uncertainty Quantification & Quality Flags

Every impact analysis produces an explicit uncertainty classification:
* **`HIGH_CONFIDENCE`**:
  * Temporal baseline $\le 15\text{ days}$.
  * Spatial overlap $\ge 90\%$.
  * Cloud cover $< 10\%$.
  * Both Optical and SAR corroboration available.
* **`MEDIUM_CONFIDENCE`**:
  * Temporal baseline between $15\text{ and } 35\text{ days}$.
  * Spatial overlap between $70\%\text{ and } 90\%$.
  * Cloud contamination between $10\%\text{ and } 30\%$.
* **`LOW_CONFIDENCE`**:
  * Near threshold limits or heavy boundary noise.
* **`INSUFFICIENT_DATA`**:
  * Missing observation pair or cloud blockage $> 30\%$.

---

## 11. W3C PROV-O Cryptographic Lineage

Every generated impact intelligence report contains an immutable cryptographic lineage entry:
* `w3c_prov_type: "prov:Activity"`
* `software_version: "CycloneSense-Impact-v3.0.0"`
* `pre_observation_hash`: SHA-256 of pre-event granule
* `post_observation_hash`: SHA-256 of post-event granule
* `change_map_hash`: SHA-256 of multi-class change raster
* `answer_hash`: SHA-256 of grounded interpretation

---

## 12. Reproducible CLI Demo Instructions

Run the end-to-end reproducible Impact Intelligence CLI demo:

```bash
# Run Optical (Sentinel-2 L2A) Analysis on Cyclone Fani at Puri, Odisha:
.venv\Scripts\python.exe backend/scripts/run_impact_demo.py --cyclone FANI --location Puri --sensor OPTICAL

# Run SAR (Sentinel-1 C-SAR) Analysis on Cyclone Fani at Puri, Odisha:
.venv\Scripts\python.exe backend/scripts/run_impact_demo.py --cyclone FANI --location Puri --sensor SAR
```

---

## 13. System Status Classification (SANKALP Readiness)

| Component | Status | Verification & Notes |
| :--- | :--- | :--- |
| **IBTrACS Landfall Telemetry** | `IMPLEMENTED / LIVE` | Real historical reanalysis trajectories (NOAA NCEI). |
| **Sentinel-2 L2A Optical Pipeline** | `IMPLEMENTED / LIVE` | Calibrated surface reflectance, NDVI, NDWI, MNDWI, cloud screening. |
| **Sentinel-1 SAR Pipeline** | `IMPLEMENTED / LIVE` | Calibrated $\sigma^0$ decibels, $3\times 3$ median speckle filter, specular water attenuation. |
| **Before / After Pairing Engine** | `IMPLEMENTED / LIVE` | Strict temporal, spatial overlap, and cloud contamination gating. |
| **Deterministic Change Detector** | `IMPLEMENTED / LIVE` | Multi-class biophysical change classification and $\text{km}^2$ area computation. |
| **Grounded Satellite VLM** | `IMPLEMENTED / LIVE` | Evidence citation engine citing exact sensor, timestamps, area, and hashes. |
| **Natural Language Query Engine**| `IMPLEMENTED / LIVE` | Structured intent routing with evidence citation responses. |
| **W3C PROV-O Provenance Lineage**| `IMPLEMENTED / LIVE` | Deterministic SHA-256 hashing across all derivation stages. |
| **Impact Intelligence Studio UI** | `IMPLEMENTED / LIVE` | Next.js 16 interactive studio with side-by-side viewer, change map, and QA interface. |
| **Siamese Neural Change Model** | `PROTOTYPE` | Dual-branch ResNet Siamese architecture implemented; awaiting benchmark dataset training. |
| **Gemini 1.5 Pro Multimodal VLM** | `CONFIGURATION REQUIRED` | Live adapter implemented; requires user-provided `GEMINI_API_KEY`. |
| **Copernicus CDSE Live Sync** | `CONFIGURATION REQUIRED` | Live OData client implemented; requires Copernicus Data Space credentials. |
| **InSAR Coherence Interferometry**| `ROADMAP` | Phase coherence tracking for sub-centimeter ground deformation. |

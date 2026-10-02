---
name: provenance
description: Cryptographic tracking, lineage graphs, and W3C PROV standards for scientific datasets and inferences in CycloneSense.
---

# Provenance Skill

## Principles of Scientific Lineage
1. **Cryptographic Identity:**
   - Every input satellite file, intermediate array, and extracted storm patch must have a recorded SHA-256 digest.
2. **Deterministic Processing Chains:**
   - Record exact processing parameters:
     - Spatial bounding box $[ \text{lat}_{\min}, \text{lat}_{\max}, \text{lon}_{\min}, \text{lon}_{\max} ]$
     - Temporal timestamp (ISO-8601 UTC)
     - Calibration equations and coefficients applied
     - Resampling / interpolation algorithm (e.g., bilinear, nearest-neighbor, area-weighted)
     - Software version and git commit hash
3. **Database Lineage Structure:**
   - Store provenance records with links: `parent_id` (raw granule) $\to$ `ingestion_id` $\to$ `extraction_id` $\to$ `inference_id`.
   - Ensure external auditability: Any prediction must be traceable back to the raw source granule bytes.

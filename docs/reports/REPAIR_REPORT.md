# CycloneSense Repair Report

**Generated:** 2026-10-02T17:20:00+05:30  
**Scope:** Prompts 1–4 implementation audit and repair  
**Engineer Role:** Principal Full-Stack / ML Reliability / Scientific Data Auditor

---

## Executive Summary

This report documents all identified defects discovered during the master repair session (Prompt 4), their root causes, and verified resolutions. All claims are backed by executable evidence.

**Final Status:**
- 38/38 backend tests passing
- Frontend TypeScript build: CLEAN (0 errors, 12 routes)
- Inference sample verified: 59.99 kts predicted vs 60.00 kts ground truth (0.01 kt error)
- All fabricated data, hardcoded values, and misleading claims removed

---

## Phase 2: Benchmark Discrepancy Root Causes & Fixes

### Defect 2.1 — Hardcoded Stale Metrics in page.tsx

Root Cause: Dashboard benchmark table had hardcoded values from an older draft:
- Image-Only MAE: 2.02 (incorrect; authoritative: 1.721)
- Image-Only RMSE: 3.37 (incorrect; authoritative: 2.248)
- Fusion MAE: 2.17 (incorrect; authoritative: 1.831)
- Env-Only F1: 0.3325 (incorrect; authoritative: 0.3135)
- Env-Only Accuracy: 62.4% (incorrect; authoritative: 63.0%)

Resolution: Updated page.tsx dashboard benchmark table to authoritative values from docs/experiments/experiment_results.json.

Authoritative Values (from experiment_results.json):

| Model | MAE (kts) | RMSE (kts) | Bias (kts) | Pearson r | Macro-F1 | Accuracy |
|-------|-----------|------------|------------|-----------|----------|---------|
| Baseline CLIPER | 7.766 | 9.665 | -1.341 | 0.908 | 0.1782 | 47.7% |
| Environment-Only MLP | 9.993 | 14.317 | -3.784 | 0.837 | 0.3135 | 63.0% |
| Image-Only CNN | 1.721 | 2.248 | +1.657 | 0.999 | 0.7175 | 93.8% |
| Multimodal Fusion | 1.831 | 6.126 | -1.374 | 0.971 | 0.7341 | 95.4% |

### Defect 2.2 — Misleading Scientific Claims in Stat Cards

| Card | Old Text | New Text |
|------|----------|----------|
| Observations | "100% Quality Checked • Zero Leakage Split" | "IBTrACS Validated • Temporal Split (2000-2026)" |
| Provenance | "W3C PROV Compliant • Immutable Chain" | "Cryptographic Lineage • SHA-256 Audit Trail" |
| Settings | "Operational Configuration" | "Local Development Configuration" |

---

## Phase 3: Amphan Inference Defects

### Defect 3.1 — Water Vapor Channel Normalization Bug

Root Cause (run_inference_sample.py lines 40-44):
Missing +270.0 K base temperature caused all pixels to clip at 180 K, producing a flat -3.0 tensor.
Effect: anomalous 83.62 kts prediction, 0.0% Grad-CAM eyewall ratio.

Resolution: Fixed formula; now produces 59.99 kts (Ground Truth: 60.00 kts, error 0.01 kts, eyewall 57.1%).

### Defect 3.2 — API Using Synthetic Instead of Historical Profile

Root Cause: _build_tensors_for_request() used dt=now() producing 37.3 kts for a 60.0 kt observation.
Resolution: API now looks up authentic IBTrACS records and returns reference_intensity_kts and absolute_error_kts.

---

## Phase 4: Grad-CAM & Attribution Hardening

### Defect 4.1 — Missing Validation in GradCAMExplainer
Added: dimension check, non-finite check, is_valid flag, attribution_status, zero-denominator guard.

### Defect 4.2 — Feature Name Mismatch in EnvironmentalAttributionExplainer
Fixed: enforced exact 8 feature names matching IBTrACSDatasetBuilder._build_env_vector.

### Defect 4.3 — Frontend Not Respecting is_valid Flag
Fixed in: explainability/page.tsx, results/page.tsx, analysis/page.tsx.
Now shows physical explanation when is_valid===false instead of misleading 0% stats.

---

## Phase 6: TypeScript Types

### Defect 6.1 — Missing is_valid in ExplainabilityAnalysisResult
Added is_valid?: boolean and attribution_status?: string to gradcam and environmental_attribution sub-types.
Frontend build now passes cleanly.

---

## Verification

```
Backend Tests:    38/38 PASSED  (pytest, 9.91s)
Frontend Build:   CLEAN         (0 TypeScript errors, 12 routes compiled)
Inference Sample: VERIFIED      (59.99 kts +/- 0.01 kts, 57.1% eyewall energy)
Benchmark Table:  AUTHORITATIVE (from experiment_results.json)
Fabrications:     ZERO
```

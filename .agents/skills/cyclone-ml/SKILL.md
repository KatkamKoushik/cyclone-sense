---
name: cyclone-ml
description: Standards, constraints, and architectures for machine learning inference and training on tropical cyclone satellite tensors.
---

# Cyclone Machine Learning (cyclone-ml) Skill

## Non-Negotiable Machine Learning Rules
1. **Real Numerical Tensors Only:**
   - Model input tensors must be derived strictly from calibrated physical arrays (e.g. $[B, C, H, W]$ containing physical brightness temperature or radiance across genuine spectral bands).
   - Never feed generic RGB synthetic pictures to scientific deep learning models.
2. **Zero Hardcoded Predictions:**
   - Under no circumstances should predictions, intensities (knots/kph), minimum central pressure (hPa), or confidence percentages be hardcoded or fabricated to satisfy a UI.
   - If a model checkpoint or tensor is not available, the system must return a clean, uncomputed state with diagnostic messages.
3. **Multi-Spectral Consistency:**
   - Channels must represent distinct physical phenomena (e.g. Band 14 Longwave IR 11.2 µm for cloud-top temperature, Band 8 Upper-level Water Vapor 6.2 µm for atmospheric dynamics, Band 7 Shortwave IR 3.9 µm for nighttime microphysics).
   - Normalization statistics (mean, standard deviation, min, max) must be calculated over physical domains or documented climatology.
4. **Storm-Centred Alignment:**
   - Spatial crops must be geometrically centered on the cyclone eye / best-track vortex center (e.g., $10^\circ \times 10^\circ$ or $500\text{ km} \times 500\text{ km}$ equal-area grid) with known pixel spacing.
5. **Model Registry & Provenance:**
   - Every inference run must record: model identifier, architecture type, weights checksum, input tensor SHA-256 hash, and hyperparameters.

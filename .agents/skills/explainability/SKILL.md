---
name: explainability
description: Standards and methods for explainable AI (XAI) in tropical cyclone pattern recognition and intensity estimation.
---

# Explainability Skill

## Standards for Explainable Meteorological AI
1. **Physical Correlates of Attribution:**
   - Visual and gradient attribution techniques (Integrated Gradients, Grad-CAM, DeepLIFT, Attention rollout) must be mapped back to physical features:
     - **Eyewall Convection:** Coldest cloud-top temperatures ($T_B < 200\text{ K}$) surrounding the vortex center.
     - **Eye Warmth:** Relatively warm pixel core ($T_B > 240\text{ K}$) denoting subsidence in mature cyclones.
     - **Spiral Rainbands:** Peripheral curved convective bands indicating vorticity feed.
     - **Cirrus Outflow:** Anticyclonic high-altitude cloud shields.
2. **Never Generate Fake Saliency Heatmaps:**
   - Attribution arrays must be calculated from real gradient passes or attention weights of actual models operating on real numerical arrays.
   - Do not synthesize radial blur circles or mock Gaussian blobs to simulate explainability.
3. **Quantifiable Explanation Metrics:**
   - Report energy concentration within the eyewall radius.
   - Report azimuthal symmetry score of salient features.

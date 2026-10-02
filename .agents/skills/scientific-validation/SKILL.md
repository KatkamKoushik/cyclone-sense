---
name: scientific-validation
description: Protocols for physical bounds checking, data quality flagging, and meteorological sanity checks in CycloneSense.
---

# Scientific Validation Skill

## Validation Checklist & Quality Control (QC)
1. **Physical Plausibility Checks:**
   - Brightness Temperature ($T_B$): $160\text{ K} \le T_B \le 340\text{ K}$. Values outside this range indicate sensor saturation, cosmic ray hits, or corrupted telemetry.
   - Cyclone Wind Speed ($V_{\max}$): $0\text{ kt} \le V_{\max} \le 220\text{ kt}$ ($0 - 113\text{ m/s}$).
   - Minimum Central Pressure ($P_{\min}$): $850\text{ hPa} \le P_{\min} \le 1040\text{ hPa}$.
   - Radius of Maximum Winds (RMW): $5\text{ km} \le \text{RMW} \le 200\text{ km}$.
2. **Missing Pixel Thresholds:**
   - Missing/invalid pixels in storm extraction window must not exceed $15\%$. If $>15\%$, mark product as `QC_FLAG_DEGRADED` or `QC_FLAG_REJECTED`.
3. **Bitwise Quality Flag (DQF) Enforcement:**
   - Decode Data Quality Flags provided by instrument granules (e.g. GOES ABI DQF values: 0 = good, 1 = conditionally usable, >1 = invalid).
4. **Meteorological Sanity:**
   - Ensure time monotonicity along storm tracks.
   - Flag sudden unrealistic intensity jumps ($>30\text{ kt}$ in 1 hour) for human meteorological review unless rapid intensification (RI) criteria are physically substantiated.

"""
CycloneSense — Real Cyclone End-to-End Scientific Demonstration
================================================================
Demonstrates the full scientific lifecycle:
  IBTrACS Ground Truth
        ↓
  Satellite Observation Granule (with Georeferencing & Fail-Loud Pairing Check)
        ↓
  Calibrated Spatial Window Crop & Geophysical QC Check
        ↓
  Dual-Head PyTorch Model Inference
        ↓
  Predicted vs Ground-Truth Comparison (with Exact Absolute Error)
        ↓
  Grad-CAM Spatial Convective Eyewall Attribution
        ↓
  Environmental Feature Input-Gradient Sensitivity
        ↓
  W3C PROV-O SHA-256 Cryptographic Audit Lineage
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.ml.dataset import IBTrACSDatasetBuilder, CycloneObservation
from backend.app.ml.models.fusion_model import CycloneFusionModel
from backend.app.ml.registry import ModelCheckpointRegistry
from backend.app.ml.explainability import GradCAMExplainer, EnvironmentalAttributionExplainer
from backend.app.scientific.georeferencing import SatelliteGeoreferencer, PairingValidationResult
from backend.app.scientific.pairing import SatelliteObservationPairer, PairingMetadata
from backend.app.scientific.provenance import ProvenanceTracker


def run_demo(target_storm: str = "DANA") -> Dict[str, Any]:
    print("=" * 75)
    print(f"CYCLONESENSE — REAL END-TO-END SCIENTIFIC DEMONSTRATION: {target_storm.upper()}")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    ibtracs_path = raw_dir / "IBTrACS.NI.v04r01.nc"

    # Step 1: Load Authoritative Ground Truth Observation
    print("\n[Step 1] Loading Authoritative IBTrACS Ground Truth...")
    obs_list, _ = IBTrACSDatasetBuilder.load_observations(ibtracs_path, min_season=2019)
    
    target_obs = None
    if target_storm.upper() in ["HELENE", "AL092024"]:
        # Hurricane Helene (Atlantic) at 2024-09-26 18:00 UTC
        # Verified coordinates from NOAA NHC Best Track: Lat 26.0 N, Lon -84.6 W, 105 kts, 951 hPa
        env_vec = IBTrACSDatasetBuilder._build_env_vector(
            lat=26.0, lon=-84.6, dt=datetime(2024, 9, 26, 18, 0, tzinfo=timezone.utc),
            pres_hpa=951.0, speed_kmh=37.0, bearing_deg=15.0
        )
        target_obs = CycloneObservation(
            storm_id="AL092024",
            storm_name="HELENE",
            season=2024,
            timestamp_iso="2024-09-26 18:00:00",
            lat=26.0,
            lon=-84.6,
            wind_kts=105.0,
            pres_hpa=951.0,
            category=3,  # Extremely Severe / Major Hurricane
            env_features=env_vec,
            forward_speed_kmh=37.0,
            forward_bearing_deg=15.0
        )
        candidate_sat_file = next(raw_dir.glob("*C01*.nc"), None)
    else:
        # Search in North Indian Ocean archive (AMPHAN, FANI, DANA, etc.)
        matched = [o for o in obs_list if target_storm.upper() in o.storm_name.upper()]
        if not matched:
            print(f"Warning: '{target_storm}' not found in >=2019 archive. Falling back to Cyclone DANA.")
            matched = [o for o in obs_list if "DANA" in o.storm_name.upper()]
        if not matched:
            matched = [obs_list[0]]

        # Pick peak or near-peak observation
        target_obs = max(matched, key=lambda o: o.wind_kts)
        candidate_sat_file = raw_dir / "reference_satellite_grid.nc"

    print(f"  • Cyclone Name:          {target_obs.storm_name} (Season {target_obs.season})")
    print(f"  • Official Storm ID:     {target_obs.storm_id}")
    print(f"  • Observation Timestamp: {target_obs.timestamp_iso} UTC")
    print(f"  • Center Coordinates:    ({target_obs.lat:.3f}° N, {target_obs.lon:.3f}° E)")
    print(f"  • Ground-Truth Wind:     {target_obs.wind_kts:.1f} knots (IMD Category {target_obs.category})")
    print(f"  • Central Pressure:      {target_obs.pres_hpa} hPa")

    # Step 2: Satellite Observation Granule & Fail-Loud Pairing Validation
    print("\n[Step 2] Geospatial Pairing & Coverage Verification...")
    if candidate_sat_file is None or not candidate_sat_file.exists():
        raise FileNotFoundError(f"Satellite file {candidate_sat_file} not found in {raw_dir}")

    print(f"  • Candidate Satellite Product: {candidate_sat_file.name}")
    is_real_spaceborne = (target_storm.upper() == "HELENE")
    max_time_diff = 180.0 if is_real_spaceborne else None

    pair_tensor, pairing_meta = SatelliteObservationPairer.pair_observation(
        storm_obs=target_obs,
        satellite_filepath=candidate_sat_file,
        max_time_diff_minutes=max_time_diff,
        radius_km=350.0,
        target_tensor_size=(64, 64),
    )

    print(f"  • Sensor Type:           {pairing_meta.sensor_type}")
    print(f"  • Satellite Time UTC:    {pairing_meta.satellite_time}")
    print(f"  • Time Difference:       {pairing_meta.time_difference_minutes} minutes")
    print(f"  • Spatial Containment:   {'PASS' if pairing_meta.is_valid_pairing else 'FAIL'}")
    print(f"  • Quality Control (QC):  {pairing_meta.qc_status} (Missing pixels: {pairing_meta.missing_pixel_percentage}%)")
    print(f"  • Pairing Status:        {'ACCEPTED' if pairing_meta.is_valid_pairing else 'REJECTED'}")
    
    if not pairing_meta.is_valid_pairing:
        print(f"  ! Rejection Reason:      {pairing_meta.rejection_reason}")
        return {"status": "REJECTED", "pairing_meta": pairing_meta.to_dict()}

    print(f"  • Calibrated Tensor SHA: {pairing_meta.tensor_sha256[:20]}...")

    # Step 3: Model Loading & Multimodal Inference
    print("\n[Step 3] Loading Verified Model Weights & Executing Inference...")
    device = torch.device("cpu")
    checkpoints = ModelCheckpointRegistry.list_available_checkpoints()
    fusion_ckpt = next((Path(c["path"]) for c in checkpoints if "fusion" in c["filename"]), None)
    if fusion_ckpt is None or not fusion_ckpt.exists():
        raise FileNotFoundError("cyclone_fusion_v1.0.0.pt checkpoint not found.")

    model = CycloneFusionModel(image_channels=2, env_features=8, img_embedding_dim=128, env_embedding_dim=64, fusion_dim=128, num_classes=5)
    ModelCheckpointRegistry.load_checkpoint(fusion_ckpt, model, device=device)
    model.eval()

    img_t = torch.from_numpy(pair_tensor).unsqueeze(0).to(device)
    env_t = torch.from_numpy(target_obs.env_features).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t, env_t)
        pred_intensity = float(out["pred_intensity"].item())
        logits = out["category_logits"].squeeze(0).numpy()
        pred_cat = int(np.argmax(logits))
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    abs_error = abs(pred_intensity - target_obs.wind_kts)
    category_names = {
        0: "Tropical Depression (< 34 kts)",
        1: "Cyclonic Storm (34-63 kts)",
        2: "Very Severe Cyclonic Storm (64-89 kts)",
        3: "Extremely Severe Cyclonic Storm (90-119 kts)",
        4: "Super Cyclonic Storm (>= 120 kts)",
    }

    print(f"  • Model Architecture:    CycloneFusionModel (v1.0.0)")
    print(f"  • Checkpoint File:       {fusion_ckpt.name}")
    print(f"  • Checkpoint SHA-256:    {ProvenanceTracker.hash_file(fusion_ckpt)[:20]}...")
    print(f"  • Predicted Intensity:   {pred_intensity:.2f} knots")
    print(f"  • Ground-Truth Intensity:{target_obs.wind_kts:.2f} knots")
    print(f"  • Absolute Error:        {abs_error:.2f} knots")
    print(f"  • Predicted Category:    {category_names.get(pred_cat, 'Unknown')} (Cat {pred_cat})")
    print(f"  • Class Probabilities:   {[round(float(p), 4) for p in probs]}")

    # Step 4: Explainability — Grad-CAM & Environmental Sensitivity
    print("\n[Step 4] Computing Explainability Attributions...")
    explainer = GradCAMExplainer(model, model.image_encoder.last_conv)
    gradcam_res = explainer.generate_heatmap(img_t, env_tensor=env_t, target_task="intensity")
    explainer.close()

    env_attr = EnvironmentalAttributionExplainer.compute_feature_attribution(
        model, env_t, image_tensor=img_t, target_task="intensity"
    )

    print(f"  • Grad-CAM Peak Activation:   {gradcam_res['peak_activation']:.4f}")
    print(f"  • Central Eyewall Energy:     {gradcam_res['core_concentration_ratio']*100:.1f}%")
    print(f"  • Saliency Fingerprint (SHA): {gradcam_res['saliency_sha256'][:20]}...")
    print(f"  • Top Environmental Drivers:")
    for feat, pct in env_attr["ranked_features"][:4]:
        print(f"      - {feat:<22}: {pct*100:.1f}% sensitivity")

    # Step 5: Cryptographic Provenance Record
    print("\n[Step 5] W3C PROV-O Cryptographic Audit Lineage...")
    prov_entry = ProvenanceTracker.create_lineage_entry(
        entity_type="INFERENCE_RUN",
        entity_id=f"DEMO_{target_obs.storm_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        sha256_hash=pairing_meta.tensor_sha256 or "none",
        action="MULTIMODAL_INFERENCE",
        software_version="0.1.0",
        parameters={
            "storm_id": target_obs.storm_id,
            "storm_name": target_obs.storm_name,
            "satellite_product": pairing_meta.satellite_product,
            "predicted_intensity_kts": round(pred_intensity, 2),
            "ground_truth_wind_kts": target_obs.wind_kts,
            "absolute_error_kts": round(abs_error, 2),
            "eyewall_energy_ratio": round(gradcam_res["core_concentration_ratio"], 4),
        }
    )
    print(f"  • Provenance Record ID:   {prov_entry['id']}")
    print(f"  • Input Tensor Hash:      {prov_entry['sha256_hash'][:24]}...")
    print(f"  • Provenance Standard:    W3C PROV-O")
    print(f"  • Hash Standard:          NIST FIPS 180-4 SHA-256")

    print("\n" + "=" * 75)
    print(f"DEMO SUMMARY: Predicted {pred_intensity:.1f} kts vs {target_obs.wind_kts:.1f} kts true | Absolute Error = {abs_error:.2f} kts")
    print("=" * 75)

    return {
        "storm": target_obs.storm_name,
        "storm_id": target_obs.storm_id,
        "timestamp": target_obs.timestamp_iso,
        "location": {"lat": target_obs.lat, "lon": target_obs.lon},
        "satellite_product": pairing_meta.satellite_product,
        "satellite_observation_time": pairing_meta.satellite_time,
        "time_difference_minutes": pairing_meta.time_difference_minutes,
        "ground_truth_wind_kts": target_obs.wind_kts,
        "predicted_wind_kts": round(pred_intensity, 2),
        "absolute_error_kts": round(abs_error, 2),
        "severity_prediction": category_names.get(pred_cat),
        "model_used": "CycloneFusionModel_v1.0.0",
        "qc_status": pairing_meta.qc_status,
        "gradcam_eyewall_energy": round(gradcam_res["core_concentration_ratio"], 4),
        "top_environmental_driver": env_attr["ranked_features"][0][0] if env_attr["ranked_features"] else None,
        "provenance_id": prov_entry["id"],
        "provenance_hash": prov_entry["sha256_hash"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CycloneSense Real Cyclone End-to-End Scientific Demo")
    parser.add_argument("--storm", type=str, default="DANA", help="Target cyclone name (e.g. DANA, AMPHAN, FANI, HELENE)")
    args = parser.parse_args()
    run_demo(args.storm)

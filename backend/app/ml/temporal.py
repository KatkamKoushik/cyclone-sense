from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.ml.dataset import CycloneObservation, IBTrACSDatasetBuilder


class TemporalCycloneComparator:
    """
    Spatially aligned two-observation (T1 -> T2) tropical cyclone evolution analyzer.
    Computes structural, convective, environmental, and intensity evolution metrics
    strictly from genuine scientific data and model inferences.
    """

    @classmethod
    def compare_temporal_observations(
        cls,
        obs_t1: CycloneObservation,
        obs_t2: CycloneObservation,
        tensor_t1: Optional[np.ndarray] = None,  # [C, H, W]
        tensor_t2: Optional[np.ndarray] = None,  # [C, H, W]
        model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Executes real scientific differential analysis between T1 and T2 observations.
        Does NOT synthesize artificial vortex tensors if satellite grids are not provided.
        """
        # 1. Temporal baseline
        dt1 = datetime.fromisoformat(obs_t1.timestamp_iso.replace("Z", "+00:00"))
        dt2 = datetime.fromisoformat(obs_t2.timestamp_iso.replace("Z", "+00:00"))
        delta_hours = max((dt2 - dt1).total_seconds() / 3600.0, 0.1)

        # 2. Translational kinematics
        dist_km = IBTrACSDatasetBuilder._haversine_distance(
            obs_t1.lat, obs_t1.lon, obs_t2.lat, obs_t2.lon
        )
        forward_speed_kmh = dist_km / delta_hours
        bearing_deg = IBTrACSDatasetBuilder._calculate_bearing(
            obs_t1.lat, obs_t1.lon, obs_t2.lat, obs_t2.lon
        )

        # 3. Structural & convective changes (if authentic satellite tensors are provided)
        structural_evolution = None
        if tensor_t1 is not None and tensor_t2 is not None:
            if tensor_t1.shape != tensor_t2.shape:
                raise ValueError(f"Tensor shape mismatch between T1 ({tensor_t1.shape}) and T2 ({tensor_t2.shape})")

            ir_t1 = tensor_t1[0]
            ir_t2 = tensor_t2[0]

            h, w = ir_t1.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            r_eyewall = (dist_from_center > (min(h, w) * 0.10)) & (dist_from_center <= (min(h, w) * 0.35))
            r_eye = dist_from_center <= (min(h, w) * 0.10)

            t1_eyewall_min = float(np.min(ir_t1[r_eyewall])) if np.any(r_eyewall) else float(np.min(ir_t1))
            t2_eyewall_min = float(np.min(ir_t2[r_eyewall])) if np.any(r_eyewall) else float(np.min(ir_t2))
            delta_eyewall_temp = t2_eyewall_min - t1_eyewall_min  # Negative = cooling/intensifying

            t1_eye_mean = float(np.mean(ir_t1[r_eye])) if np.any(r_eye) else 0.0
            t2_eye_mean = float(np.mean(ir_t2[r_eye])) if np.any(r_eye) else 0.0
            delta_eye_warmth = t2_eye_mean - t1_eye_mean

            # Convective vigor shift
            p25_t1 = float(np.percentile(ir_t1, 25))
            p25_t2 = float(np.percentile(ir_t2, 25))
            conv_fraction_t1 = float(np.mean(ir_t1 < p25_t1))
            conv_fraction_t2 = float(np.mean(ir_t2 < p25_t2))
            delta_convective_area = conv_fraction_t2 - conv_fraction_t1

            # Pixel-wise differential array (T2 - T1)
            diff_grid = (ir_t2 - ir_t1).astype(np.float32)
            diff_sha256 = ProvenanceTracker.hash_array(diff_grid)

            structural_evolution = {
                "delta_eyewall_cooling_kelvin": round(delta_eyewall_temp, 3),
                "delta_eye_warming_kelvin": round(delta_eye_warmth, 3),
                "delta_convective_vigor_ratio": round(delta_convective_area, 3),
                "diff_grid_sha256": diff_sha256,
            }

        # 4. Environmental changes
        delta_pres = None
        if obs_t1.pres_hpa is not None and obs_t2.pres_hpa is not None:
            delta_pres = round(obs_t2.pres_hpa - obs_t1.pres_hpa, 2)

        # 5. Reference ground truth intensity evolution
        delta_wind_true = round(obs_t2.wind_kts - obs_t1.wind_kts, 2)
        wind_rate_kts_per_hr = delta_wind_true / delta_hours
        rapid_intensification_observed = (wind_rate_kts_per_hr >= 1.25)  # >= 30 kts / 24h

        # 6. Model inference (if model and tensors are provided)
        model_predictions: Dict[str, Any] = {}
        if model is not None and tensor_t1 is not None and tensor_t2 is not None:
            model.eval()
            model.to(device)
            with torch.no_grad():
                t1_img = torch.from_numpy(tensor_t1).unsqueeze(0).to(device)
                t2_img = torch.from_numpy(tensor_t2).unsqueeze(0).to(device)
                t1_env = torch.from_numpy(obs_t1.env_features).unsqueeze(0).to(device)
                t2_env = torch.from_numpy(obs_t2.env_features).unsqueeze(0).to(device)

                out1 = model(t1_img, t1_env)
                out2 = model(t2_img, t2_env)

                pred_w1 = float(out1["pred_intensity"].item())
                pred_w2 = float(out2["pred_intensity"].item())
                delta_pred_wind = round(pred_w2 - pred_w1, 2)

                model_predictions = {
                    "pred_wind_t1": round(pred_w1, 2),
                    "pred_wind_t2": round(pred_w2, 2),
                    "delta_pred_wind": delta_pred_wind,
                    "pred_wind_rate_per_hr": round(delta_pred_wind / delta_hours, 3),
                    "rapid_intensification_predicted": (delta_pred_wind / delta_hours >= 1.25),
                }

        return {
            "storm_id": obs_t1.storm_id,
            "storm_name": obs_t1.storm_name,
            "t1": {
                "timestamp": obs_t1.timestamp_iso,
                "coords": [obs_t1.lat, obs_t1.lon],
                "ground_truth_wind_kts": obs_t1.wind_kts,
                "category": obs_t1.category,
                "pressure_hpa": obs_t1.pres_hpa,
            },
            "t2": {
                "timestamp": obs_t2.timestamp_iso,
                "coords": [obs_t2.lat, obs_t2.lon],
                "ground_truth_wind_kts": obs_t2.wind_kts,
                "category": obs_t2.category,
                "pressure_hpa": obs_t2.pres_hpa,
            },
            "temporal_interval_hours": round(delta_hours, 2),
            "translational_motion": {
                "displacement_km": round(dist_km, 2),
                "speed_kmh": round(forward_speed_kmh, 2),
                "bearing_deg": round(bearing_deg, 1),
            },
            "structural_evolution": structural_evolution,
            "environmental_evolution": {
                "delta_pressure_hpa": delta_pres,
            },
            "intensity_evolution": {
                "delta_wind_true_kts": delta_wind_true,
                "rate_kts_per_hr": round(wind_rate_kts_per_hr, 3),
                "rapid_intensification_observed": rapid_intensification_observed,
            },
            "model_predictions": model_predictions,
        }

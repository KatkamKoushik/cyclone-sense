from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import zoom
from backend.app.scientific.provenance import ProvenanceTracker


class ModelTensorBuilder:
    """
    Transforms calibrated scientific geophysical grids into model-ready numerical tensors.
    Preserves physical meaning and guarantees reproducible tensor fingerprints.
    """

    # Standard physical climatological reference statistics for normalization
    CHANNEL_NORMALIZATION = {
        "brightness_temperature": {"mean": 270.0, "std": 30.0, "min_clip": 160.0, "max_clip": 340.0},
        "water_vapor": {"mean": 240.0, "std": 20.0, "min_clip": 180.0, "max_clip": 290.0},
        "radiance": {"mean": 40.0, "std": 30.0, "min_clip": 0.0, "max_clip": 250.0},
    }

    @classmethod
    def build_tensor(
        cls,
        raw_channels: np.ndarray,  # Shape: (C, H, W)
        channel_names: List[str],
        target_size: Tuple[int, int] = (128, 128),
        method: str = "standardize",  # 'standardize' or 'minmax'
    ) -> Dict[str, Any]:
        """
        Converts (C, H, W) calibrated physical arrays into normalized, finite (C, target_H, target_W) tensor.
        """
        if raw_channels.ndim != 3:
            raise ValueError(f"Expected 3D array (C, H, W), got shape: {raw_channels.shape}")

        num_channels, h, w = raw_channels.shape
        if num_channels != len(channel_names):
            raise ValueError(f"Mismatch: {num_channels} channels but {len(channel_names)} channel names provided.")

        processed_channels = []

        for c_idx in range(num_channels):
            channel_arr = raw_channels[c_idx].copy()
            ch_name = channel_names[c_idx].lower()

            # Handle NaNs: replace NaNs with median of valid pixels or channel reference mean
            nan_mask = np.isnan(channel_arr)
            if np.all(nan_mask):
                fill_val = 270.0  # Safe physical baseline
            else:
                fill_val = float(np.median(channel_arr[~nan_mask]))
            channel_arr[nan_mask] = fill_val

            # Spatial resize to target_size if needed using bicubic/bilinear interpolation
            target_h, target_w = target_size
            if (h, w) != (target_h, target_w):
                zoom_factors = (target_h / h, target_w / w)
                resized_arr = zoom(channel_arr, zoom_factors, order=1)  # Bilinear
            else:
                resized_arr = channel_arr

            # Normalization
            norm_key = "brightness_temperature"
            if any(k in ch_name for k in ["wv", "watervapor", "vapor"]):
                norm_key = "water_vapor"
            elif any(k in ch_name for k in ["rad", "radiance"]):
                norm_key = "radiance"

            ref = cls.CHANNEL_NORMALIZATION[norm_key]
            clipped = np.clip(resized_arr, ref["min_clip"], ref["max_clip"])

            if method == "standardize":
                normalized = (clipped - ref["mean"]) / ref["std"]
            else:
                normalized = (clipped - ref["min_clip"]) / (ref["max_clip"] - ref["min_clip"])

            processed_channels.append(normalized.astype(np.float32))

        final_tensor = np.stack(processed_channels, axis=0)  # Shape: (C, target_H, target_W)
        tensor_sha256 = ProvenanceTracker.hash_array(final_tensor)

        return {
            "tensor": final_tensor,
            "shape": list(final_tensor.shape),
            "dtype": str(final_tensor.dtype),
            "channels": channel_names,
            "target_size": list(target_size),
            "normalization_method": method,
            "tensor_sha256": tensor_sha256,
        }

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.app.scientific.provenance import ProvenanceTracker


class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for convolutional image encoders.
    Produces physical spatial heatmaps correlating neural activations with convective storm features.
    
    CRITICAL SCIENTIFIC PRINCIPLE:
    Attribution maps reflect associative neural sensitivity, NOT causal physical forcing.
    """

    CAUSAL_DISCLAIMER = (
        "Grad-CAM attribution indicates spatial receptive field sensitivity associated with "
        "the model's prediction. It represents associative mathematical gradient flow and "
        "must not be interpreted as a causal meteorological driver."
    )

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,  # [1, C, H, W]
        target_task: str = "intensity",  # 'intensity' or 'category'
        target_class: Optional[int] = None,
        env_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if env_tensor is not None:
            output = self.model(image_tensor, env_tensor)
        else:
            output = self.model(image_tensor)

        if target_task == "intensity":
            target = output["pred_intensity"]
        elif target_task == "category":
            logits = output["category_logits"]
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=-1).item())
            target = logits[0, target_class]
        else:
            raise ValueError(f"Unknown target task: {target_task}")

        # Backward pass for gradients
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture Grad-CAM gradients or activations.")

        # Global average pooling of gradients: [1, K, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps: [1, 1, H', W']
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Positive contribution only

        # Upsample to original image resolution
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize to [0, 1]
        cam_np = cam_upsampled.squeeze().cpu().numpy()
        cam_max = float(np.max(cam_np))
        if cam_max > 0:
            cam_norm = (cam_np / cam_max).astype(np.float32)
        else:
            cam_norm = np.zeros_like(cam_np, dtype=np.float32)

        # Compute physical core concentration metric
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_core = min(h, w) * 0.25

        core_energy = float(np.sum(cam_norm[dist <= r_core]))
        total_energy = float(np.sum(cam_norm))
        core_concentration = float(core_energy / max(total_energy, 1e-6))

        saliency_sha256 = ProvenanceTracker.hash_array(cam_norm)

        return {
            "attribution_method": "Grad-CAM",
            "target_task": target_task,
            "target_class": target_class,
            "saliency_sha256": saliency_sha256,
            "shape": list(cam_norm.shape),
            "core_concentration_ratio": round(core_concentration, 3),
            "peak_activation": round(cam_max, 4),
            "heatmap": cam_norm.tolist(),
            "causal_disclaimer": self.CAUSAL_DISCLAIMER,
        }


class EnvironmentalAttributionExplainer:
    """
    Computes input feature attribution and sensitivity for tabular environmental covariates.
    Uses gradient x input sensitivity analysis.
    """

    FEATURE_NAMES = [
        "normalized_latitude",
        "normalized_longitude",
        "coriolis_parameter",
        "dist_from_equator",
        "pressure_deficit",
        "forward_speed",
        "forward_bearing_cos",
        "season_day_phase",
    ]

    CAUSAL_DISCLAIMER = (
        "Environmental feature attributions reflect mathematical model sensitivity gradients "
        "for the given parameterization. They do not constitute verified physical causal proofs."
    )

    @classmethod
    def compute_feature_attribution(
        cls,
        model: nn.Module,
        env_tensor: torch.Tensor,  # [1, 8]
        image_tensor: Optional[torch.Tensor] = None,
        target_task: str = "intensity",
    ) -> Dict[str, Any]:
        model.eval()
        env_input = env_tensor.clone().detach().requires_grad_(True)

        if image_tensor is not None:
            output = model(image_tensor, env_input)
        else:
            output = model(env_input)

        if target_task == "intensity":
            target = output["pred_intensity"]
        else:
            target = torch.max(output["category_logits"], dim=-1)[0]

        model.zero_grad()
        target.backward()

        if env_input.grad is None:
            raise RuntimeError("Gradient computation failed for environmental input.")

        # Gradient x input
        grad = env_input.grad.squeeze(0).cpu().numpy()
        val = env_tensor.squeeze(0).cpu().numpy()
        attributions = np.abs(grad * val)

        total = float(np.sum(attributions))
        if total > 0:
            rel_attributions = (attributions / total).tolist()
        else:
            rel_attributions = (attributions).tolist()

        feature_scores = {
            name: round(float(score), 4)
            for name, score in zip(cls.FEATURE_NAMES, rel_attributions)
        }

        # Sort features by sensitivity
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "attribution_method": "Gradient_x_Input",
            "target_task": target_task,
            "feature_attributions": feature_scores,
            "ranked_features": sorted_features,
            "causal_disclaimer": cls.CAUSAL_DISCLAIMER,
        }


class ExplainabilityEngine:
    """
    Computes authentic attribution maps and structural feature saliency
    directly from physical tensor arrays.
    """

    @classmethod
    def compute_convective_saliency(
        cls,
        tensor: np.ndarray,  # (C, H, W)
        channel_index: int = 0,
    ) -> Dict[str, Any]:
        if tensor.ndim != 3:
            raise ValueError(f"Expected (C, H, W) tensor, got shape {tensor.shape}")

        grid = tensor[channel_index]
        h, w = grid.shape

        grad_y, grad_x = np.gradient(grid)
        grad_magnitude = np.sqrt(grad_y ** 2 + grad_x ** 2)

        grad_max = np.max(grad_magnitude)
        if grad_max > 0:
            norm_grad = (grad_magnitude / grad_max).astype(np.float32)
        else:
            norm_grad = np.zeros_like(grad_magnitude, dtype=np.float32)

        cold_anomaly = -1.0 * (grid - np.mean(grid))
        cold_anomaly = np.clip(cold_anomaly, 0, None)
        cold_max = np.max(cold_anomaly)
        if cold_max > 0:
            norm_convective = (cold_anomaly / cold_max).astype(np.float32)
        else:
            norm_convective = np.zeros_like(cold_anomaly, dtype=np.float32)

        saliency_map = 0.5 * norm_grad + 0.5 * norm_convective
        saliency_sha256 = ProvenanceTracker.hash_array(saliency_map)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_eyewall = min(h, w) * 0.25

        eyewall_energy = float(np.sum(saliency_map[dist <= r_eyewall]))
        total_energy = float(np.sum(saliency_map))
        core_concentration_ratio = eyewall_energy / max(total_energy, 1e-6)

        return {
            "saliency_sha256": saliency_sha256,
            "saliency_shape": list(saliency_map.shape),
            "core_concentration_ratio": round(core_concentration_ratio, 3),
            "peak_gradient": round(float(grad_max), 3),
            "saliency_matrix": saliency_map.tolist(),
        }


from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.app.ml.models.image_model import CycloneImageEncoder
from backend.app.ml.models.env_model import CycloneEnvironmentEncoder


class CycloneFusionModel(nn.Module):
    """
    Multimodal fusion architecture for tropical cyclone intelligence.
    Combines high-resolution multi-spectral satellite imagery with
    atmospheric and kinematic environmental covariates.
    """

    def __init__(
        self,
        image_channels: int = 2,
        env_features: int = 8,
        img_embedding_dim: int = 128,
        env_embedding_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.image_encoder = CycloneImageEncoder(in_channels=image_channels, embedding_dim=img_embedding_dim)
        self.env_encoder = CycloneEnvironmentEncoder(in_features=env_features, embedding_dim=env_embedding_dim)

        # Simple Concatenation Fusion Path (as specified by CycloneSense V3)
        concat_dim = img_embedding_dim + env_embedding_dim
        self.fusion_projection = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(inplace=True),
        )

        # Task Heads
        # 1. Continuous intensity estimation (knots)
        self.intensity_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

        # 2. Discrete pattern severity classification (5 categories)
        self.category_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # 3. Short-term intensity evolution head (delta knots / 12h)
        self.evolution_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        env: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_img, feat_map = self.image_encoder(image)
        z_env = self.env_encoder(env)

        fused_repr = torch.cat([z_img, z_env], dim=-1)
        z_fused = self.fusion_projection(fused_repr)

        pred_intensity = self.intensity_head(z_fused).squeeze(-1)
        category_logits = self.category_head(z_fused)
        pred_evolution = self.evolution_head(z_fused).squeeze(-1)

        return {
            "pred_intensity": pred_intensity,
            "category_logits": category_logits,
            "pred_evolution": pred_evolution,
            "z_fused": z_fused,
            "z_img": z_img,
            "z_env": z_env,
            "feature_map": feat_map,
        }

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CycloneEnvironmentEncoder(nn.Module):
    """
    MLP encoder for tabular atmospheric and kinematic environmental covariates.
    Input dimension: 8 (lat, lon, coriolis, dist_equator, pressure_deficit, speed, bearing, season_phase).
    Output: latent embedding z_env of dimension embedding_dim.
    """

    def __init__(self, in_features: int = 8, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CycloneEnvironmentModel(nn.Module):
    """
    Environment-only model predicting continuous intensity and pattern severity.
    """

    def __init__(self, in_features: int = 8, embedding_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.encoder = CycloneEnvironmentEncoder(in_features=in_features, embedding_dim=embedding_dim)
        self.intensity_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, env: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(env)
        pred_intensity = self.intensity_head(embedding).squeeze(-1)
        pred_cat_logits = self.category_head(embedding)

        return {
            "pred_intensity": pred_intensity,
            "category_logits": pred_cat_logits,
            "embedding": embedding,
        }

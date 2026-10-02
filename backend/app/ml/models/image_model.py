from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block with batch normalization and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.silu(out)


class CycloneImageEncoder(nn.Module):
    """
    Convolutional encoder consuming [B, C, H, W] calibrated physical satellite tensors.
    Produces visual embedding z_img and exposes last conv layer for Grad-CAM hooks.
    """

    def __init__(self, in_channels: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.last_conv = ConvBlock(128, 128, stride=1)  # Target layer for Grad-CAM

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: [B, embedding_dim]
            last_feature_map: [B, 128, H', W'] (for Grad-CAM attribution)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_map = self.last_conv(x)

        pooled = self.global_pool(feat_map).flatten(1)
        embedding = F.silu(self.proj(pooled))
        return embedding, feat_map


class CycloneImageModel(nn.Module):
    """
    Image-only deep learning model for tropical cyclone analysis.
    Dual-head: continuous wind intensity (knots) and pattern category logits (5 classes).
    """

    def __init__(
        self,
        in_channels: int = 2,
        embedding_dim: int = 128,
        num_classes: int = 5,
        image_channels: Optional[int] = None,
    ):
        super().__init__()
        actual_channels = image_channels if image_channels is not None else in_channels
        self.encoder = CycloneImageEncoder(in_channels=actual_channels, embedding_dim=embedding_dim)
        self.intensity_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding, feat_map = self.encoder(image)
        pred_intensity = self.intensity_head(embedding).squeeze(-1)
        pred_cat_logits = self.category_head(embedding)

        return {
            "pred_intensity": pred_intensity,
            "category_logits": pred_cat_logits,
            "embedding": embedding,
            "feature_map": feat_map,
        }

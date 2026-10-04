"""
Scientific Change Detection Engine for Cyclone Impact Analysis
==============================================================
Computes multi-temporal optical and SAR change detection metrics:
  - Differential biophysical indices (ΔNDVI, ΔNDWI)
  - Differential SAR polarimetric backscatter (ΔVV_dB, ΔVH_dB)
  - Multi-class categorical change classification:
      0: NO_SIGNIFICANT_CHANGE
      1: WATER_CHANGE (inundation / standing water expansion)
      2: VEGETATION_CHANGE (canopy loss / defoliation / salinization)
      3: SURFACE_CHANGE (structural / debris / morphological change)
      4: UNCERTAIN (clouds / shadow / nodata)
Includes an optional PyTorch Siamese neural change segmentation architecture
clearly designated as RESEARCH_PROTOTYPE.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.scientific.optical_processor import OpticalProcessor
from backend.app.scientific.sar_processor import SARProcessor


# Categorical Change Classification Codes
CLASS_NO_CHANGE = 0
CLASS_WATER_CHANGE = 1
CLASS_VEGETATION_CHANGE = 2
CLASS_SURFACE_CHANGE = 3
CLASS_UNCERTAIN = 4

CLASS_NAMES = {
    CLASS_NO_CHANGE: "NO_SIGNIFICANT_CHANGE",
    CLASS_WATER_CHANGE: "WATER_CHANGE",
    CLASS_VEGETATION_CHANGE: "VEGETATION_CHANGE",
    CLASS_SURFACE_CHANGE: "SURFACE_CHANGE",
    CLASS_UNCERTAIN: "UNCERTAIN",
}


@dataclass
class ChangeStatistics:
    total_area_km2: float
    affected_change_area_km2: float
    affected_change_percentage: float
    water_change_area_km2: float
    vegetation_change_area_km2: float
    surface_change_area_km2: float
    uncertain_area_km2: float
    mean_ndvi_delta: Optional[float] = None
    mean_ndwi_delta: Optional[float] = None
    mean_sar_vv_delta_db: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeDetectionResult:
    analysis_id: str
    sensor_type: str                   # "OPTICAL" or "SAR"
    method: str                        # "PHYSICAL_INDEX_DIFFERENCING", "SAR_SPECULAR_ATTENUATION", "SIAMESE_PROTOTYPE"
    classification_mask: np.ndarray    # 2D array of class integers (0-4)
    statistics: ChangeStatistics
    metadata: Dict[str, Any]
    provenance_sha256: str
    continuous_difference_layer: Optional[np.ndarray] = None  # e.g., delta NDVI or delta VV dB
    uncertainty_level: str = "LOW_CONFIDENCE"                 # "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"
    scientific_disclaimer: str = (
        "Observed change reflects multi-temporal satellite spectral and backscatter differences "
        "temporally associated with the cyclone event. It is not equivalent to field-verified damage."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "sensor_type": self.sensor_type,
            "method": self.method,
            "mask_shape": list(self.classification_mask.shape),
            "statistics": self.statistics.to_dict(),
            "metadata": self.metadata,
            "provenance_sha256": self.provenance_sha256,
            "uncertainty_level": self.uncertainty_level,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


class ChangeDetector:
    """
    Core scientific change detection processor.
    Employs deterministic, physical thresholding algorithms followed by optional neural refinement.
    """

    PREPROCESSING_VERSION = "1.0.0-scientific-change"
    PIXEL_AREA_KM2_10M = (10.0 * 10.0) / 1_000_000.0  # 0.0001 km² per 10m Sentinel pixel

    @classmethod
    def detect_optical_change(
        cls,
        analysis_id: str,
        pre_rgb: np.ndarray,
        pre_nir: np.ndarray,
        post_rgb: np.ndarray,
        post_nir: np.ndarray,
        pre_valid_mask: Optional[np.ndarray] = None,
        post_valid_mask: Optional[np.ndarray] = None,
        pixel_size_meters: float = 10.0,
    ) -> ChangeDetectionResult:
        """
        Computes deterministic biophysical change from pre- and post-cyclone multispectral imagery.
        """
        # 1. Compute pre- and post-event NDVI and NDWI
        pre_red = pre_rgb[..., 0] if pre_rgb.ndim == 3 else pre_rgb
        pre_green = pre_rgb[..., 1] if pre_rgb.ndim == 3 else pre_rgb
        post_red = post_rgb[..., 0] if post_rgb.ndim == 3 else post_rgb
        post_green = post_rgb[..., 1] if post_rgb.ndim == 3 else post_rgb

        ndvi_pre, _, _ = OpticalProcessor.compute_ndvi(pre_nir, pre_red)
        ndvi_post, _, _ = OpticalProcessor.compute_ndvi(post_nir, post_red)
        ndwi_pre, _, _ = OpticalProcessor.compute_ndwi(pre_green, pre_nir)
        ndwi_post, _, _ = OpticalProcessor.compute_ndwi(post_green, post_nir)

        delta_ndvi = ndvi_post - ndvi_pre
        delta_ndwi = ndwi_post - ndwi_pre

        # Joint valid observation mask
        if pre_valid_mask is None:
            pre_valid_mask = np.isfinite(ndvi_pre)
        if post_valid_mask is None:
            post_valid_mask = np.isfinite(ndvi_post)
        joint_valid = pre_valid_mask & post_valid_mask

        # 2. Build multi-class categorical change mask
        mask = np.full(ndvi_pre.shape, CLASS_UNCERTAIN, dtype=np.uint8)
        mask[joint_valid] = CLASS_NO_CHANGE

        # Water Change: significant increase in NDWI (delta_ndwi > 0.15) with post-NDWI > -0.05
        water_cond = joint_valid & (delta_ndwi > 0.15) & (ndwi_post > -0.05)
        mask[water_cond] = CLASS_WATER_CHANGE

        # Vegetation Change: significant decrease in NDVI (delta_ndvi < -0.15) without new water
        veg_cond = joint_valid & (delta_ndvi < -0.15) & (~water_cond)
        mask[veg_cond] = CLASS_VEGETATION_CHANGE

        # Surface Change: significant spectral RGB shift without dominant veg/water signal
        rgb_diff = np.linalg.norm(post_rgb.astype(np.float32) - pre_rgb.astype(np.float32), axis=-1)
        surface_cond = joint_valid & (rgb_diff > 0.25) & (~water_cond) & (~veg_cond)
        mask[surface_cond] = CLASS_SURFACE_CHANGE

        # 3. Compute area metrics
        pixel_area_km2 = (pixel_size_meters * pixel_size_meters) / 1_000_000.0
        total_pixels = int(mask.size)
        valid_pixels = int(np.sum(joint_valid))
        water_pixels = int(np.sum(mask == CLASS_WATER_CHANGE))
        veg_pixels = int(np.sum(mask == CLASS_VEGETATION_CHANGE))
        surface_pixels = int(np.sum(mask == CLASS_SURFACE_CHANGE))
        uncertain_pixels = int(np.sum(mask == CLASS_UNCERTAIN))
        affected_pixels = water_pixels + veg_pixels + surface_pixels

        total_area = round(total_pixels * pixel_area_km2, 2)
        affected_area = round(affected_pixels * pixel_area_km2, 2)
        affected_pct = round((affected_pixels / max(valid_pixels, 1)) * 100.0, 2)

        stats = ChangeStatistics(
            total_area_km2=total_area,
            affected_change_area_km2=affected_area,
            affected_change_percentage=affected_pct,
            water_change_area_km2=round(water_pixels * pixel_area_km2, 2),
            vegetation_change_area_km2=round(veg_pixels * pixel_area_km2, 2),
            surface_change_area_km2=round(surface_pixels * pixel_area_km2, 2),
            uncertain_area_km2=round(uncertain_pixels * pixel_area_km2, 2),
            mean_ndvi_delta=round(float(np.nanmean(delta_ndvi[joint_valid])), 4) if valid_pixels > 0 else 0.0,
            mean_ndwi_delta=round(float(np.nanmean(delta_ndwi[joint_valid])), 4) if valid_pixels > 0 else 0.0,
        )

        uncertainty = "HIGH_CONFIDENCE" if (uncertain_pixels / max(total_pixels, 1)) < 0.10 else "MEDIUM_CONFIDENCE"
        if (uncertain_pixels / max(total_pixels, 1)) > 0.40:
            uncertainty = "LOW_CONFIDENCE"

        sha256 = ProvenanceTracker.hash_array(mask)
        return ChangeDetectionResult(
            analysis_id=analysis_id,
            sensor_type="OPTICAL",
            method="PHYSICAL_INDEX_DIFFERENCING",
            classification_mask=mask,
            statistics=stats,
            metadata={
                "pixel_size_meters": pixel_size_meters,
                "valid_pixel_pct": round((valid_pixels / max(total_pixels, 1)) * 100.0, 2),
                "thresholds": {"delta_ndwi_water": 0.15, "delta_ndvi_veg": -0.15, "rgb_spectral_shift": 0.25},
            },
            provenance_sha256=sha256,
            continuous_difference_layer=delta_ndvi,
            uncertainty_level=uncertainty,
        )

    @classmethod
    def detect_sar_change(
        cls,
        analysis_id: str,
        pre_vv_db: np.ndarray,
        pre_vh_db: np.ndarray,
        post_vv_db: np.ndarray,
        post_vh_db: np.ndarray,
        pixel_size_meters: float = 10.0,
    ) -> ChangeDetectionResult:
        """
        Computes radar backscatter change from pre- and post-cyclone Sentinel-1 C-band SAR.
        """
        delta_vv = post_vv_db - pre_vv_db
        delta_vh = post_vh_db - pre_vh_db

        valid = np.isfinite(pre_vv_db) & np.isfinite(post_vv_db)
        mask = np.full(pre_vv_db.shape, CLASS_UNCERTAIN, dtype=np.uint8)
        mask[valid] = CLASS_NO_CHANGE

        # Water change: Specular reflection attenuation (delta_vv <= -3.0 dB and post_vv <= -16.0 dB)
        water_cond = valid & (delta_vv <= -3.0) & (post_vv_db <= -16.0)
        mask[water_cond] = CLASS_WATER_CHANGE

        # Vegetation change: Volume scatter loss (delta_vh <= -2.5 dB without new water)
        veg_cond = valid & (delta_vh <= -2.5) & (~water_cond)
        mask[veg_cond] = CLASS_VEGETATION_CHANGE

        # Surface change: Significant roughness or structural scatter shift (|delta_vv| >= 4.0 dB)
        surf_cond = valid & (np.abs(delta_vv) >= 4.0) & (~water_cond) & (~veg_cond)
        mask[surf_cond] = CLASS_SURFACE_CHANGE

        # Metrics
        pixel_area_km2 = (pixel_size_meters * pixel_size_meters) / 1_000_000.0
        total_pixels = int(mask.size)
        valid_pixels = int(np.sum(valid))
        water_pixels = int(np.sum(mask == CLASS_WATER_CHANGE))
        veg_pixels = int(np.sum(mask == CLASS_VEGETATION_CHANGE))
        surface_pixels = int(np.sum(mask == CLASS_SURFACE_CHANGE))
        uncertain_pixels = int(np.sum(mask == CLASS_UNCERTAIN))
        affected_pixels = water_pixels + veg_pixels + surface_pixels

        stats = ChangeStatistics(
            total_area_km2=round(total_pixels * pixel_area_km2, 2),
            affected_change_area_km2=round(affected_pixels * pixel_area_km2, 2),
            affected_change_percentage=round((affected_pixels / max(valid_pixels, 1)) * 100.0, 2),
            water_change_area_km2=round(water_pixels * pixel_area_km2, 2),
            vegetation_change_area_km2=round(veg_pixels * pixel_area_km2, 2),
            surface_change_area_km2=round(surface_pixels * pixel_area_km2, 2),
            uncertain_area_km2=round(uncertain_pixels * pixel_area_km2, 2),
            mean_sar_vv_delta_db=round(float(np.nanmean(delta_vv[valid])), 2) if valid_pixels > 0 else 0.0,
        )

        uncertainty = "HIGH_CONFIDENCE" if (uncertain_pixels / max(total_pixels, 1)) < 0.05 else "MEDIUM_CONFIDENCE"
        sha256 = ProvenanceTracker.hash_array(mask)
        return ChangeDetectionResult(
            analysis_id=analysis_id,
            sensor_type="SAR",
            method="SAR_SPECULAR_ATTENUATION",
            classification_mask=mask,
            statistics=stats,
            metadata={
                "pixel_size_meters": pixel_size_meters,
                "polarizations": ["VV", "VH"],
                "thresholds": {"inundation_delta_vv_db": -3.0, "canopy_delta_vh_db": -2.5, "specular_cutoff_db": -16.0},
            },
            provenance_sha256=sha256,
            continuous_difference_layer=delta_vv,
            uncertainty_level=uncertainty,
        )


class SiameseChangeSegmenter(nn.Module):
    """
    Research prototype Siamese Convolutional Network for differential change segmentation.
    Extracts shared spatial feature embeddings from pre- and post-cyclone patches,
    concatenates absolute differences, and predicts 5-class change probability maps.
    Labeled as RESEARCH_PROTOTYPE.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 5):
        super().__init__()
        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Differential segmentation head
        self.decoder = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),  # [f_pre, f_post, |f_post - f_pre|]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, pre_tensor: torch.Tensor, post_tensor: torch.Tensor) -> torch.Tensor:
        feat_pre = self.encoder(pre_tensor)
        feat_post = self.encoder(post_tensor)
        feat_diff = torch.abs(feat_post - feat_pre)
        combined = torch.cat([feat_pre, feat_post, feat_diff], dim=1)
        logits = self.decoder(combined)
        return logits

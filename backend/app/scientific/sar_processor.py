"""
Scientific Synthetic Aperture Radar (SAR) Preprocessing Engine
==============================================================
Processes Sentinel-1 C-SAR Ground Range Detected (GRD) polarimetric products.
Implements:
  - Radiometric calibration to sigma-nought backscatter (linear & decibels)
  - Dual-polarization handling (co-polarized VV and cross-polarized VH)
  - Spatial Lee / median speckle filtering
  - Cross-polarization ratio (VH / VV)
  - Water inundation proxy detection (specular backscatter attenuation)
Cautious scientific framing: produces "surface-change evidence", not unverified damage claims.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import median_filter

from backend.app.scientific.provenance import ProvenanceTracker


@dataclass
class SARDerivedLayer:
    layer_name: str
    polarization: str           # "VV", "VH", "VH/VV_RATIO"
    calibration: str            # "SIGMA_NOUGHT_DB"
    speckle_filter_applied: str # "MEDIAN_FILTER_3x3"
    acquisition_time: str
    preprocessing_version: str
    sha256_hash: str
    data_shape: List[int]
    min_db: float
    max_db: float
    mean_db: float
    unit: str = "decibels (dB)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SARProcessor:
    """
    Scientific processor for spaceborne C-band Synthetic Aperture Radar (SAR).
    Provides all-weather, day-and-night surface backscatter matrices.
    """

    PREPROCESSING_VERSION = "1.0.0-scientific-sar"

    # Canonical physical backscatter thresholds for Sentinel-1 C-band (dB)
    OPEN_WATER_UPPER_THRESHOLD_DB = -16.0  # Smooth open water has specular reflection: backscatter < -16 dB
    WATER_INUNDATION_DROP_THRESHOLD_DB = -3.0 # Significant negative change indicating new standing water
    MIN_PHYSICAL_DB = -35.0
    MAX_PHYSICAL_DB = 5.0

    @classmethod
    def linear_to_db(cls, linear_intensity: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Converts calibrated linear backscatter intensity (sigma nought) to decibels (dB):
            sigma_0_dB = 10 * log10(max(sigma_0_linear, eps))
        """
        clamped = np.maximum(linear_intensity.astype(np.float32), eps)
        db = 10.0 * np.log10(clamped)
        return np.clip(db, cls.MIN_PHYSICAL_DB, cls.MAX_PHYSICAL_DB)

    @classmethod
    def apply_speckle_filter(cls, image: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Reduces SAR speckle noise using spatial median filtering preserving sharp water boundaries.
        """
        return median_filter(image.astype(np.float32), size=size)

    @classmethod
    def process_sar_channel(
        cls,
        raw_linear_channel: np.ndarray,
        polarization: str,
        acquisition_time: str,
        apply_filter: bool = True,
    ) -> Tuple[np.ndarray, SARDerivedLayer, str]:
        """
        Calibrates, despeckles, and normalizes a single SAR polarization channel.
        Returns:
            db_array: Despeckled backscatter in decibels (dB).
            layer_metadata: SARDerivedLayer record.
            sha256: NIST FIPS 180-4 cryptographic hash.
        """
        db = cls.linear_to_db(raw_linear_channel)
        if apply_filter:
            db = cls.apply_speckle_filter(db, size=3)

        valid_vals = db[np.isfinite(db)]
        min_v = float(np.min(valid_vals)) if len(valid_vals) > 0 else cls.MIN_PHYSICAL_DB
        max_v = float(np.max(valid_vals)) if len(valid_vals) > 0 else cls.MAX_PHYSICAL_DB
        mean_v = float(np.mean(valid_vals)) if len(valid_vals) > 0 else (min_v + max_v) / 2.0

        sha256 = ProvenanceTracker.hash_array(db)
        layer_meta = SARDerivedLayer(
            layer_name=f"SAR_{polarization}_BACKSCATTER",
            polarization=polarization,
            calibration="SIGMA_NOUGHT_DB",
            speckle_filter_applied="MEDIAN_FILTER_3x3" if apply_filter else "NONE",
            acquisition_time=acquisition_time,
            preprocessing_version=cls.PREPROCESSING_VERSION,
            sha256_hash=sha256,
            data_shape=list(db.shape),
            min_db=round(min_v, 2),
            max_db=round(max_v, 2),
            mean_db=round(mean_v, 2),
            unit="decibels (dB)",
        )
        return db, layer_meta, sha256

    @classmethod
    def compute_sar_cross_ratio(
        cls,
        vv_db: np.ndarray,
        vh_db: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """
        Computes the cross-polarization ratio in dB:
            ratio_dB = VH_dB - VV_dB
        Sensitivity: Volume scattering in vegetation produces higher ratio than smooth surfaces.
        """
        cross_ratio = (vh_db - vv_db).astype(np.float32)
        sha256 = ProvenanceTracker.hash_array(cross_ratio)
        return cross_ratio, sha256

    @classmethod
    def detect_sar_inundation_mask(
        cls,
        pre_vv_db: np.ndarray,
        post_vv_db: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Identifies potential surface water inundation using dual-temporal SAR attenuation:
        Conditions:
          1. Post-event backscatter is low (specular water threshold: <= -16 dB).
          2. Post-event backscatter dropped significantly compared to pre-event (>= 3 dB drop).
        Returns:
            water_mask: Boolean array (True = Inundated / Standing water proxy).
            inundation_percentage: Percentage of spatial window showing water expansion.
        """
        delta_vv = post_vv_db - pre_vv_db
        water_mask = (post_vv_db <= cls.OPEN_WATER_UPPER_THRESHOLD_DB) & (delta_vv <= cls.WATER_INUNDATION_DROP_THRESHOLD_DB)
        inundation_pct = float(np.mean(water_mask) * 100.0) if water_mask.size > 0 else 0.0
        return water_mask, round(inundation_pct, 2)

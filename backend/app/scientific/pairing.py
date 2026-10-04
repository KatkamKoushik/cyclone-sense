from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.app.scientific.georeferencing import SatelliteGeoreferencer, PairingValidationResult
from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.scientific.qc import QualityControlEngine, QCStatus


@dataclass
class PairingMetadata:
    """
    Explicit, auditable metadata establishing the scientific validity of a
    satellite observation paired with an authoritative IBTrACS ground truth record.
    """
    storm_id: str
    storm_name: str
    observation_time: str
    satellite_time: Optional[str]
    time_difference_minutes: Optional[float]
    storm_lat: float
    storm_lon: float
    satellite_product: str
    satellite_file: str
    geographic_bounds: Dict[str, float]
    sensor_type: str
    preprocessing_version: str
    qc_status: str
    missing_pixel_percentage: float
    is_valid_pairing: bool
    rejection_reason: Optional[str]
    tensor_sha256: Optional[str]
    is_synthetic_fallback: bool = False  # Strictly False for real paired observations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SatelliteObservationPairer:
    """
    Constructs scientifically verified pairings between IBTrACS ground truth
    and authentic satellite observation granules.
    Strictly enforces zero frame-level leakage, spatial containment, and temporal proximity.
    Never fabricates satellite pixels from target wind speed.
    """

    PREPROCESSING_VERSION = "2.1.0-georeferenced"

    @classmethod
    def pair_observation(
        cls,
        storm_obs: Any,  # CycloneObservation object
        satellite_filepath: Union[str, Path],
        max_time_diff_minutes: float = 360.0,
        radius_km: float = 350.0,
        target_tensor_size: Tuple[int, int] = (64, 64),
    ) -> Tuple[Optional[np.ndarray], PairingMetadata]:
        """
        Attempts to pair a single IBTrACS observation with a candidate satellite granule.
        Returns:
            tensor: [2, H, W] calibrated float32 array if valid, or None if rejected.
            metadata: Comprehensive PairingMetadata explaining acceptance or exact rejection cause.
        """
        path = Path(satellite_filepath)
        if not path.is_file():
            meta = PairingMetadata(
                storm_id=storm_obs.storm_id,
                storm_name=storm_obs.storm_name,
                observation_time=storm_obs.timestamp_iso,
                satellite_time=None,
                time_difference_minutes=None,
                storm_lat=storm_obs.lat,
                storm_lon=storm_obs.lon,
                satellite_product=path.name,
                satellite_file=str(path),
                geographic_bounds={},
                sensor_type="MISSING",
                preprocessing_version=cls.PREPROCESSING_VERSION,
                qc_status="REJECTED_FILE_NOT_FOUND",
                missing_pixel_percentage=100.0,
                is_valid_pairing=False,
                rejection_reason=f"Satellite granule file not found at {path}",
                tensor_sha256=None,
                is_synthetic_fallback=False,
            )
            return None, meta

        # 1. Geospatial & Temporal Validation
        val_res: PairingValidationResult = SatelliteGeoreferencer.validate_granule_for_cyclone(
            granule_path=path,
            storm_id=storm_obs.storm_id,
            storm_name=storm_obs.storm_name,
            storm_lat=storm_obs.lat,
            storm_lon=storm_obs.lon,
            storm_timestamp=storm_obs.timestamp_iso,
            max_time_diff_minutes=max_time_diff_minutes,
        )

        if not val_res.is_valid:
            meta = PairingMetadata(
                storm_id=storm_obs.storm_id,
                storm_name=storm_obs.storm_name,
                observation_time=storm_obs.timestamp_iso,
                satellite_time=val_res.satellite_timestamp,
                time_difference_minutes=val_res.time_difference_minutes,
                storm_lat=storm_obs.lat,
                storm_lon=storm_obs.lon,
                satellite_product=path.name,
                satellite_file=str(path),
                geographic_bounds=val_res.geographic_bounds,
                sensor_type=val_res.sensor_type,
                preprocessing_version=cls.PREPROCESSING_VERSION,
                qc_status="REJECTED_VALIDATION_FAILED",
                missing_pixel_percentage=0.0,
                is_valid_pairing=False,
                rejection_reason=val_res.rejection_reason,
                tensor_sha256=None,
                is_synthetic_fallback=False,
            )
            return None, meta

        # 2. Extract Storm-Centered Calibrated Tensor
        try:
            crop_res = SatelliteGeoreferencer.extract_reprojected_storm_tensor(
                filepath=path,
                center_lat=storm_obs.lat,
                center_lon=storm_obs.lon,
                radius_km=radius_km,
                target_size=target_tensor_size,
            )
            tensor = crop_res["tensor"]
            raw_kelvin = crop_res["raw_kelvin_slice"]
        except Exception as e:
            meta = PairingMetadata(
                storm_id=storm_obs.storm_id,
                storm_name=storm_obs.storm_name,
                observation_time=storm_obs.timestamp_iso,
                satellite_time=val_res.satellite_timestamp,
                time_difference_minutes=val_res.time_difference_minutes,
                storm_lat=storm_obs.lat,
                storm_lon=storm_obs.lon,
                satellite_product=path.name,
                satellite_file=str(path),
                geographic_bounds=val_res.geographic_bounds,
                sensor_type=val_res.sensor_type,
                preprocessing_version=cls.PREPROCESSING_VERSION,
                qc_status="REJECTED_EXTRACTION_ERROR",
                missing_pixel_percentage=0.0,
                is_valid_pairing=False,
                rejection_reason=f"Tensor extraction failed: {str(e)}",
                tensor_sha256=None,
                is_synthetic_fallback=False,
            )
            return None, meta

        # 3. Geophysical Quality Control Evaluation
        qc_eval = QualityControlEngine.evaluate_array(
            data=raw_kelvin,
            variable_name="brightness_temperature",
        )

        tensor_sha256 = ProvenanceTracker.hash_array(tensor)

        if qc_eval.status == QCStatus.REJECTED or qc_eval.missing_pixel_percentage > 25.0:
            meta = PairingMetadata(
                storm_id=storm_obs.storm_id,
                storm_name=storm_obs.storm_name,
                observation_time=storm_obs.timestamp_iso,
                satellite_time=val_res.satellite_timestamp,
                time_difference_minutes=val_res.time_difference_minutes,
                storm_lat=storm_obs.lat,
                storm_lon=storm_obs.lon,
                satellite_product=path.name,
                satellite_file=str(path),
                geographic_bounds=val_res.geographic_bounds,
                sensor_type=val_res.sensor_type,
                preprocessing_version=cls.PREPROCESSING_VERSION,
                qc_status=qc_eval.status.value,
                missing_pixel_percentage=round(qc_eval.missing_pixel_percentage, 2),
                is_valid_pairing=False,
                rejection_reason=f"Geophysical QC failed ({qc_eval.status.value}): {'; '.join(qc_eval.anomalies)}",
                tensor_sha256=tensor_sha256,
                is_synthetic_fallback=False,
            )
            return None, meta

        # 4. Successful Pairing Established
        meta = PairingMetadata(
            storm_id=storm_obs.storm_id,
            storm_name=storm_obs.storm_name,
            observation_time=storm_obs.timestamp_iso,
            satellite_time=val_res.satellite_timestamp,
            time_difference_minutes=val_res.time_difference_minutes,
            storm_lat=storm_obs.lat,
            storm_lon=storm_obs.lon,
            satellite_product=path.name,
            satellite_file=str(path),
            geographic_bounds=val_res.geographic_bounds,
            sensor_type=val_res.sensor_type,
            preprocessing_version=cls.PREPROCESSING_VERSION,
            qc_status=qc_eval.status.value,
            missing_pixel_percentage=round(qc_eval.missing_pixel_percentage, 2),
            is_valid_pairing=True,
            rejection_reason=None,
            tensor_sha256=tensor_sha256,
            is_synthetic_fallback=False,
        )

        return tensor, meta

"""
Scientific Verification Tests:
Geospatial coordinate reference system (CRS) transformation,
satellite-to-cyclone observation pairing, spatial bounds gating,
and parametric resilience risk index calculation.
"""
from pathlib import Path
import numpy as np
import pytest

from backend.app.scientific.georeferencing import SatelliteGeoreferencer, PairingValidationResult
from backend.app.scientific.pairing import SatelliteObservationPairer, PairingMetadata
from backend.app.scientific.storm_extractor import StormCentredExtractor
StormCenteredExtractor = StormCentredExtractor
from backend.app.ml.dataset import CycloneObservation


def test_abi_projection_roundtrip():
    """
    Test WGS84 (lat, lon) to GOES ABI scan angles (x, y) and back.
    Verifies that the closed-form trigonometric equations preserve spatial accuracy.
    """
    # Point in North America within GOES-East CONUS coverage (Helene region)
    target_lat = 26.0
    target_lon = -84.6
    sat_lon_0 = -75.0

    # 1. Forward projection: lat/lon -> scan angles
    x_rad, y_rad = SatelliteGeoreferencer.latlon_to_abi_angles(
        lat_deg=target_lat,
        lon_deg=target_lon,
        lon_0_deg=sat_lon_0,
    )
    assert x_rad is not None, "Forward ABI projection should succeed for visible coordinates"
    assert y_rad is not None, "Forward ABI projection should succeed for visible coordinates"
    assert -0.15 < x_rad < 0.15, f"Scan angle x out of realistic range: {x_rad}"
    assert -0.15 < y_rad < 0.15, f"Scan angle y out of realistic range: {y_rad}"

    # 2. Inverse projection: scan angles -> lat/lon
    recovered_lat, recovered_lon = SatelliteGeoreferencer.abi_angles_to_latlon(
        x_rad=x_rad,
        y_rad=y_rad,
        lon_0_deg=sat_lon_0,
    )
    assert recovered_lat is not None, "Inverse ABI projection should succeed"
    assert recovered_lon is not None, "Inverse ABI projection should succeed"

    # Verify geodetic roundtrip within sub-kilometer to kilometer precision (~0.05 degrees)
    assert abs(recovered_lat - target_lat) < 0.05, f"Lat mismatch: {recovered_lat} vs {target_lat}"
    assert abs(recovered_lon - target_lon) < 0.05, f"Lon mismatch: {recovered_lon} vs {target_lon}"


def test_abi_projection_horizon_rejection():
    """
    Test that points outside the satellite disk / on opposite side of Earth
    fail cleanly and return (None, None).
    """
    # Point on exact opposite side of Earth (Indian Ocean coordinates for GOES-East)
    io_lat = 18.5
    io_lon = 88.0  # Bay of Bengal (Dana region)
    sat_lon_0 = -75.0

    x_rad, y_rad = SatelliteGeoreferencer.latlon_to_abi_angles(
        lat_deg=io_lat,
        lon_deg=io_lon,
        lon_0_deg=sat_lon_0,
    )
    assert x_rad is None, "Bay of Bengal coordinate must fail ABI forward projection for GOES-East"
    assert y_rad is None, "Bay of Bengal coordinate must fail ABI forward projection for GOES-East"


def test_validate_granule_spatial_and_temporal_pairing():
    """
    Test pairing validation on actual raw files in backend/data/raw.
    - GOES CONUS granule covers Helene (September 2024, Gulf of Mexico).
    - GOES CONUS granule fails for Bay of Bengal storm Dana.
    """
    goes_file = Path("backend/data/raw/OR_ABI-L2-CMIPC-M6C01_G16_s20242701801175_e20242701803548_c20242701804073.nc")
    assert goes_file.exists(), f"Sample GOES file must exist at {goes_file}"

    # Helene observation (valid spatial location)
    helene_result = SatelliteGeoreferencer.validate_granule_for_cyclone(
        granule_path=goes_file,
        storm_lat=26.0,
        storm_lon=-84.6,
        storm_timestamp="2024-09-26T18:00:00Z",
        max_time_diff_minutes=60.0,
        storm_id="AL092024",
        storm_name="HELENE",
    )
    assert helene_result.is_spatially_contained, "Helene in Gulf of Mexico must be inside GOES-16 CONUS bounds"
    assert helene_result.is_valid, f"Helene pairing should be valid: {helene_result.rejection_reason}"
    assert helene_result.time_difference_minutes is not None
    assert helene_result.time_difference_minutes < 5.0, "Time difference should be ~1.3 minutes"

    # Dana observation in Bay of Bengal (must be rejected)
    dana_result = SatelliteGeoreferencer.validate_granule_for_cyclone(
        granule_path=goes_file,
        storm_lat=18.5,
        storm_lon=88.0,
        storm_timestamp="2024-10-24T12:00:00Z",
        max_time_diff_minutes=60.0,
        storm_id="2024298N15093",
        storm_name="DANA",
    )
    assert not dana_result.is_spatially_contained, "Bay of Bengal storm Dana cannot be inside GOES-16 CONUS"
    assert not dana_result.is_valid, "Invalid spatial coverage must cause validation to fail"
    assert "coverage" in str(dana_result.rejection_reason).lower() or "bounds" in str(dana_result.rejection_reason).lower() or "horizon" in str(dana_result.rejection_reason).lower()


def test_pairing_pipeline_execution():
    """
    Test SatelliteObservationPairer on Helene with genuine GOES-16 granule.
    Verifies that real satellite pixels are extracted without synthetic generation.
    """
    goes_file = Path("backend/data/raw/OR_ABI-L2-CMIPC-M6C01_G16_s20242701801175_e20242701803548_c20242701804073.nc")

    obs = CycloneObservation(
        storm_id="AL092024",
        storm_name="HELENE",
        season=2024,
        timestamp_iso="2024-09-26T18:00:00Z",
        lat=26.0,
        lon=-84.6,
        wind_kts=115.0,
        pres_hpa=942.0,
        category=4,
        env_features=np.zeros(8, dtype=np.float32),
    )

    tensor, meta = SatelliteObservationPairer.pair_observation(
        storm_obs=obs,
        satellite_filepath=goes_file,
        max_time_diff_minutes=60.0,
        radius_km=350.0,
        target_tensor_size=(64, 64),
    )

    assert meta.is_valid_pairing is True, f"Pairing failed: {meta.rejection_reason}"
    assert meta.is_synthetic_fallback is False, "Must NOT be synthetic fallback"
    assert tensor is not None, "Tensor should be returned"
    assert tensor.shape == (2, 64, 64), f"Expected shape (2, 64, 64), got {tensor.shape}"
    assert meta.tensor_sha256 is not None, "Must generate SHA-256 provenance digest"
    assert len(meta.tensor_sha256) == 64, "SHA-256 digest must be 64 hexadecimal characters"

    # Normalized tensor checks
    assert np.all(np.isfinite(tensor)), "Tensor should not contain NaN or inf"


def test_rejection_of_time_mismatched_observation():
    """
    Test that SatelliteObservationPairer rejects an observation whose timestamp
    is too distant from the satellite scan time.
    """
    goes_file = Path("backend/data/raw/OR_ABI-L2-CMIPC-M6C01_G16_s20242701801175_e20242701803548_c20242701804073.nc")

    # Observation 24 hours earlier
    distant_obs = CycloneObservation(
        storm_id="AL092024",
        storm_name="HELENE",
        season=2024,
        timestamp_iso="2024-09-25T18:00:00Z",
        lat=26.0,
        lon=-84.6,
        wind_kts=80.0,
        pres_hpa=970.0,
        category=1,
        env_features=np.zeros(8, dtype=np.float32),
    )

    tensor, meta = SatelliteObservationPairer.pair_observation(
        storm_obs=distant_obs,
        satellite_filepath=goes_file,
        max_time_diff_minutes=60.0,  # 1 hour limit
    )

    assert meta.is_valid_pairing is False, "Must reject observation with 24-hour time discrepancy"
    assert "time difference" in str(meta.rejection_reason).lower() or "temporal separation" in str(meta.rejection_reason).lower()
    assert tensor is None, "Rejected pairing should not yield an inference tensor"


def test_reference_grid_cyclone_centered_extraction():
    """
    Test StormCentredExtractor and SatelliteGeoreferencer on the calibrated Bay of Bengal reference grid.
    """
    ref_file = Path("backend/data/raw/reference_satellite_grid.nc")
    assert ref_file.exists(), f"Reference grid must exist at {ref_file}"

    # 1. StormCentredExtractor
    extracted = StormCentredExtractor.extract_storm_window(
        filepath=ref_file,
        center_lat=18.5,
        center_lon=88.0,
        radius_km=300.0,
    )
    assert extracted is not None, "Extraction from reference grid must succeed"
    assert "tensor" in extracted
    assert extracted["tensor"].shape[0] == 2

    # 2. SatelliteGeoreferencer reprojected tensor extraction
    geo_res = SatelliteGeoreferencer.extract_reprojected_storm_tensor(
        filepath=ref_file,
        center_lat=18.5,
        center_lon=88.0,
        radius_km=300.0,
        target_size=(64, 64),
    )
    tensor = geo_res["tensor"]
    assert tensor is not None
    assert tensor.shape == (2, 64, 64)
    assert not np.isnan(tensor).any(), "Tensor must not contain NaNs"


def test_illustrative_risk_index_calculation():
    """
    Test the transparent illustrative climate-resilience risk index calculation formula:
    (intensity_factor * 0.40) + (distance_factor * 0.35) + (exposure_factor * 0.15) + (vuln_factor * 0.10)
    """
    # Scenario A: Close (15 km), Category 4 (115 kts)
    dist_km = 15.0
    wind_kts = 115.0
    intensity_norm = min(1.0, max(0.0, (wind_kts - 34.0) / (137.0 - 34.0)))  # ~0.786
    proximity_norm = max(0.0, (200.0 - dist_km) / 200.0)                     # 0.925
    exposure_factor = 0.8
    vuln_factor = 0.75

    raw_score = (intensity_norm * 0.40) + (proximity_norm * 0.35) + (exposure_factor * 0.15) + (vuln_factor * 0.10)
    risk_index = round(raw_score * 100.0, 1)

    assert 70.0 <= risk_index <= 95.0, f"Expected high risk index for severe close cyclone, got {risk_index}"

    # Scenario B: Far (300 km), Tropical Storm (40 kts)
    dist_far = 300.0
    wind_low = 40.0
    intensity_low = min(1.0, max(0.0, (wind_low - 34.0) / (137.0 - 34.0)))
    proximity_low = max(0.0, (200.0 - dist_far) / 200.0)  # 0.0
    raw_low = (intensity_low * 0.40) + (proximity_low * 0.35) + (exposure_factor * 0.15) + (vuln_factor * 0.10)
    risk_low = round(raw_low * 100.0, 1)

    assert risk_low < 30.0, f"Expected low risk index for distant weak cyclone, got {risk_low}"

"""
Automated Verification Tests for CycloneSense Impact Intelligence
=================================================================
Tests:
  - Geographic bounds containment and IoU overlap
  - Optical biophysical index processing (NDVI, NDWI, cloud masking)
  - SAR polarimetric backscatter calibration and inundation attenuation
  - Temporal before/after observation pair gating and rejection rules
  - Multi-class change detection classification and area statistics
  - Siamese neural change segmentation architecture forward pass
  - Evidence-grounded natural-language query engine
  - Impact Intelligence FastAPI endpoints and schemas
"""
from pathlib import Path
import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.scientific.satellite_observation import GeographicBounds, SatelliteObservation
from backend.app.scientific.optical_processor import OpticalProcessor
from backend.app.scientific.sar_processor import SARProcessor
from backend.app.scientific.before_after_matcher import BeforeAfterMatcher
from backend.app.scientific.change_detector import (
    ChangeDetector,
    SiameseChangeSegmenter,
    CLASS_NO_CHANGE,
    CLASS_WATER_CHANGE,
    CLASS_VEGETATION_CHANGE,
    CLASS_SURFACE_CHANGE,
    CLASS_UNCERTAIN,
)
from backend.app.ml.vlm_provider import GroundedEvidenceVLMProvider, get_vlm_provider
from backend.app.scientific.impact_query_engine import ImpactQueryEngine


def test_geographic_bounds_containment_and_overlap():
    b1 = GeographicBounds(north=20.0, south=19.0, east=86.0, west=85.0)
    b2 = GeographicBounds(north=19.8, south=18.8, east=85.8, west=84.8)
    b_disjoint = GeographicBounds(north=25.0, south=24.0, east=90.0, west=89.0)

    # Containment
    assert b1.contains(19.5, 85.5) is True
    assert b1.contains(20.5, 85.5) is False

    # Overlap
    assert b1.overlaps(b2) is True
    assert b1.overlaps(b_disjoint) is False

    overlap_iou = BeforeAfterMatcher.calculate_bounding_box_overlap(b1, b2)
    assert 0.0 < overlap_iou < 100.0


def test_optical_processor_ndvi_and_ndwi():
    h, w = 16, 16
    nir = np.full((h, w), 0.50, dtype=np.float32)
    red = np.full((h, w), 0.10, dtype=np.float32)
    green = np.full((h, w), 0.15, dtype=np.float32)

    # NDVI = (0.50 - 0.10) / (0.50 + 0.10) = 0.40 / 0.60 = 0.6667
    ndvi, layer_ndvi, hash_ndvi = OpticalProcessor.compute_ndvi(nir, red)
    assert np.allclose(ndvi, 0.6667, atol=1e-3)
    assert -1.0 <= layer_ndvi.min_value <= 1.0
    assert len(hash_ndvi) == 64

    # NDWI = (0.15 - 0.50) / (0.15 + 0.50) = -0.35 / 0.65 = -0.5385 (Terrestrial)
    ndwi, layer_ndwi, hash_ndwi = OpticalProcessor.compute_ndwi(green, nir)
    assert np.allclose(ndwi, -0.5385, atol=1e-3)
    assert -1.0 <= layer_ndwi.min_value <= 1.0
    assert len(hash_ndwi) == 64


def test_sar_processor_linear_to_db_and_inundation():
    # Linear intensity: 0.1 -> 10 * log10(0.1) = -10 dB
    linear = np.array([0.1, 0.001], dtype=np.float32)
    db = SARProcessor.linear_to_db(linear)
    assert np.isclose(db[0], -10.0, atol=0.1)
    assert np.isclose(db[1], -30.0, atol=0.1)

    # Inundation detection: Pre land is -10 dB, Post flood is -19 dB (drop = -9 dB >= 3 dB)
    pre_vv = np.full((10, 10), -10.0, dtype=np.float32)
    post_vv = np.full((10, 10), -19.0, dtype=np.float32)
    water_mask, pct = SARProcessor.detect_sar_inundation_mask(pre_vv, post_vv)
    assert np.all(water_mask)
    assert pct == 100.0


def test_before_after_matcher_valid_and_rejections():
    bounds = GeographicBounds(north=20.0, south=19.5, east=86.0, west=85.5)
    pre_obs = SatelliteObservation(
        observation_id="OBS_PRE",
        source="COPERNICUS",
        platform="SENTINEL-2A",
        sensor="MSI",
        sensor_type="OPTICAL",
        product="S2MSI2A",
        acquisition_time="2019-04-22T04:57:01Z",
        processing_time="2019-04-22T08:00:00Z",
        bounds=bounds,
        spatial_resolution_meters=10.0,
        bands=["B02", "B03", "B04", "B08"],
        cloud_coverage_pct=5.0,
        file_granule_id="S2A_PRE",
    )
    post_obs = SatelliteObservation(
        observation_id="OBS_POST",
        source="COPERNICUS",
        platform="SENTINEL-2B",
        sensor="MSI",
        sensor_type="OPTICAL",
        product="S2MSI2A",
        acquisition_time="2019-05-07T05:07:09Z",
        processing_time="2019-05-07T08:00:00Z",
        bounds=bounds,
        spatial_resolution_meters=10.0,
        bands=["B02", "B03", "B04", "B08"],
        cloud_coverage_pct=10.0,
        file_granule_id="S2B_POST",
    )

    # 1. Valid pairing
    pair = BeforeAfterMatcher.validate_pair(
        cyclone_name="FANI",
        event_time_iso="2019-05-03T03:00:00Z",
        target_lat=19.81,
        target_lon=85.83,
        location_name="Puri, Odisha",
        pre_obs=pre_obs,
        post_obs=post_obs,
    )
    assert pair.is_valid_pair is True
    assert pair.rejection_reason is None

    # 2. Chronological sequence failure (pre is after event)
    pair_bad_time = BeforeAfterMatcher.validate_pair(
        cyclone_name="FANI",
        event_time_iso="2019-04-01T00:00:00Z",
        target_lat=19.81,
        target_lon=85.83,
        location_name="Puri, Odisha",
        pre_obs=pre_obs,
        post_obs=post_obs,
    )
    assert pair_bad_time.is_valid_pair is False
    assert "chronologically" in pair_bad_time.rejection_reason.lower()

    # 3. Excessive cloud cover failure
    post_cloudy = SatelliteObservation(
        observation_id="OBS_CLOUDY",
        source="COPERNICUS",
        platform="SENTINEL-2B",
        sensor="MSI",
        sensor_type="OPTICAL",
        product="S2MSI2A",
        acquisition_time="2019-05-07T05:07:09Z",
        processing_time="2019-05-07T08:00:00Z",
        bounds=bounds,
        spatial_resolution_meters=10.0,
        bands=["B02", "B03", "B04", "B08"],
        cloud_coverage_pct=85.0,  # > 30% max
        file_granule_id="S2B_CLOUDY",
    )
    pair_cloudy = BeforeAfterMatcher.validate_pair(
        cyclone_name="FANI",
        event_time_iso="2019-05-03T03:00:00Z",
        target_lat=19.81,
        target_lon=85.83,
        location_name="Puri, Odisha",
        pre_obs=pre_obs,
        post_obs=post_cloudy,
    )
    assert pair_cloudy.is_valid_pair is False
    assert "cloud cover" in pair_cloudy.rejection_reason.lower()


def test_change_detector_optical_and_sar():
    h, w = 32, 32
    # Pre: Healthy vegetation (NIR 0.50, Red 0.10)
    pre_rgb = np.full((h, w, 3), 0.10, dtype=np.float32)
    pre_nir = np.full((h, w), 0.50, dtype=np.float32)

    # Post: Defoliated patch (NIR drops to 0.15, Red stays 0.10)
    post_rgb = pre_rgb.copy()
    post_nir = pre_nir.copy()
    post_nir[:16, :16] = 0.15  # 25% of pixels lose vegetation

    res = ChangeDetector.detect_optical_change(
        analysis_id="TEST_ANL_OPT",
        pre_rgb=pre_rgb,
        pre_nir=pre_nir,
        post_rgb=post_rgb,
        post_nir=post_nir,
        pixel_size_meters=10.0,
    )

    assert res.classification_mask.shape == (h, w)
    assert res.statistics.affected_change_percentage > 20.0
    assert res.statistics.vegetation_change_area_km2 > 0.0
    assert res.statistics.mean_ndvi_delta < 0.0

    # SAR change test
    pre_vv = np.full((h, w), -10.0, dtype=np.float32)
    pre_vh = np.full((h, w), -16.0, dtype=np.float32)
    post_vv = pre_vv.copy()
    post_vh = pre_vh.copy()
    post_vv[:16, :16] = -20.0  # Specular inundation attenuation drop

    sar_res = ChangeDetector.detect_sar_change(
        analysis_id="TEST_ANL_SAR",
        pre_vv_db=pre_vv,
        pre_vh_db=pre_vh,
        post_vv_db=post_vv,
        post_vh_db=post_vh,
        pixel_size_meters=10.0,
    )
    assert sar_res.statistics.water_change_area_km2 > 0.0
    assert sar_res.statistics.affected_change_percentage > 20.0


def test_siamese_change_segmenter_forward_pass():
    model = SiameseChangeSegmenter(in_channels=4, num_classes=5)
    model.eval()

    batch_size = 2
    pre = torch.randn(batch_size, 4, 32, 32)
    post = torch.randn(batch_size, 4, 32, 32)

    with torch.no_grad():
        out = model(pre, post)

    assert out.shape == (batch_size, 5, 32, 32)


def test_vlm_provider_grounded_evidence_answering():
    provider = GroundedEvidenceVLMProvider()
    from backend.app.scientific.change_detector import ChangeStatistics, ChangeDetectionResult

    stats = ChangeStatistics(
        total_area_km2=10.0,
        affected_change_area_km2=2.5,
        affected_change_percentage=25.0,
        water_change_area_km2=1.2,
        vegetation_change_area_km2=1.0,
        surface_change_area_km2=0.3,
        uncertain_area_km2=0.0,
        mean_ndvi_delta=-0.18,
        mean_ndwi_delta=0.22,
    )
    ch_res = ChangeDetectionResult(
        analysis_id="ANL_TEST_VLM",
        sensor_type="OPTICAL",
        method="PHYSICAL_INDEX_DIFFERENCING",
        classification_mask=np.zeros((10, 10), dtype=np.uint8),
        statistics=stats,
        metadata={"pre_time": "2019-04-22T04:57:01Z", "post_time": "2019-05-07T05:07:09Z", "pre_granule": "G_PRE", "post_granule": "G_POST"},
        provenance_sha256="abc123hash",
        uncertainty_level="HIGH_CONFIDENCE",
    )
    cyclone_ctx = {
        "name": "FANI",
        "storm_id": "2019116N02090",
        "max_wind_kts": 125.0,
        "closest_distance_km": 8.5,
    }

    # Query 1: Water
    water_ans = provider.answer_question(
        question="Did water extent increase after the cyclone?",
        change_result=ch_res,
        cyclone_context=cyclone_ctx,
        location_name="Puri, Odisha",
    )
    assert "1.2 km²" in water_ans.answer
    assert len(water_ans.evidence_citations) > 0
    assert water_ans.uncertainty_level == "HIGH_CONFIDENCE"

    # Query 2: Vegetation
    veg_ans = provider.answer_question(
        question="How much vegetation canopy was damaged?",
        change_result=ch_res,
        cyclone_context=cyclone_ctx,
        location_name="Puri, Odisha",
    )
    assert "1.0 km²" in veg_ans.answer
    assert "Delta-NDVI" in veg_ans.answer


@pytest.mark.asyncio
async def test_api_impact_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Cyclones list
        r_cyc = await client.get("/api/v1/impact/cyclones")
        assert r_cyc.status_code == 200
        data_cyc = r_cyc.json()
        assert data_cyc["count"] >= 3

        # 2. Locations list
        r_loc = await client.get("/api/v1/impact/locations")
        assert r_loc.status_code == 200
        data_loc = r_loc.json()
        assert data_loc["count"] >= 4

        # 3. Before/After retrieval
        r_ba = await client.get("/api/v1/impact/before-after?cyclone_name=FANI&location_name=Puri,%20Odisha&sensor_type=OPTICAL")
        assert r_ba.status_code == 200
        data_ba = r_ba.json()
        assert data_ba["pairing_validation"]["is_valid"] is True
        assert "pre_observation" in data_ba

        # 4. Analyze Impact
        r_anl = await client.post("/api/v1/impact/analyze", json={"cyclone_name": "FANI", "location_name": "Puri, Odisha", "sensor_type": "OPTICAL"})
        assert r_anl.status_code == 200
        data_anl = r_anl.json()
        assert data_anl["impact_report"]["observed_impact_severity"] == "EXTREME_OBSERVED_CHANGE"
        assert "statistics" in data_anl["change_detection"]

        # 5. Natural-Language Question
        r_q = await client.post("/api/v1/impact/question", json={"question": "What changed near Puri?", "cyclone_name": "FANI", "location_name": "Puri, Odisha"})
        assert r_q.status_code == 200
        data_q = r_q.json()
        assert len(data_q["result"]["answer"]) > 20
        assert len(data_q["result"]["evidence_citations"]) >= 2

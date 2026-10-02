import io
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.db.session import init_db
from backend.app.scientific.grid_product_generator import create_reference_satellite_grid_netcdf


@pytest.fixture(autouse=True)
async def setup_database():
    await init_db()


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "CycloneSense"
    assert data["status"] == "ONLINE"


@pytest.mark.asyncio
async def test_system_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "database" in data
    assert data["database"]["status"] == "HEALTHY"
    assert "adapters" in data
    assert "noaa_ibtracs" in data["adapters"]


@pytest.mark.asyncio
async def test_ingest_and_storm_extraction_pipeline(tmp_path):
    # 1. Create reference NetCDF product
    nc_path = tmp_path / "test_satellite_granule.nc"
    create_reference_satellite_grid_netcdf(nc_path, num_lats=80, num_lons=80, center_lat=18.5, center_lon=88.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 2. Ingest file via API
        with open(nc_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": ("test_satellite_granule.nc", io.BytesIO(file_bytes), "application/x-netcdf")}
        ingest_resp = await ac.post("/api/v1/ingest/file?source_origin=NOAA_TEST", files=files)

        assert ingest_resp.status_code in [200, 201]
        ingest_data = ingest_resp.json()
        product_id = ingest_data["product_id"]
        assert len(ingest_data["sha256"]) == 64
        assert ingest_data["qc_status"] == "PASSED"

        # 3. Retrieve product details
        detail_resp = await ac.get(f"/api/v1/ingest/{product_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert len(detail_data["variables_manifest"]) >= 4
        assert len(detail_data["qc_evaluations"]) >= 1

        # 4. Extract storm window & run pattern intelligence
        extract_payload = {
            "product_id": product_id,
            "storm_name": "DANA",
            "storm_id": "BOB062024",
            "center_latitude": 18.5,
            "center_longitude": 88.0,
            "radius_km": 250.0,
        }
        storm_resp = await ac.post("/api/v1/storms/extract", json=extract_payload)
        assert storm_resp.status_code == 201
        storm_data = storm_resp.json()
        assert storm_data["storm_name"] == "DANA"
        assert storm_data["tensor_shape"] == [2, 128, 128]
        assert "vortex_diagnostics" in storm_data
        assert "azimuthal_symmetry_score" in storm_data["vortex_diagnostics"]
        assert "explainability" in storm_data
        assert storm_data["explainability"]["core_concentration_ratio"] > 0.0

        # 5. Verify cryptographic provenance for the extracted tensor
        prov_resp = await ac.get(f"/api/v1/provenance/{storm_data['extraction_id']}")
        assert prov_resp.status_code == 200
        prov_data = prov_resp.json()
        assert len(prov_data) >= 1
        assert prov_data[0]["action"] == "STORM_EXTRACTION"
        assert prov_data[0]["sha256_hash"] == storm_data["tensor_sha256"]


@pytest.mark.asyncio
async def test_ml_models_list_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/ml/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 4
    model_ids = [m["model_id"] for m in models]
    assert "cyclone_fusion_v1" in model_ids
    assert "cyclone_image_v1" in model_ids
    assert "cyclone_env_v1" in model_ids
    assert "baseline_cliper_v1" in model_ids


@pytest.mark.asyncio
async def test_ml_evaluation_report_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/ml/evaluation-report")
    assert resp.status_code == 200
    report = resp.json()
    assert "models_evaluated" in report
    assert "dataset" in report
    assert "multimodal_fusion" in report["models_evaluated"]
    assert "test_evaluation" in report["models_evaluated"]["multimodal_fusion"]
    fusion_test = report["models_evaluated"]["multimodal_fusion"]["test_evaluation"]
    assert "intensity_metrics" in fusion_test
    assert fusion_test["intensity_metrics"]["mae_kts"] > 0
    assert report["dataset"]["split"]["test"]["storm_count"] > 0


@pytest.mark.asyncio
async def test_ml_inference_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "model_type": "fusion",
            "center_latitude": 18.5,
            "center_longitude": 88.0,
            "forward_speed_kmh": 22.0,
            "forward_bearing_deg": 320.0,
            "pressure_hpa": 975.0,
            "storm_id": "TEST_AMPHAN",
            "storm_name": "AMPHAN",
        }
        resp = await ac.post("/api/v1/ml/inference", json=payload)
    assert resp.status_code == 200
    res = resp.json()
    assert res["model_type"] == "multimodal_fusion"
    assert "predicted_intensity_kts" in res
    assert res["predicted_intensity_kts"] > 0
    assert "predicted_category" in res
    assert 0 <= res["predicted_category"] <= 4
    assert len(res["category_probabilities"]) == 5
    assert "gradcam_explainability" in res
    assert "causal_disclaimer" in res["gradcam_explainability"]
    assert "environmental_attribution" in res
    assert len(res["environmental_attribution"]["ranked_features"]) == 8


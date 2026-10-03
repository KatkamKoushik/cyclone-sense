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
        # 1. Test invalid input: multimodal fusion without required satellite product_id must be rejected (HTTP 400)
        invalid_payload = {
            "model_type": "fusion",
            "center_latitude": 18.5,
            "center_longitude": 88.0,
            "forward_speed_kmh": 22.0,
            "forward_bearing_deg": 320.0,
            "pressure_hpa": 975.0,
            "storm_id": "TEST_AMPHAN",
            "storm_name": "AMPHAN",
        }
        rej_resp = await ac.post("/api/v1/ml/inference", json=invalid_payload)
        assert rej_resp.status_code == 400
        assert "requires an authentic 2D satellite product tensor" in rej_resp.json()["detail"]

        # 2. Test valid environmental model inference (only requires environmental covariates, no satellite tensor)
        env_payload = {
            "model_type": "environment",
            "center_latitude": 18.5,
            "center_longitude": 88.0,
            "forward_speed_kmh": 22.0,
            "forward_bearing_deg": 320.0,
            "pressure_hpa": 975.0,
            "storm_id": "TEST_AMPHAN",
            "storm_name": "AMPHAN",
        }
        env_resp = await ac.post("/api/v1/ml/inference", json=env_payload)
        assert env_resp.status_code == 200
        env_res = env_resp.json()
        assert env_res["model_type"] == "environment_only"
        assert env_res["predicted_intensity_kts"] > 0
        assert "environmental_attribution" in env_res

        # 3. Test valid multimodal fusion inference with genuine ingested satellite product
        prod_resp = await ac.get("/api/v1/ingest?limit=1")
        assert prod_resp.status_code == 200
        products = prod_resp.json()
        assert len(products) > 0
        target_product_id = products[0]["id"]

        valid_payload = {
            "model_type": "fusion",
            "product_id": target_product_id,
            "center_latitude": 18.5,
            "center_longitude": 88.0,
            "forward_speed_kmh": 22.0,
            "forward_bearing_deg": 320.0,
            "pressure_hpa": 975.0,
            "storm_id": "TEST_AMPHAN",
            "storm_name": "AMPHAN",
        }
        resp = await ac.post("/api/v1/ml/inference", json=valid_payload)
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


@pytest.mark.asyncio
async def test_storm_catalog_and_track_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        catalog_resp = await ac.get("/api/v1/storms/catalog?limit=10")
        assert catalog_resp.status_code == 200
        cat_data = catalog_resp.json()
        assert "total_storms" in cat_data
        assert cat_data["total_storms"] > 0
        assert len(cat_data["storms"]) <= 10

        sample_sid = cat_data["storms"][0]["storm_id"]
        track_resp = await ac.get(f"/api/v1/storms/track/{sample_sid}")
        assert track_resp.status_code == 200
        track_data = track_resp.json()
        assert track_data["storm_id"] == sample_sid
        assert track_data["total_observations"] > 0
        assert "latitude" in track_data["observations"][0]


@pytest.mark.asyncio
async def test_analysis_job_lifecycle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Get an authentic satellite product
        prod_resp = await ac.get("/api/v1/ingest?limit=1")
        products = prod_resp.json()
        product_id = products[0]["id"] if products else None

        cat_resp = await ac.get("/api/v1/storms/catalog?search=AMPHAN")
        storms = cat_resp.json()["storms"]
        if not storms:
            cat_resp = await ac.get("/api/v1/storms/catalog?limit=1")
            storms = cat_resp.json()["storms"]
        test_storm = storms[0]

        payload = {
            "model_type": "fusion",
            "product_id": product_id,
            "center_latitude": 15.0,
            "center_longitude": 88.0,
            "forward_speed_kmh": 20.0,
            "forward_bearing_deg": 310.0,
            "pressure_hpa": 980.0,
            "storm_id": test_storm["storm_id"],
            "storm_name": test_storm["storm_name"],
        }
        create_resp = await ac.post("/api/v1/ml/jobs", json=payload)
        assert create_resp.status_code == 201
        job_data = create_resp.json()
        assert job_data["status"] == "COMPLETED"
        assert job_data["predicted_intensity_kts"] > 0
        assert job_data["provenance_id"] is not None

        list_resp = await ac.get("/api/v1/ml/jobs?limit=5")
        assert list_resp.status_code == 200
        jobs = list_resp.json()
        assert len(jobs) >= 1
        assert any(j["job_id"] == job_data["job_id"] for j in jobs)

        detail_resp = await ac.get(f"/api/v1/ml/jobs/{job_data['job_id']}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["job_id"] == job_data["job_id"]

        # Cleanup: delete the test-created job so it doesn't pollute the production dashboard
        delete_resp = await ac.delete(f"/api/v1/ml/jobs/{job_data['job_id']}")
        assert delete_resp.status_code in [200, 204, 404]


@pytest.mark.asyncio
async def test_temporal_comparison_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        cat_resp = await ac.get("/api/v1/storms/catalog?limit=5")
        storms = cat_resp.json()["storms"]
        multi_obs_storm = next((s for s in storms if s["obs_count"] >= 2), None)
        assert multi_obs_storm is not None

        # 1. Compare without satellite products -> computes authentic kinematics & track intensity evolution
        payload_track_only = {
            "storm_id": multi_obs_storm["storm_id"],
            "obs_index_t1": 0,
            "obs_index_t2": 1,
        }
        resp = await ac.post("/api/v1/ml/temporal-comparison", json=payload_track_only)
        assert resp.status_code == 200
        data = resp.json()
        assert "translational_motion" in data
        assert "intensity_evolution" in data
        assert data["structural_evolution"] is None  # Accurately reflects no synthetic fallback

        # 2. Compare WITH authentic satellite products -> computes structural differential
        prod_resp = await ac.get("/api/v1/ingest?limit=2")
        products = prod_resp.json()
        if products:
            pid = products[0]["id"]
            payload_with_sat = {
                "storm_id": multi_obs_storm["storm_id"],
                "obs_index_t1": 0,
                "obs_index_t2": 1,
                "product_id_t1": pid,
                "product_id_t2": pid,
            }
            resp_sat = await ac.post("/api/v1/ml/temporal-comparison", json=payload_with_sat)
            assert resp_sat.status_code == 200
            data_sat = resp_sat.json()
            assert data_sat["structural_evolution"] is not None
            assert "delta_eyewall_cooling_kelvin" in data_sat["structural_evolution"]


@pytest.mark.asyncio
async def test_explainability_analyze_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        cat_resp = await ac.get("/api/v1/storms/catalog?limit=1")
        sample_sid = cat_resp.json()["storms"][0]["storm_id"]

        payload = {
            "storm_id": sample_sid,
            "obs_index": 0,
            "target_task": "intensity",
        }
        resp = await ac.post("/api/v1/ml/explainability/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "gradcam" in data
        assert "diagnostic_notes" in data["gradcam"]
        assert "core_concentration_ratio" in data["gradcam"]
        assert "environmental_attribution" in data


@pytest.mark.asyncio
async def test_system_settings_and_adapter_test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        settings_resp = await ac.get("/api/v1/system/settings")
        assert settings_resp.status_code == 200
        s_data = settings_resp.json()
        assert "adapters" in s_data
        assert "noaa_ibtracs" in s_data["adapters"]

        test_resp = await ac.post(
            "/api/v1/system/settings/test-adapter",
            json={"adapter_name": "noaa_ibtracs"},
        )
        assert test_resp.status_code == 200
        assert test_resp.json()["status"] in ["CONNECTED", "ONLINE"]

        # Test NASA adapter
        nasa_test_resp = await ac.post(
            "/api/v1/system/settings/test-adapter",
            json={"adapter_name": "nasa_earthdata"},
        )
        assert nasa_test_resp.status_code == 200
        assert nasa_test_resp.json()["status"] in ["ONLINE", "CONNECTED"]


@pytest.mark.asyncio
async def test_realtime_satellite_search_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Search NOAA GOES
        goes_resp = await ac.get("/api/v1/ingest/realtime/search?source=NOAA_GOES&limit=2")
        assert goes_resp.status_code == 200
        goes_data = goes_resp.json()
        assert goes_data["source"] == "NOAA_GOES"
        assert len(goes_data["granules"]) > 0
        assert "key" in goes_data["granules"][0]

        # 2. Search NASA Earthdata CMR
        nasa_resp = await ac.get("/api/v1/ingest/realtime/search?source=NASA_EARTHDATA&limit=2")
        assert nasa_resp.status_code == 200
        nasa_data = nasa_resp.json()
        assert nasa_data["source"] == "NASA_EARTHDATA"
        assert len(nasa_data["granules"]) > 0
        assert "granule_id" in nasa_data["granules"][0]


@pytest.mark.asyncio
async def test_direct_satellite_granule_inference_pipeline():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Find any ingested scientific product
        prod_resp = await ac.get("/api/v1/ingest?limit=10")
        assert prod_resp.status_code == 200
        products = prod_resp.json()
        if not products:
            pytest.skip("No scientific products ingested in test database")

        target_product = products[0]
        payload = {
            "model_type": "fusion",
            "product_id": target_product["id"],
            "center_latitude": 18.5,
            "center_longitude": -65.0,
            "pressure_hpa": 980.0,
            "forward_speed_kmh": 16.0,
            "forward_bearing_deg": 310.0,
            "storm_name": "TEST_REALTIME",
        }
        job_resp = await ac.post("/api/v1/ml/jobs", json=payload)
        assert job_resp.status_code == 201
        job_data = job_resp.json()
        assert job_data["status"] == "COMPLETED"
        assert job_data["predicted_intensity_kts"] > 0
        assert job_data["provenance_id"] is not None

        # Clean up test job
        await ac.delete(f"/api/v1/ml/jobs/{job_data['job_id']}")



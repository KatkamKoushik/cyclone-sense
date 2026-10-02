import pytest
from pathlib import Path
from backend.app.scientific.reader import ScientificReader
from backend.app.scientific.grid_product_generator import create_reference_satellite_grid_netcdf


@pytest.fixture(scope="module")
def sample_grid_file(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_data")
    file_path = tmp_dir / "test_satellite_grid.nc"
    create_reference_satellite_grid_netcdf(file_path, num_lats=60, num_lons=60, center_lat=20.0, center_lon=85.0)
    return file_path


def test_detect_format(sample_grid_file):
    fmt = ScientificReader.detect_format(sample_grid_file)
    assert fmt == "NETCDF4"


def test_inspect_metadata(sample_grid_file):
    meta = ScientificReader.inspect(sample_grid_file)
    assert meta.file_format == "NETCDF4"
    assert len(meta.sha256_hash) == 64
    assert "lat" in meta.dimensions
    assert "lon" in meta.dimensions
    assert meta.dimensions["lat"] == 60
    assert meta.dimensions["lon"] == 60
    assert len(meta.channels) == 2
    assert meta.spatial_bounds["lat_min"] is not None


def test_read_variable_with_calibration(sample_grid_file):
    data, attrs = ScientificReader.read_variable(sample_grid_file, "clean_ir_brightness_temp")
    assert data.shape == (60, 60)
    assert attrs.get("units") == "K"
    assert not (data < 100.0).any()  # All physical temperatures > 100 K


def test_inspect_real_ibtracs_netcdf():
    ibtracs_path = Path("backend/data/raw/IBTrACS.NI.v04r01.nc")
    if not ibtracs_path.exists():
        pytest.skip("IBTrACS NetCDF file not yet downloaded")
    meta = ScientificReader.inspect(ibtracs_path)
    assert meta.file_format == "NETCDF4"
    assert meta.sha256_hash == "795dc55a5018a13c5f19b78feb1a3e43614c636eb6d5af73069adc140374eb9a"
    assert "storm" in meta.dimensions
    assert len(meta.variables) >= 150

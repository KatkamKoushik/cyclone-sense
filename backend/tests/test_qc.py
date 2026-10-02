import numpy as np
import pytest
from backend.app.scientific.qc import QualityControlEngine, QCStatus


def test_qc_passing_array():
    # Valid brightness temperatures (200 - 300 K)
    arr = np.linspace(200.0, 300.0, 100, dtype=np.float32).reshape(10, 10)
    res = QualityControlEngine.evaluate_array(arr, "brightness_temperature")
    assert res.status == QCStatus.PASSED
    assert res.physical_bounds_passed is True
    assert res.missing_pixel_percentage == 0.0
    assert len(res.anomalies) == 0


def test_qc_physical_bounds_violation():
    # Violates lower bound (e.g. 50 K is impossible on Earth atmospheric satellite IR)
    arr = np.array([50.0, 250.0, 280.0], dtype=np.float32)
    res = QualityControlEngine.evaluate_array(arr, "brightness_temperature")
    assert res.physical_bounds_passed is False
    assert res.status == QCStatus.DEGRADED
    assert any("Physical lower bound violated" in a for a in res.anomalies)


def test_qc_excessive_missing_pixels():
    arr = np.array([250.0, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    res = QualityControlEngine.evaluate_array(arr, "brightness_temperature", max_missing_pct=15.0)
    assert res.status == QCStatus.REJECTED
    assert res.missing_pixel_percentage == 80.0


def test_qc_dqf_mask_evaluation():
    arr = np.full((10, 10), 280.0, dtype=np.float32)
    dqf = np.zeros((10, 10), dtype=np.int16)
    dqf[:3, :] = 1  # 30% non-nominal quality flags
    res = QualityControlEngine.evaluate_array(arr, "clean_ir", dqf_mask=dqf, max_missing_pct=15.0)
    assert res.status == QCStatus.DEGRADED
    assert res.dqf_summary["nominal_good_percentage"] == 70.0

import numpy as np
import pytest
import torch
from backend.app.ml.dataset import CycloneObservation
from backend.app.ml.temporal import TemporalCycloneComparator
from backend.app.ml.models.fusion_model import CycloneFusionModel


def test_temporal_cyclone_comparator():
    obs_t1 = CycloneObservation(
        storm_id="1999298N12093",
        storm_name="ODISHA_SUPER_CYCLONE",
        season=1999,
        timestamp_iso="1999-10-28 00:00:00",
        lat=16.5,
        lon=88.5,
        wind_kts=115.0,
        pres_hpa=930.0,
        category=3,
        env_features=np.random.randn(8).astype(np.float32),
    )

    obs_t2 = CycloneObservation(
        storm_id="1999298N12093",
        storm_name="ODISHA_SUPER_CYCLONE",
        season=1999,
        timestamp_iso="1999-10-28 12:00:00",
        lat=17.8,
        lon=87.2,
        wind_kts=140.0,
        pres_hpa=912.0,
        category=4,
        env_features=np.random.randn(8).astype(np.float32),
    )

    tensor_t1 = np.ones((2, 64, 64), dtype=np.float32)
    tensor_t2 = np.ones((2, 64, 64), dtype=np.float32) * 0.8  # Colder eyewall

    model = CycloneFusionModel(image_channels=2, env_features=8, img_embedding_dim=32, env_embedding_dim=16, fusion_dim=32)

    res = TemporalCycloneComparator.compare_temporal_observations(
        obs_t1=obs_t1,
        obs_t2=obs_t2,
        tensor_t1=tensor_t1,
        tensor_t2=tensor_t2,
        model=model,
    )

    assert res["storm_name"] == "ODISHA_SUPER_CYCLONE"
    assert res["temporal_interval_hours"] == 12.0
    assert res["intensity_evolution"]["delta_wind_true_kts"] == 25.0
    assert res["intensity_evolution"]["rapid_intensification_observed"] is True  # 25 kts / 12h = 2.08 kts/h > 1.25
    assert res["environmental_evolution"]["delta_pressure_hpa"] == -18.0  # Pressure dropped by 18 hPa
    assert res["translational_motion"]["displacement_km"] > 0.0
    assert len(res["structural_evolution"]["diff_grid_sha256"]) == 64

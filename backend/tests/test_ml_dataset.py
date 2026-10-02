from pathlib import Path
import pytest
import numpy as np
import torch
from backend.app.ml.dataset import (
    IBTrACSDatasetBuilder,
    TropicalCycloneDataset,
    create_data_loaders,
    CycloneObservation,
    wind_to_category,
)


@pytest.fixture(scope="module")
def ibtracs_data():
    nc_path = Path("backend/data/raw/IBTrACS.NI.v04r01.nc")
    if not nc_path.exists():
        pytest.skip("IBTrACS dataset not available")
    obs, stats = IBTrACSDatasetBuilder.load_observations(nc_path, min_season=2010)
    return obs, stats


def test_category_mapping():
    assert wind_to_category(25.0) == 0  # Tropical Depression
    assert wind_to_category(45.0) == 1  # Cyclonic Storm
    assert wind_to_category(75.0) == 2  # Very Severe Cyclonic Storm
    assert wind_to_category(105.0) == 3  # Extremely Severe Cyclonic Storm
    assert wind_to_category(130.0) == 4  # Super Cyclonic Storm


def test_dataset_sample_counts_and_missing_data(ibtracs_data):
    obs, stats = ibtracs_data
    assert len(obs) > 500
    assert stats["valid_observations_loaded"] == len(obs)
    assert stats["total_observations_examined"] > len(obs)
    assert stats["missing_wind_observations"] >= 0


def test_leakage_prevention(ibtracs_data):
    obs, _ = ibtracs_data
    train_obs, val_obs, test_obs, split_meta = IBTrACSDatasetBuilder.split_by_storm(
        obs, train_frac=0.7, val_frac=0.15, test_frac=0.15, split_strategy="temporal", seed=42
    )

    # 1. Verify no storm overlap (zero frame-level leakage)
    train_storms = {o.storm_id for o in train_obs}
    val_storms = {o.storm_id for o in val_obs}
    test_storms = {o.storm_id for o in test_obs}

    assert len(train_storms.intersection(val_storms)) == 0
    assert len(train_storms.intersection(test_storms)) == 0
    assert len(val_storms.intersection(test_storms)) == 0

    # 2. Verify temporal leakage prevention (seasons in train <= seasons in test)
    max_train_season = max(split_meta["train"]["seasons"])
    min_test_season = min(split_meta["test"]["seasons"])
    assert max_train_season <= min_test_season


def test_dataloader_batch_shapes(ibtracs_data):
    obs, _ = ibtracs_data
    train_obs, val_obs, test_obs, _ = IBTrACSDatasetBuilder.split_by_storm(obs, seed=42)

    train_loader, _, _ = create_data_loaders(
        train_obs, val_obs, test_obs, batch_size=8, image_shape=(2, 32, 32), seed=42
    )

    batch = next(iter(train_loader))
    assert batch["image"].shape == (8, 2, 32, 32)
    assert batch["environment"].shape == (8, 8)
    assert batch["target_intensity"].shape == (8,)
    assert batch["target_category"].shape == (8,)
    assert not torch.isnan(batch["image"]).any()
    assert not torch.isnan(batch["environment"]).any()

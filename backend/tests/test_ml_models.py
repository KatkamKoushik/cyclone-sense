import numpy as np
import pytest
import torch
from backend.app.ml.models.baseline import BaselineClimatologyPersistenceModel
from backend.app.ml.models.image_model import CycloneImageModel, CycloneImageEncoder
from backend.app.ml.models.env_model import CycloneEnvironmentModel, CycloneEnvironmentEncoder
from backend.app.ml.models.fusion_model import CycloneFusionModel


def test_baseline_cliper_model():
    X = np.random.randn(50, 8).astype(np.float32)
    y_wind = np.random.uniform(20.0, 120.0, 50).astype(np.float32)
    y_cat = np.random.randint(0, 5, 50)

    model = BaselineClimatologyPersistenceModel(alpha=1.0)
    model.fit(X, y_wind, y_cat)

    assert model.is_fitted is True
    pred_w, pred_c = model.predict(X[:10])
    assert pred_w.shape == (10,)
    assert pred_c.shape == (10,)
    assert (pred_w >= 10.0).all()  # Physical lower bound enforced
    assert (pred_c >= 0).all() and (pred_c < 5).all()

    importances = model.get_feature_importances([f"feat_{i}" for i in range(8)])
    assert len(importances) == 8
    assert abs(sum(importances.values()) - 1.0) < 1e-3


def test_image_encoder_and_model_shapes():
    model = CycloneImageModel(in_channels=2, embedding_dim=64, num_classes=5)
    imgs = torch.randn(4, 2, 64, 64)

    out = model(imgs)
    assert out["pred_intensity"].shape == (4,)
    assert out["category_logits"].shape == (4, 5)
    assert out["embedding"].shape == (4, 64)
    assert out["feature_map"].shape[0] == 4
    assert out["feature_map"].shape[1] == 128  # 128 channels in last_conv


def test_env_encoder_and_model_shapes():
    model = CycloneEnvironmentModel(in_features=8, embedding_dim=32, num_classes=5)
    env = torch.randn(4, 8)

    out = model(env)
    assert out["pred_intensity"].shape == (4,)
    assert out["category_logits"].shape == (4, 5)
    assert out["embedding"].shape == (4, 32)


def test_fusion_model_shapes():
    model = CycloneFusionModel(
        image_channels=2,
        env_features=8,
        img_embedding_dim=64,
        env_embedding_dim=32,
        fusion_dim=64,
        num_classes=5,
    )
    imgs = torch.randn(4, 2, 64, 64)
    env = torch.randn(4, 8)

    out = model(imgs, env)
    assert out["pred_intensity"].shape == (4,)
    assert out["category_logits"].shape == (4, 5)
    assert out["pred_evolution"].shape == (4,)
    assert out["z_fused"].shape == (4, 64)
    assert out["feature_map"].shape[0] == 4

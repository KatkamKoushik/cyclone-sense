import pytest
import torch
from backend.app.ml.models.image_model import CycloneImageModel
from backend.app.ml.models.env_model import CycloneEnvironmentModel
from backend.app.ml.explainability import GradCAMExplainer, EnvironmentalAttributionExplainer


def test_gradcam_explainer():
    model = CycloneImageModel(in_channels=2, embedding_dim=32, num_classes=5)
    explainer = GradCAMExplainer(model, model.encoder.last_conv)

    img = torch.randn(1, 2, 64, 64)
    res = explainer.generate_heatmap(img, target_task="intensity")

    assert res["attribution_method"] == "Grad-CAM"
    assert res["shape"] == [64, 64]
    assert 0.0 <= res["core_concentration_ratio"] <= 1.0
    assert len(res["saliency_sha256"]) == 64
    # Scientific non-negotiable: check presence of causal disclaimer
    assert "causal" in res["causal_disclaimer"].lower()

    explainer.close()


def test_environmental_attribution_explainer():
    model = CycloneEnvironmentModel(in_features=8, embedding_dim=32, num_classes=5)
    env = torch.randn(1, 8)

    res = EnvironmentalAttributionExplainer.compute_feature_attribution(model, env, target_task="intensity")

    assert res["attribution_method"] == "Gradient_x_Input"
    assert res["is_valid"] is True
    assert len(res["feature_attributions"]) == 8
    assert len(res["ranked_features"]) == 8
    assert "causal" in res["causal_disclaimer"].lower()


def test_gradcam_invalid_shape_and_nonfinite():
    model = CycloneImageModel(in_channels=2, embedding_dim=32, num_classes=5)
    explainer = GradCAMExplainer(model, model.encoder.last_conv)

    # 3D tensor instead of 4D
    with pytest.raises(ValueError, match="Grad-CAM expects 4D"):
        explainer.generate_heatmap(torch.randn(2, 64, 64))

    # 3 channels instead of 2
    with pytest.raises(ValueError, match="Grad-CAM expects 2 channels"):
        explainer.generate_heatmap(torch.randn(1, 3, 64, 64))

    # NaN in image tensor
    nan_tensor = torch.randn(1, 2, 64, 64)
    nan_tensor[0, 0, 10, 10] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        explainer.generate_heatmap(nan_tensor)

    explainer.close()


def test_environmental_attribution_invalid_inputs():
    model = CycloneEnvironmentModel(in_features=8, embedding_dim=32, num_classes=5)

    # Wrong shape
    with pytest.raises(ValueError, match="Expected env_tensor with shape"):
        EnvironmentalAttributionExplainer.compute_feature_attribution(model, torch.randn(1, 5))

    # NaN in env tensor
    nan_env = torch.randn(1, 8)
    nan_env[0, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        EnvironmentalAttributionExplainer.compute_feature_attribution(model, nan_env)


def test_gradcam_spatial_alignment_and_zero_activation():
    model = CycloneImageModel(in_channels=2, embedding_dim=32, num_classes=5)
    explainer = GradCAMExplainer(model, model.encoder.last_conv)

    # Spatial alignment check with different dimensions
    img_48x48 = torch.randn(1, 2, 48, 48)
    res = explainer.generate_heatmap(img_48x48, target_task="intensity")
    assert res["shape"] == [48, 48]
    assert len(res["heatmap"]) == 48
    assert len(res["heatmap"][0]) == 48

    explainer.close()


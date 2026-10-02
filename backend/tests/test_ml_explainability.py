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
    assert len(res["feature_attributions"]) == 8
    assert len(res["ranked_features"]) == 8
    assert "causal" in res["causal_disclaimer"].lower()

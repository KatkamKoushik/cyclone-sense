import numpy as np
import pytest
import torch
from backend.app.ml.evaluation import EvaluationMetrics, ModelEvaluator
from backend.app.ml.models.env_model import CycloneEnvironmentModel
from backend.app.ml.dataset import CycloneObservation, create_data_loaders


def test_intensity_metrics():
    y_true = np.array([30.0, 50.0, 70.0, 90.0])
    y_pred = np.array([32.0, 48.0, 75.0, 85.0])

    metrics = EvaluationMetrics.compute_intensity_metrics(y_true, y_pred)
    assert metrics["sample_count"] == 4
    assert metrics["mae_kts"] == 3.5  # (|2| + |2| + |5| + |5|) / 4 = 14 / 4 = 3.5
    assert metrics["bias_kts"] == 0.0  # (2 - 2 + 5 - 5) / 4 = 0.0
    assert metrics["rmse_kts"] > 0.0
    assert metrics["pearson_r"] > 0.95


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 3, 4, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 3, 1, 2])  # 6 out of 7 correct

    metrics = EvaluationMetrics.compute_classification_metrics(y_true, y_pred)
    assert metrics["sample_count"] == 7
    assert metrics["accuracy"] == round(6 / 7, 4)
    assert metrics["precision_macro"] > 0.0
    assert metrics["recall_macro"] > 0.0
    assert metrics["f1_macro"] > 0.0


def test_model_evaluator_execution():
    obs = [
        CycloneObservation(
            storm_id=f"S{i % 3}",
            storm_name=f"STORM_{i % 3}",
            season=2020,
            timestamp_iso="2020-05-15 12:00:00",
            lat=15.0 + i * 0.1,
            lon=85.0 + i * 0.1,
            wind_kts=40.0 + (i % 4) * 15.0,
            pres_hpa=990.0,
            category=(i % 4),
            env_features=np.random.randn(8).astype(np.float32),
        )
        for i in range(20)
    ]

    _, _, test_loader = create_data_loaders(obs[:10], obs[10:14], obs[14:], batch_size=4)
    model = CycloneEnvironmentModel(in_features=8, embedding_dim=16, num_classes=5)

    eval_res = ModelEvaluator.evaluate_pytorch_model(model, test_loader, model_type="environment")
    assert eval_res["total_samples_evaluated"] == len(obs[14:])
    assert "mae_kts" in eval_res["intensity_metrics"]
    assert "f1_macro" in eval_res["classification_metrics"]

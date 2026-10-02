import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from backend.app.ml.models.baseline import BaselineClimatologyPersistenceModel
from backend.app.ml.models.image_model import CycloneImageModel
from backend.app.ml.models.env_model import CycloneEnvironmentModel
from backend.app.ml.models.fusion_model import CycloneFusionModel
from backend.app.ml.evaluation import ModelEvaluator, EvaluationMetrics
from backend.app.ml.registry import ModelCheckpointRegistry


class ModelTrainer:
    """
    Executes reproducible, genuine training runs for baseline, image, environment, and fusion models.
    Enforces multi-task loss (Huber loss for continuous wind + CrossEntropy for pattern category).
    Never synthesizes or fabricates validation metrics.
    """

    @classmethod
    def train_baseline_model(
        cls,
        train_obs: List[Any],
        test_obs: List[Any],
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """Fits Baseline CLIPER Ridge model on real training observations and evaluates on test observations."""
        X_train = np.vstack([o.env_features for o in train_obs])
        y_w_train = np.array([o.wind_kts for o in train_obs], dtype=np.float32)
        y_c_train = np.array([o.category for o in train_obs], dtype=np.int64)

        X_test = np.vstack([o.env_features for o in test_obs])
        y_w_test = np.array([o.wind_kts for o in test_obs], dtype=np.float32)
        y_c_test = np.array([o.category for o in test_obs], dtype=np.int64)

        model = BaselineClimatologyPersistenceModel(alpha=alpha)
        model.fit(X_train, y_w_train, y_c_train)

        pred_w, pred_c = model.predict(X_test)
        int_metrics = EvaluationMetrics.compute_intensity_metrics(y_w_test, pred_w)
        cat_metrics = EvaluationMetrics.compute_classification_metrics(y_c_test, pred_c)

        feature_names = [
            "lat", "lon", "coriolis", "dist_equator", "pressure_deficit", "speed", "bearing_cos", "season_phase"
        ]
        importances = model.get_feature_importances(feature_names)

        return {
            "model_type": "baseline_cliper",
            "train_samples": len(train_obs),
            "test_samples": len(test_obs),
            "intensity_metrics": int_metrics,
            "classification_metrics": cat_metrics,
            "feature_importances": importances,
        }

    @classmethod
    def train_pytorch_model(
        cls,
        model: nn.Module,
        model_type: str,  # 'image', 'environment', or 'fusion'
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        model_name: str = "cyclone_model",
        version: str = "1.0.0",
        dataset_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes genuine training loop using Huber loss and CrossEntropy.
        Tracks actual training history and saves checkpoint.
        """
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        criterion_intensity = nn.SmoothL1Loss()  # Huber loss
        criterion_cat = nn.CrossEntropyLoss()

        history: List[Dict[str, Any]] = []
        best_val_mae = float("inf")
        best_weights_path = None
        best_sha256 = ""

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0
            n_train_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                y_wind = batch["target_intensity"].to(device)
                y_cat = batch["target_category"].to(device)

                if model_type == "image":
                    out = model(batch["image"].to(device))
                elif model_type == "environment":
                    out = model(batch["environment"].to(device))
                elif model_type == "fusion":
                    out = model(batch["image"].to(device), batch["environment"].to(device))
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                pred_wind = out["pred_intensity"]
                cat_logits = out["category_logits"]

                loss_int = criterion_intensity(pred_wind, y_wind)
                loss_c = criterion_cat(cat_logits, y_cat)
                loss = loss_int + 0.5 * loss_c

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                train_loss_total += float(loss.item())
                n_train_batches += 1

            scheduler.step()
            train_avg_loss = train_loss_total / max(n_train_batches, 1)

            # Validation
            val_eval = ModelEvaluator.evaluate_pytorch_model(
                model=model, data_loader=val_loader, model_type=model_type, device=device
            )
            val_mae = val_eval["intensity_metrics"]["mae_kts"]
            val_acc = val_eval["classification_metrics"]["accuracy"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_avg_loss, 4),
                "val_mae_kts": val_mae,
                "val_accuracy": val_acc,
            }
            history.append(epoch_record)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                # Save best checkpoint
                best_weights_path, best_sha256 = ModelCheckpointRegistry.save_checkpoint(
                    model=model,
                    model_name=model_name,
                    version=version,
                    config={
                        "model_type": model_type,
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    },
                    metrics=val_eval,
                    dataset_meta=dataset_meta or {},
                    optimizer=optimizer,
                    epoch=epoch,
                )

        elapsed = time.time() - start_time

        return {
            "model_name": model_name,
            "version": version,
            "model_type": model_type,
            "epochs_completed": epochs,
            "elapsed_seconds": round(elapsed, 2),
            "best_val_mae_kts": best_val_mae,
            "checkpoint_path": str(best_weights_path) if best_weights_path else None,
            "checkpoint_sha256": best_sha256,
            "training_history": history,
        }

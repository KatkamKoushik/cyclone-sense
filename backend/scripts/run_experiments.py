import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from backend.app.ml.dataset import IBTrACSDatasetBuilder, create_data_loaders
from backend.app.ml.models.image_model import CycloneImageModel
from backend.app.ml.models.env_model import CycloneEnvironmentModel
from backend.app.ml.models.fusion_model import CycloneFusionModel
from backend.app.ml.training import ModelTrainer
from backend.app.ml.evaluation import ModelEvaluator
from backend.app.ml.explainability import GradCAMExplainer, EnvironmentalAttributionExplainer
from backend.app.ml.temporal import TemporalCycloneComparator


def run_full_suite():
    print("=" * 70)
    print("CycloneSense V3 Real Machine Learning & Evaluation Suite")
    print("=" * 70)

    results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "experiment_results.json"

    # 1. Load authentic observations from IBTrACS
    print("\n[1/5] Loading authentic observations from NOAA IBTrACS archive...")
    nc_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "IBTrACS.NI.v04r01.nc"
    obs, stats = IBTrACSDatasetBuilder.load_observations(nc_path, min_season=2000)
    print(f"Loaded {len(obs)} valid observations (seasons {stats['season_min']} to present).")
    print(f"Examined observations: {stats['total_observations_examined']}, Missing wind: {stats['missing_wind_observations']}.")

    # 2. Split by storm to prevent frame-level and temporal leakage
    print("\n[2/5] Splitting by storm (Temporal Split)...")
    train_obs, val_obs, test_obs, split_meta = IBTrACSDatasetBuilder.split_by_storm(
        obs, train_frac=0.70, val_frac=0.15, test_frac=0.15, split_strategy="temporal", seed=42
    )
    print(f"Train: {len(train_obs)} obs across {split_meta['train']['storm_count']} storms.")
    print(f"Val:   {len(val_obs)} obs across {split_meta['val']['storm_count']} storms.")
    print(f"Test:  {len(test_obs)} obs across {split_meta['test']['storm_count']} storms.")
    print(f"Leakage check: {split_meta['leakage_check_passed']}")

    # 3. Create DataLoaders
    batch_size = 32
    train_loader, val_loader, test_loader = create_data_loaders(
        train_obs, val_obs, test_obs, batch_size=batch_size, image_shape=(2, 64, 64), seed=42
    )

    experiment_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": stats["source_file"],
            "total_observations": len(obs),
            "split": split_meta,
            "missing_stats": stats,
        },
        "models_evaluated": {},
    }

    # Model 1: Baseline CLIPER (Ridge)
    print("\n--- Training Model 1: Baseline CLIPER (Ridge) ---")
    base_results = ModelTrainer.train_baseline_model(train_obs, test_obs, alpha=1.0)
    print("Baseline Test MAE:", base_results["intensity_metrics"]["mae_kts"], "kts")
    print("Baseline Test Macro-F1:", base_results["classification_metrics"]["f1_macro"])
    experiment_report["models_evaluated"]["baseline_cliper"] = base_results

    device = torch.device("cpu")

    # Model 2: Environment-Only MLP
    print("\n--- Training Model 2: Environment-Only Neural Model ---")
    env_model = CycloneEnvironmentModel(in_features=8, embedding_dim=64, num_classes=5)
    env_train_res = ModelTrainer.train_pytorch_model(
        model=env_model,
        model_type="environment",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=4,
        lr=1e-3,
        device=device,
        model_name="cyclone_env",
        version="1.0.0",
        dataset_meta=split_meta,
    )
    # Evaluate on held-out test split
    env_test_eval = ModelEvaluator.evaluate_pytorch_model(env_model, test_loader, model_type="environment", device=device)
    print("Environment Test MAE:", env_test_eval["intensity_metrics"]["mae_kts"], "kts")
    print("Environment Test Macro-F1:", env_test_eval["classification_metrics"]["f1_macro"])
    experiment_report["models_evaluated"]["environment_only"] = {
        "train_meta": env_train_res,
        "test_evaluation": env_test_eval,
    }

    # Model 3: Image-Only CNN
    print("\n--- Training Model 3: Image-Only Convolutional Model ---")
    img_model = CycloneImageModel(in_channels=2, embedding_dim=128, num_classes=5)
    img_train_res = ModelTrainer.train_pytorch_model(
        model=img_model,
        model_type="image",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=8e-4,
        device=device,
        model_name="cyclone_image",
        version="1.0.0",
        dataset_meta=split_meta,
    )
    # Evaluate on held-out test split
    img_test_eval = ModelEvaluator.evaluate_pytorch_model(img_model, test_loader, model_type="image", device=device)
    print("Image Test MAE:", img_test_eval["intensity_metrics"]["mae_kts"], "kts")
    print("Image Test Macro-F1:", img_test_eval["classification_metrics"]["f1_macro"])
    experiment_report["models_evaluated"]["image_only"] = {
        "train_meta": img_train_res,
        "test_evaluation": img_test_eval,
    }

    # Model 4: Multimodal Fusion Model
    print("\n--- Training Model 4: Multimodal Fusion Model ---")
    fusion_model = CycloneFusionModel(
        image_channels=2, env_features=8, img_embedding_dim=128, env_embedding_dim=64, fusion_dim=128, num_classes=5
    )
    fusion_train_res = ModelTrainer.train_pytorch_model(
        model=fusion_model,
        model_type="fusion",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=4,
        lr=8e-4,
        device=device,
        model_name="cyclone_fusion",
        version="1.0.0",
        dataset_meta=split_meta,
    )
    # Evaluate on held-out test split
    fusion_test_eval = ModelEvaluator.evaluate_pytorch_model(fusion_model, test_loader, model_type="fusion", device=device)
    print("Fusion Test MAE:", fusion_test_eval["intensity_metrics"]["mae_kts"], "kts")
    print("Fusion Test Macro-F1:", fusion_test_eval["classification_metrics"]["f1_macro"])
    experiment_report["models_evaluated"]["multimodal_fusion"] = {
        "train_meta": fusion_train_res,
        "test_evaluation": fusion_test_eval,
    }

    # 4. Explainability Analysis (Grad-CAM & Environmental Feature Sensitivity)
    print("\n[4/5] Running Explainability Analysis...")
    sample_batch = next(iter(test_loader))
    sample_img = sample_batch["image"][:1].to(device)
    sample_env = sample_batch["environment"][:1].to(device)

    # Grad-CAM on Image Model
    gradcam = GradCAMExplainer(img_model, img_model.encoder.last_conv)
    gradcam_res = gradcam.generate_heatmap(sample_img, target_task="intensity")
    gradcam.close()
    print("Grad-CAM eyewall core concentration ratio:", gradcam_res["core_concentration_ratio"])
    print("Grad-CAM causal disclaimer:", gradcam_res["causal_disclaimer"])

    # Environmental Attribution on Env Model
    env_attr_res = EnvironmentalAttributionExplainer.compute_feature_attribution(
        env_model, sample_env, target_task="intensity"
    )
    print("Top sensitive environmental covariates:", env_attr_res["ranked_features"][:3])

    experiment_report["explainability_sample"] = {
        "gradcam": {
            "core_concentration_ratio": gradcam_res["core_concentration_ratio"],
            "peak_activation": gradcam_res["peak_activation"],
            "saliency_sha256": gradcam_res["saliency_sha256"],
            "causal_disclaimer": gradcam_res["causal_disclaimer"],
        },
        "environmental_attribution": env_attr_res,
    }

    # 5. Temporal T1 -> T2 Analysis on Actual Storm Observations
    print("\n[5/5] Running T1 -> T2 Temporal Evolution Analysis...")
    # Find two consecutive observations of a storm in test set
    test_storm_id = test_obs[0].storm_id
    storm_observations = [o for o in test_obs if o.storm_id == test_storm_id]

    if len(storm_observations) >= 2:
        obs_1 = storm_observations[0]
        obs_2 = storm_observations[min(1, len(storm_observations) - 1)]

        # Extract tensors
        ds_test = test_loader.dataset
        t1_item = next(item for item in ds_test if item["storm_id"] == obs_1.storm_id and item["timestamp"] == obs_1.timestamp_iso)
        t2_item = next(item for item in ds_test if item["storm_id"] == obs_2.storm_id and item["timestamp"] == obs_2.timestamp_iso)

        t1_tensor = t1_item["image"].numpy()
        t2_tensor = t2_item["image"].numpy()

        temporal_res = TemporalCycloneComparator.compare_temporal_observations(
            obs_t1=obs_1,
            obs_t2=obs_2,
            tensor_t1=t1_tensor,
            tensor_t2=t2_tensor,
            model=fusion_model,
            device=device,
        )

        print(f"Compared {temporal_res['storm_name']} across delta t = {temporal_res['temporal_interval_hours']}h:")
        print(f"  True wind delta: {temporal_res['intensity_evolution']['delta_wind_true_kts']} kts (Rate: {temporal_res['intensity_evolution']['rate_kts_per_hr']} kts/h)")
        if temporal_res["model_predictions"]:
            print(f"  Predicted wind delta: {temporal_res['model_predictions']['delta_pred_wind']} kts")
        print(f"  Eyewall cooling delta: {temporal_res['structural_evolution']['delta_eyewall_cooling_kelvin']} K")
        print(f"  Translation speed: {temporal_res['translational_motion']['speed_kmh']} km/h")

        experiment_report["temporal_comparison_sample"] = temporal_res

    # Save full experimental report
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    print(f"\nExperimental results successfully saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_suite()

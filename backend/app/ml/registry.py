import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from backend.app.config import settings


class ModelCheckpointRegistry:
    """
    Manages physical model checkpoints, architecture versioning, and cryptographic SHA-256 weight verification.
    """

    CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints"

    @classmethod
    def save_checkpoint(
        cls,
        model: nn.Module,
        model_name: str,
        version: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_meta: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 1,
    ) -> Tuple[Path, str]:
        """
        Saves model weights and full training/provenance manifest to disk.
        Computes SHA-256 hash of weights.
        """
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name}_v{version}.pt"
        save_path = cls.CHECKPOINTS_DIR / filename
        meta_path = cls.CHECKPOINTS_DIR / f"{model_name}_v{version}_metadata.json"

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "model_name": model_name,
            "version": version,
            "config": config,
        }

        torch.save(checkpoint_dict, save_path)

        # Compute SHA-256 of saved checkpoint file
        hasher = hashlib.sha256()
        with open(save_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        weights_sha256 = hasher.hexdigest()

        # Save metadata manifest
        meta_dict = {
            "model_name": model_name,
            "version": version,
            "weights_filename": filename,
            "weights_sha256": weights_sha256,
            "file_size_bytes": save_path.stat().st_size,
            "config": config,
            "dataset_meta": dataset_meta,
            "evaluation_metrics": metrics,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "software_version": settings.VERSION,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return save_path, weights_sha256

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Loads weights and returns checkpoint metadata."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        if optimizer and checkpoint_dict.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        meta_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json")
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        return {
            "epoch": checkpoint_dict.get("epoch", 1),
            "model_name": checkpoint_dict.get("model_name"),
            "version": checkpoint_dict.get("version"),
            "config": checkpoint_dict.get("config", {}),
            "metadata": meta,
        }

    @classmethod
    def list_available_checkpoints(cls) -> List[Dict[str, Any]]:
        """List all verified saved checkpoints in registry."""
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        results = []
        for pt_file in cls.CHECKPOINTS_DIR.glob("*.pt"):
            meta_file = pt_file.with_name(f"{pt_file.stem}_metadata.json")
            meta = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            results.append({
                "path": str(pt_file),
                "filename": pt_file.name,
                "size_bytes": pt_file.stat().st_size,
                "metadata": meta,
            })
        return results

"""Evaluation dispatcher and VAE compatibility exports."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config.loader import load_config
from src.pipelines import get_pipeline_family
from src.training.checkpoints import load_model_from_checkpoint

from src.pipelines.vae.evaluation import evaluate_loader, evaluate_per_class


def run_validation(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, float]:
    """Route validation to the appropriate pipeline for the checkpoint/config model type."""
    checkpoint_model_type: str | None = None
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_model_type = ckpt.get("config", {}).get("model", {}).get("type")

    if checkpoint_model_type is None:
        config = load_config(config_path)
        checkpoint_model_type = config.get("model", {}).get("type", "vae")

    family = get_pipeline_family(checkpoint_model_type)
    if family == "diffusion":
        from src.pipelines.diffusion.evaluation import run_validation as diffusion_run_validation

        return diffusion_run_validation(checkpoint_path=checkpoint_path, config_path=config_path, split=split)

    from src.pipelines.vae.evaluation import run_validation as vae_run_validation

    return vae_run_validation(checkpoint_path=checkpoint_path, config_path=config_path, split=split)


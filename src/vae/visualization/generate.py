"""Generation dispatcher for VAE and diffusion pipelines."""

from __future__ import annotations

from pathlib import Path

import torch

from vae.config.loader import load_config
from vae.pipelines import get_pipeline_family


def _resolve_model_type(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("config", {}).get("model", {}).get("type")
    if model_type:
        return model_type
    config = load_config(config_path)
    return config.get("model", {}).get("type", "vae")


def generate_samples(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    num_samples: int = 64,
    output_dir: str | Path | None = None,
) -> Path:
    model_type = _resolve_model_type(checkpoint_path, config_path)
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from vae.pipelines.diffusion.generation import generate_samples as diffusion_generate_samples

        return diffusion_generate_samples(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            num_samples=num_samples,
            output_dir=output_dir,
        )

    from vae.pipelines.vae.generation import generate_samples as vae_generate_samples

    return vae_generate_samples(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        num_samples=num_samples,
        output_dir=output_dir,
    )


def generate_reconstructions(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    num_images: int = 16,
    output_dir: str | Path | None = None,
) -> Path:
    model_type = _resolve_model_type(checkpoint_path, config_path)
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from vae.pipelines.diffusion.generation import generate_reconstructions as diffusion_generate_reconstructions

        return diffusion_generate_reconstructions(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            num_images=num_images,
            output_dir=output_dir,
        )

    from vae.pipelines.vae.generation import generate_reconstructions as vae_generate_reconstructions

    return vae_generate_reconstructions(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        num_images=num_images,
        output_dir=output_dir,
    )


def generate_interpolations(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    n_pairs: int = 4,
    n_steps: int = 10,
    output_dir: str | Path | None = None,
) -> Path:
    model_type = _resolve_model_type(checkpoint_path, config_path)
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from vae.pipelines.diffusion.generation import generate_interpolations as diffusion_generate_interpolations

        return diffusion_generate_interpolations(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            n_pairs=n_pairs,
            n_steps=n_steps,
            output_dir=output_dir,
        )

    from vae.pipelines.vae.generation import generate_interpolations as vae_generate_interpolations

    return vae_generate_interpolations(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        n_pairs=n_pairs,
        n_steps=n_steps,
        output_dir=output_dir,
    )

